import torch
import argparse
import utils
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights, vgg19, VGG19_Weights
from collections import OrderedDict

def get_input_args():
    parser = argparse.ArgumentParser(description="Training the network")

    parser.add_argument(
        "data_dir",
        type=str,
        help="Image directory. Example: `flower`",
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Output directory",
    )

    parser.add_argument(
        "--arch",
        choices=["vgg19", "resnet"],
        default="vgg19",
        help="Model architectures available from torchvision.models",
    )

    parser.add_argument(
        "--learning_rate",
        metavar="rate",
        type=float,
        default=0.001,
        help="Learning rate. Example 0.001",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=4096,
        help="Number of hidden units. Example 512",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs. Example 3",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU",
    )

    return parser.parse_args()

def build_model(args):
    
    if args.arch == "vgg19":
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    elif args.arch == "resnet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("The model architecture must be vgg19 or resnet")

    # Freeze feature extractor layers to focus on the Classifier
    for param in model.features.parameters():
        param.requires_grad = False
    
    in_classes = 512 * 7 * 7 if args.arch == "vgg19" else 512
    hidden_units = args.hidden_units
    num_classes = 102
    
    classifier_structure = OrderedDict([
        ('fc1', nn.Linear(in_classes, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, hidden_units / 2)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(hidden_units / 2, num_classes)),
        ('log_softmax', nn.LogSoftmax(dim=1))
    ])
    
    model.classifier = nn.Sequential(classifier_structure)
    
    return model

def validate_model(device, model, criterion, loader):
    model.to(device)
    model.eval()

    running_loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            probabilities = torch.exp(outputs)
            top_p, top_class = probabilities.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += equals.type(torch.FloatTensor).mean().item()

    avg_loss = running_loss / len(loader)
    avg_accuracy = accuracy / len(loader)
    
    return avg_loss, avg_accuracy

def train_model(device, model, optimizer, criterion, train_loader, valid_loader, epochs=3):
    print("Starting training...")
    model.to(device)
    print_step = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        steps = 0

        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_step == 0 or steps == len(train_loader):
                avg_loss = train_loss / steps
                print(f"[Epoch {epoch+1}/{epochs}] Step {steps}/{len(train_loader)}: Train Loss = {avg_loss:.3f}")

        # Validate after each epoch
        valid_loss, accuracy = validate_model(model, criterion, valid_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Validation: Loss = {valid_loss:.3f}, Accuracy = {accuracy*100:.2f}%")

def save_checkpoint(device, model, image_datasets, epochs, learning_rate, optimizer, saveDir, fileName='CMDCheckPoint.pth'):
    model.class_to_idx = image_datasets['trainData'].class_to_idx

    checkpoint_dict = {
        'device': device,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }

    save_path = f"{saveDir}/{fileName}"
    torch.save(checkpoint_dict, save_path)

def main():
    args = get_input_args()

    print("Get transforms and loaders", end="... ")
    image_datasets, dataloaders, _ = utils.get_transforms_and_loaders(args.data_dir)
    print("------------------------------------------------------------------")
    
    print("Get device name", end="... ")
    device = utils.get_device_name(args.gpu)
    print("------------------------------------------------------------------")
    
    print("Build model", end="... ")
    model = build_model(args)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    loss_function = nn.NLLLoss()
    print("------------------------------------------------------------------")
    
    print("Start training...")

    # Train model 
    train_model(device, model, optimizer, loss_function, dataloaders['trainLoader'], dataloaders['validLoader'], args.epochs)

    # Validate model
    print("Start validating...")
    valid_loss, accuracy = validate_model(device, model, loss_function, dataloaders['testLoader'])
    print('[Test result] Valid loss: {:.3f}, Accuracy: {:.3f}'.format(valid_loss, accuracy * 100))
    print("------------------------------------------------------------------")
    
    # Save checkpoint
    print("Save checkpoint", end="... ")
    save_checkpoint(device, model, image_datasets, args.epochs, args.learning_rate, optimizer, args.save_dir)
    print("------------------------------------------------------------------")
    
    print("Training finished")
    return 0


if __name__ == "__main__":
    main()