import torch
import argparse
import utils
from torchvision.models import resnet18, ResNet18_Weights, vgg19, VGG19_Weights
from PIL import Image

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict")

    parser.add_argument(
        "image_path",
        type=str,
        default=None,
        help="Image path",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="flower",
        help="Image directory. Example: `flower`",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./",
        help="Checkpoint path. Example: `./`",
    )

    parser.add_argument(
        "--cat_to_name_path",
        type=str,
        default="./",
        help="cat_to_name file path. Example: `./`",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top K classes",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU",
    )

    return parser.parse_args()

def load_checkpoint(device, filepath):
    checkpoint = torch.load(filepath)
    device = checkpoint['device']
    
    if device == "vgg19":
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    elif device == "resnet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("The model architecture must be vgg19 or resnet")

    # Freeze feature extractor layers to focus on the Classifier
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, checkpoint['class_to_idx']

def process_image(image, transforms):
    return transforms(Image.open(image))

def predict(device, cat_to_name, image_path, model, topk=5):
    image = process_image(image_path).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
    
    top_ps, top_classes = probabilities.topk(topk, dim=1)
    
    # Map indices to labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[idx] for idx in top_classes.squeeze(0).tolist()]
    top_flowers = [cat_to_name[str(label)] for label in top_labels]

    return top_ps.squeeze(0).tolist(), top_flowers

def main():
    args = get_input_args()

    print("Get transforms and loaders", end="... ")
    _, _, data_transforms = utils.get_transforms_and_loaders(args.data_dir)
    print("------------------------------------------------------------------")
    
    print("Get device name", end="... ")
    device = utils.get_device_name(args.gpu)
    print("------------------------------------------------------------------")

    print("Load checkpoint", end="... ")
    model, _ = load_checkpoint(args.checkpoint_path)
    model = model.to(device)
    print("------------------------------------------------------------------")

    print("Load labels", end="... ")
    labels = utils.load_cat_to_name(args.cat_to_name_path)
    print("------------------------------------------------------------------")
    
    print("Process image", end="... ")
    image_path = args.image_path
    image = process_image(image_path=image_path, image_transform=data_transforms["validTest"])
    print("------------------------------------------------------------------")

    print("Predict...")
    top_ps, top_labels, top_flowers = predict(model, device, image, labels, args.top_k)

    print("Result:")
    print("Top probabilities: ", top_ps)
    print("Top labels: ", top_labels)
    print("Top flower names: ", top_flowers)

    return 0


if __name__ == "__main__":
    main()