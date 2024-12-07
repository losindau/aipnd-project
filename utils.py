# import argparse
import json
import torch
from torchvision import datasets, transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_device_name(use_gpu=True):
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    return device

def get_paths(data_dir = 'flower'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    return [train_dir, valid_dir, test_dir]

def load_cat_to_name(cat_to_name_path="./"):
    with open(str(cat_to_name_path + "cat_to_name.json"), "r") as f:
        labels = json.load(f)

    return labels

def get_transforms_and_loaders(data_dir, batchSize=64):
    train_dir, valid_dir, test_dir = get_paths(data_dir)

    trainingTransforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

    commonValidationTestingTransforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    validationTransforms = transforms.Compose(commonValidationTestingTransforms)
    testingTransforms = transforms.Compose(commonValidationTestingTransforms)

    data_transforms = {'trainingTransforms' : trainingTransforms, 'validationTransforms' : validationTransforms, 'testingTransforms' : testingTransforms}

    image_datasets = {
        'trainData': datasets.ImageFolder(
            train_dir, 
            transform=data_transforms['trainingTransforms']
        ),
        'testData': datasets.ImageFolder(
            valid_dir, 
            transform=data_transforms['validationTransforms']
        ),
        'validData': datasets.ImageFolder(
            test_dir, 
            transform=data_transforms['testingTransforms']
        )
    }


    dataloaders = {
        'trainLoader': torch.utils.data.DataLoader(
            image_datasets['trainData'], 
            batch_size=batchSize, 
            shuffle=True
        ),
        'testLoader': torch.utils.data.DataLoader(
            image_datasets['testData'], 
            batch_size=batchSize, 
            shuffle=True
        ),
        'validLoader': torch.utils.data.DataLoader(
            image_datasets['validData'], 
            batch_size=batchSize, 
            shuffle=True
        )
    }

    return image_datasets, dataloaders