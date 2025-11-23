"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Script that includes dataset loading and transformations.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# imports
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main_preprocess():
    """Method used to preprocess images."""
    # -------------------------------
    # TRANSFORMS
    # -------------------------------
    transform_train = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
    ])

    # -------------------------------
    # DATASET LOADING
    # -------------------------------

    train_dir = "data/train"
    test_dir = "data/test"

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform_train
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform_test
    )

    # -------------------------------
    # DATA LOADERS
    # -------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=14,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=12,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------------
    # Check classes
    # -------------------------------

    print("Class mapping:", train_dataset.class_to_idx)
    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    return train_loader, test_loader