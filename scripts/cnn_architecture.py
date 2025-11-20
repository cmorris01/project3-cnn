"""
Utility module for building Convolutional Neural Network (CNN) architectures used in the project.

The helpers below expose three ready-to-train networks:
    - baseline_cnn: small and efficient 3-layer CNN suitable for quick training
    - deep_feature_cnn: deeper and wider CNN for richer feature extraction
    - compact_batchnorm_cnn: batchnorm-heavy design for stable training on small/medium datasets
"""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """
    Baseline CNN architecture.

    Motivation:
        - Simple, fast, and strong enough for 64x64 or 128x128 images.
        - Uses 3 convolutional layers with increasing channel depths.
        - MaxPooling reduces spatial size while keeping computations low.
        - Good baseline model to debug preprocessing/overfitting.

    Structure:
        Conv(3→32) → ReLU → MaxPool
        Conv(32→64) → ReLU → MaxPool
        Conv(64→128) → ReLU → Flatten → FC layers
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 31 * 31, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeepFeatureCNN(nn.Module):
    """
    Deep feature CNN architecture.

    Motivation:
        - Adds more convolutional layers and larger channel widths.
        - Extracts more complex patterns (edges → shapes → textures).
        - Performs well on larger or more varied datasets.
        - Includes dropout for regularization.

    Structure:
        Conv(3→64) → Conv → MaxPool
        Conv(64→128) → Conv → MaxPool
        Conv(128→256) → Conv → MaxPool
        Fully connected with dropout
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 31 * 31, 256),  # 246,016
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CompactBatchnormCNN(nn.Module):
    """
    Compact CNN leveraging BatchNorm for stability.

    Motivation:
        - BatchNorm allows higher learning rates and faster convergence.
        - Especially good for small datasets or when images vary in lighting.
        - Fewer parameters than DeepFeatureCNN → less overfitting risk.
        - Strong performance + efficient computation.

    Structure:
        Conv → BN → ReLU → Pool  (x3)
        Flatten
        FC layers with BatchNorm
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 62 * 62, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(model, num_epochs, train_dl, valid_dl, optimizer, device, loss_fn):

    # explain this group
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    # what does this do
    for epoch in range(num_epochs):

        # what does this do
        model.train()

        # what does this do
        for x_batch, y_batch in train_dl:
            # what does this do
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # what does this do
            pred = model(x_batch)[:, 0]

            # what does this do
            loss = loss_fn(pred, y_batch.float())

            # what does this do
            loss.backward()

            # what does this do
            optimizer.step()

            # what does this do
            optimizer.zero_grad()

            # what does this group do
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = ((pred>=0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        # what does this group do
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        # what does this do
        model.eval()

        # what does this do
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = ((pred>=0.5).float() == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')

    # what does this function returns (do not just state the variables...describe what they represent like in a docustring)
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid
