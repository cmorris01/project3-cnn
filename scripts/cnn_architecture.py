"""
Utility module for building Convolutional Neural Network (CNN) architectures used in the project.

The helpers below expose ready-to-train networks:
    - baseline_cnn: small and efficient 3-layer CNN suitable for quick training
    - compact_batchnorm_cnn: batchnorm-heavy design for stable training on small/medium datasets
    - pretrained_resnet: wrapper around torchvision ResNet backbones for transfer learning
"""
import torch
from torch import nn
from torchvision import models

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

        # global average pooling layer due to overfitting
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)  # shape: (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # flatten to (batch, 128)
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
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 31 * 31, 256),
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


class PretrainedResNet(nn.Module):
    """
    Wrapper around torchvision's ResNet family with a binary classification head.

    Args:
        model_name: Which ResNet variant to load (resnet18/resnet34/resnet50).
        pretrained: If True, load ImageNet weights before replacing final layer.
        train_backbone: If False, keep backbone frozen and only train new head.
        dropout: Dropout applied before the new classification layer.
    """

    _MODEL_FACTORIES = {
        "resnet18": (models.resnet18, "ResNet18_Weights"),
        "resnet34": (models.resnet34, "ResNet34_Weights"),
        "resnet50": (models.resnet50, "ResNet50_Weights"),
    }

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        train_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        model_name = model_name.lower()
        if model_name not in self._MODEL_FACTORIES:
            available = ", ".join(sorted(self._MODEL_FACTORIES))
            raise ValueError(f"Unsupported ResNet '{model_name}'. Choose from: {available}")

        factory, weight_attr = self._MODEL_FACTORIES[model_name]
        weights_cls = getattr(models, weight_attr, None)
        kwargs = {}
        if weights_cls is not None:
            kwargs["weights"] = weights_cls.DEFAULT if pretrained else None
        else:
            # Fallback to legacy torchvision versions.
            kwargs["pretrained"] = pretrained

        backbone = factory(**kwargs)
        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

        for param in backbone.fc.parameters():
            param.requires_grad = True

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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
