"""MNIST classifier for digit recognition from predicted images."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleMNISTClassifier(nn.Module):
    """Simple CNN-based MNIST classifier for 32x32 images."""

    def __init__(self) -> None:
        """Initialize the classifier."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, 1, 32, 32)

        Returns:
            Logits of shape (B, 10)
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 5,
    device: str = "cpu",
) -> nn.Module:
    """Train the MNIST classifier.

    Args:
        model: The classifier model.
        train_loader: Training data loader.
        num_epochs: Number of training epochs.
        device: Device to train on.

    Returns:
        Trained model.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    return model


def create_and_train_classifier(device: str = "cpu") -> nn.Module:
    """Create and train a MNIST classifier.

    Args:
        device: Device to train on.

    Returns:
        Trained classifier model.
    """
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root="/tmp", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create and train model
    model = SimpleMNISTClassifier()
    model = train_classifier(model, train_loader, num_epochs=5, device=device)
    model.eval()

    return model


def recognize_digit(
    model: nn.Module,
    image: torch.Tensor,
    device: str = "cpu",
) -> int:
    """Recognize a digit from an image.

    Args:
        model: Trained MNIST classifier.
        image: Image tensor of shape (1, 32, 32) or (32, 32), values in [0, 1].
        device: Device to run inference on.

    Returns:
        Predicted digit (0-9).
    """
    model.eval()
    model = model.to(device)

    # Ensure image is in correct format
    if image.dim() == 2:
        image = image.unsqueeze(0)  # Add channel dimension
    if image.dim() == 3 and image.shape[0] != 1:
        image = image.unsqueeze(0)  # Add batch dimension

    # Ensure values are in [0, 1] range
    image = torch.clamp(image, 0.0, 1.0)

    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return int(predicted.item())


def load_pretrained_classifier(checkpoint_path: str | None = None, device: str = "cpu") -> nn.Module:
    """Load a pretrained classifier or create and train a new one.

    Args:
        checkpoint_path: Path to checkpoint file. If None, train a new model.
        device: Device to load model on.

    Returns:
        Trained classifier model.
    """
    if checkpoint_path is not None:
        model = SimpleMNISTClassifier()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    # Train a new model
    return create_and_train_classifier(device=device)
