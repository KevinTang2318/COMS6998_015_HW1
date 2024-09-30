from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Small_Inception(nn.Module):
    def __init__(
        self,
        in_channel: int = 1,
        num_classes: int = 10,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None,
    ) -> None:
        super().__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, Downsample]

        if len(blocks) != 3:
            raise ValueError(f"blocks length should be 3 instead of {len(blocks)}")
        conv_block = blocks[0]
        inception_block = blocks[1]
        downsample_block = blocks[2]

        self.conv1 = conv_block(in_channel, 96, kernel_size=3, stride=1, padding=1)
        self.inception1 = inception_block(96, 32, 32)
        self.inception2 = inception_block(64, 32, 48)
        self.downsample1 = downsample_block(80, 80)

        self.inception3 = inception_block(160, 112, 48)
        self.inception4 = inception_block(160, 96, 64)
        self.inception5 = inception_block(160, 80, 80)
        self.inception6 = inception_block(160, 48, 96)
        self.downsample2 = downsample_block(144, 96)

        self.inception7 = inception_block(240, 176, 160)
        self.inception8 = inception_block(336, 176, 160)

        self.mean_pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Linear(16464, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 1 x 28 x 28
        x = self.conv1(x)
        # N x 96 x 28 * 28
        x = self.inception1(x)
        # N x 64 x 28 x 28
        x = self.inception2(x)
        # N x 80 x 28 x 28
        x = self.downsample1(x)

        # N x 160 x 14 x 14
        x = self.inception3(x)
        # N x 160 x 14 x 14
        x = self.inception4(x)
        # N x 160 x 14 x 14
        x = self.inception5(x)
        # N x 160 x 14 x 14
        x = self.inception6(x)
        # N x 144 x 14 x 14
        x = self.downsample2(x)

        # N x 240 x 7 x 7
        x = self.inception7(x)
        # N x 336 x 7 x 7
        x = self.inception8(x)
        # N x 336 x 7 x 7
        x = self.mean_pooling(x)
        # N x 336 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 16464
        x = self.fc(x)
        # N x 10 (num_classes)
        return x

    def forward(self, x: Tensor):
        x = self._forward(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = conv_block(in_channels, ch3x3, kernel_size=3, padding=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        outputs = [branch1, branch2]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch3x3: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch3x3, kernel_size=3, stride=2, padding=1)

        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        outputs = [branch1, branch2]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


def train(min_lr, max_lr, train_loader, device):
    print("Training start")

    model = Small_Inception(in_channel=1, num_classes=10, init_weights=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters())

    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=min_lr,
        max_lr=max_lr,
        mode='exp_range'
    )

    num_epochs = 5
    total_iterations = 0
    iteration_loss = []
    iteration_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            total_iterations += 1
            iteration_loss.append(loss.item())
            iteration_accuracy.append(accuracy)

            # Accumulate the loss
            running_loss += loss.item()

        # Print average loss for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    return total_iterations, iteration_loss, iteration_accuracy


if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()

    # Download and create the training set
    train_dataset = torchvision.datasets.FashionMNIST(
        root='../data',
        train=True,
        transform=transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    total_iterations, iteration_loss, iteration_accuracy = train(1e-6, 1e-4, train_loader, device)

    iterations = np.arange(1, total_iterations + 1)

    # Plot the iteration loss
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(iterations, iteration_loss, marker='o', linestyle='-', linewidth=2, color='tab:blue', label='Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xscale('log')
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Create a secondary y-axis to plot the accuracy
    ax2 = ax1.twinx()
    ax2.plot(iterations, iteration_accuracy, marker='o', linestyle='-', linewidth=2, color='tab:red', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add a title
    plt.title('Training Loss and Accuracy over Iterations')

    # Save the figure
    plt.savefig('q3-2.png', dpi=300, bbox_inches='tight')
    plt.show()