#!/usr/bin/env python3
"""
MNIST Neural Network Training Script

A CNN model for MNIST digit classification with data augmentation,
batch normalization, and dropout regularization.

Originally adapted from ERA1S7F10 Colab notebook.
"""

from __future__ import print_function
from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# Constants
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
SEED = 1
DROPOUT_VALUE = 0.1
EPOCHS = 15
ROTATION_DEGREES = (-7.0, 7.0)
BATCH_SIZE_CUDA = 128
BATCH_SIZE_CPU = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SCHEDULER_STEP_SIZE = 5
SCHEDULER_GAMMA = 0.1


def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and testing data transformations.
    
    Returns:
        Tuple containing training transforms and testing transforms
    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation(ROTATION_DEGREES, fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    
    return train_transforms, test_transforms


def get_datasets() -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Load MNIST training and test datasets with transforms applied.
    
    Returns:
        Tuple containing training dataset and test dataset
    """
    train_transforms, test_transforms = get_data_transforms()
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                 transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, download=True, 
                                transform=test_transforms)
    
    return train_dataset, test_dataset

def setup_device() -> torch.device:
    """
    Setup device for training and set random seeds for reproducibility.
    
    Returns:
        torch.device: The device to use for training (cuda or cpu)
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    if cuda_available:
        torch.cuda.manual_seed(SEED)
    
    return device


def get_data_loaders(train_dataset: datasets.MNIST, 
                    test_dataset: datasets.MNIST, 
                    device: torch.device) -> Tuple[torch.utils.data.DataLoader, 
                                                   torch.utils.data.DataLoader]:
    """
    Create training and test data loaders.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset  
        device: Device being used for training
        
    Returns:
        Tuple containing train loader and test loader
    """
    is_cuda = device.type == 'cuda'
    
    dataloader_args = {
        'shuffle': True,
        'batch_size': BATCH_SIZE_CUDA if is_cuda else BATCH_SIZE_CPU,
        'num_workers': NUM_WORKERS if is_cuda else 0,
        'pin_memory': True if is_cuda else False
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    
    return train_loader, test_loader

class MNISTNet(nn.Module):
    """
    CNN model for MNIST digit classification.
    
    Architecture includes:
    - Input block with conv2d, ReLU, BatchNorm, Dropout
    - Convolution blocks with increasing channels
    - Transition block with 1x1 conv and max pooling
    - Global average pooling
    - Final classification layer with log softmax
    """
    
    def __init__(self, dropout_value: float = DROPOUT_VALUE):
        """
        Initialize the network.
        
        Args:
            dropout_value: Dropout probability for regularization
        """
        super(MNISTNet, self).__init__()
        
        # Input Block (28x28 -> 26x26)
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)

def print_model_summary(model: nn.Module, device: torch.device, 
                       input_size: Tuple[int, int, int] = (1, 28, 28)) -> None:
    """
    Print model summary using torchsummary if available.
    
    Args:
        model: The model to summarize
        device: Device the model is on
        input_size: Input tensor dimensions
    """
    try:
        from torchsummary import summary
        print(f"\nModel running on: {device}")
        summary(model, input_size=input_size)
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        print(f"Model: {model}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

class MNISTTrainer:
    """
    Trainer class for MNIST model with tracking of losses and accuracies.
    """
    
    def __init__(self):
        """Initialize trainer with empty metrics lists."""
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
    
    def train_epoch(self, model: nn.Module, device: torch.device, 
                   train_loader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """
        Train the model for one epoch.
        
        Args:
            model: The model to train
            device: Device to train on
            train_loader: Training data loader
            optimizer: Optimizer for training
            epoch: Current epoch number
        """
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        correct = 0
        processed = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = F.nll_loss(output, target)
            self.train_losses.append(loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            
            accuracy = 100.0 * correct / processed

            # Update progress bar
            pbar.set_description(
                f'Epoch {epoch}: Loss={loss.item():.6f} '
                f'Batch={batch_idx} Accuracy={accuracy:.2f}%'
            )
            
        self.train_accuracies.append(accuracy)

    def test(self, model: nn.Module, device: torch.device, 
            test_loader: torch.utils.data.DataLoader) -> float:
        """
        Test the model and return accuracy.
        
        Args:
            model: The model to test
            device: Device to test on
            test_loader: Test data loader
            
        Returns:
            Test accuracy as percentage
        """
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        
        self.test_losses.append(test_loss)
        self.test_accuracies.append(accuracy)

        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({accuracy:.2f}%)\n')

        return accuracy
        
    def plot_metrics(self) -> None:
        """Plot training and testing metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].set_xlabel("Batch")
        axs[0, 0].set_ylabel("Loss")
        
        if len(self.train_accuracies) > 0:
            axs[1, 0].plot(self.train_accuracies)
            axs[1, 0].set_title("Training Accuracy")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Accuracy (%)")
        
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")
        
        axs[1, 1].plot(self.test_accuracies)
        axs[1, 1].set_title("Test Accuracy")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Accuracy (%)")
        
        plt.tight_layout()
        plt.show()

def main() -> None:
    """
    Main function to run MNIST training and testing.
    """
    print("Starting MNIST Training...")
    
    # Setup device and datasets
    device = setup_device()
    train_dataset, test_dataset = get_datasets()
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, device)
    
    # Create model and move to device
    model = MNISTNet().to(device)
    print_model_summary(model, device)
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    
    # # Create trainer
    trainer = MNISTTrainer()
    
    # # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        trainer.train_epoch(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        trainer.test(model, device, test_loader)
    
    # # Plot results
    print("\nTraining completed! Plotting results...")
    trainer.plot_metrics()


if __name__ == "__main__":
    main()

'''
Starting MNIST Training...
Using device: cpu
100%|██████████████████████████████████████████████████████████████████████████| 9.91M/9.91M [00:00<00:00, 13.2MB/s]
100%|███████████████████████████████████████████████████████████████████████████| 28.9k/28.9k [00:00<00:00, 409kB/s]
100%|██████████████████████████████████████████████████████████████████████████| 1.65M/1.65M [00:00<00:00, 4.43MB/s]
100%|██████████████████████████████████████████████████████████████████████████| 4.54k/4.54k [00:00<00:00, 5.76MB/s]

Model running on: cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
       BatchNorm2d-7           [-1, 32, 24, 24]              64
           Dropout-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             320
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11            [-1, 8, 10, 10]             720
             ReLU-12            [-1, 8, 10, 10]               0
      BatchNorm2d-13            [-1, 8, 10, 10]              16
          Dropout-14            [-1, 8, 10, 10]               0
           Conv2d-15              [-1, 8, 8, 8]             576
             ReLU-16              [-1, 8, 8, 8]               0
      BatchNorm2d-17              [-1, 8, 8, 8]              16
          Dropout-18              [-1, 8, 8, 8]               0
           Conv2d-19              [-1, 8, 6, 6]             576
             ReLU-20              [-1, 8, 6, 6]               0
      BatchNorm2d-21              [-1, 8, 6, 6]              16
          Dropout-22              [-1, 8, 6, 6]               0
           Conv2d-23              [-1, 8, 6, 6]             576
             ReLU-24              [-1, 8, 6, 6]               0
      BatchNorm2d-25              [-1, 8, 6, 6]              16
          Dropout-26              [-1, 8, 6, 6]               0
        AvgPool2d-27              [-1, 8, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]              80
================================================================
Total params: 7,760
Trainable params: 7,760
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.01
Params size (MB): 0.03
Estimated Total Size (MB): 1.04
----------------------------------------------------------------

Starting training for 15 epochs...

Epoch 1/15
Epoch 1: Loss=0.114739 Batch=937 Accuracy=86.98%: 100%|███████████████████████████| 938/938 [00:45<00:00, 20.50it/s]

Test set: Average loss: 0.0830, Accuracy: 9775/10000 (97.75%)


Epoch 2/15
Epoch 2: Loss=0.052996 Batch=937 Accuracy=96.31%: 100%|███████████████████████████| 938/938 [00:46<00:00, 20.26it/s]

Test set: Average loss: 0.0527, Accuracy: 9846/10000 (98.46%)


Epoch 3/15
Epoch 3: Loss=0.056292 Batch=937 Accuracy=97.12%: 100%|███████████████████████████| 938/938 [00:47<00:00, 19.89it/s]

Test set: Average loss: 0.0425, Accuracy: 9878/10000 (98.78%)


Epoch 4/15
Epoch 4: Loss=0.020482 Batch=937 Accuracy=97.32%: 100%|███████████████████████████| 938/938 [00:48<00:00, 19.45it/s]

Test set: Average loss: 0.0392, Accuracy: 9879/10000 (98.79%)


Epoch 5/15
Epoch 5: Loss=0.173778 Batch=937 Accuracy=97.59%: 100%|██████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.73it/s]

Test set: Average loss: 0.0512, Accuracy: 9851/10000 (98.51%)


Epoch 6/15
Epoch 6: Loss=0.090312 Batch=937 Accuracy=97.97%: 100%|██████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.83it/s]

Test set: Average loss: 0.0327, Accuracy: 9900/10000 (99.00%)


Epoch 7/15
Epoch 7: Loss=0.195077 Batch=937 Accuracy=98.06%: 100%|██████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.73it/s]

Test set: Average loss: 0.0308, Accuracy: 9909/10000 (99.09%)


Epoch 8/15
Epoch 8: Loss=0.062981 Batch=937 Accuracy=98.26%: 100%|██████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.76it/s]

Test set: Average loss: 0.0300, Accuracy: 9916/10000 (99.16%)


Epoch 9/15
Epoch 9: Loss=0.103134 Batch=937 Accuracy=98.12%: 100%|██████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.55it/s]

Test set: Average loss: 0.0286, Accuracy: 9921/10000 (99.21%)


Epoch 10/15
Epoch 10: Loss=0.038585 Batch=937 Accuracy=98.21%: 100%|█████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.60it/s]

Test set: Average loss: 0.0289, Accuracy: 9915/10000 (99.15%)


Epoch 11/15
Epoch 11: Loss=0.051548 Batch=937 Accuracy=98.17%: 100%|█████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.56it/s]

Test set: Average loss: 0.0298, Accuracy: 9909/10000 (99.09%)


Epoch 12/15
Epoch 12: Loss=0.027728 Batch=937 Accuracy=98.24%: 100%|█████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.69it/s]

Test set: Average loss: 0.0290, Accuracy: 9913/10000 (99.13%)


Epoch 13/15
Epoch 13: Loss=0.063699 Batch=937 Accuracy=98.23%: 100%|█████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.60it/s]

Test set: Average loss: 0.0288, Accuracy: 9919/10000 (99.19%)


Epoch 14/15
Epoch 14: Loss=0.019240 Batch=937 Accuracy=98.26%: 100%|█████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.77it/s]

Test set: Average loss: 0.0292, Accuracy: 9916/10000 (99.16%)


Epoch 15/15
Epoch 15: Loss=0.021340 Batch=937 Accuracy=98.24%: 100%|█████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.56it/s]

Test set: Average loss: 0.0296, Accuracy: 9915/10000 (99.15%)
'''