# -*- coding: utf-8 -*-
"""Part1_Image_Classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Dx_wCinyT861MvwxB2l5DpxFK4955m92
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Set dataset root directory
dataset_root = './data'

# Create dataset objects
trainset = datasets.CIFAR100(dataset_root, train=True, transform=data_transforms, download=True)
testset = datasets.CIFAR100(dataset_root, train=False, transform=data_transforms, download=True)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Set number of classes
num_cls = 100

# Load VGG16 with pretrained weights
model = models.vgg16(pretrained=True)

# Extract the number of input features for the last fully connected layer
num_in_ftrs = model.classifier[6].in_features

# Replace the last fully connected layer
model.classifier[6] = nn.Linear(num_in_ftrs, num_cls)

# Freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier[6].parameters():
    param.requires_grad = True

# Move model to device
model = model.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Create optimizer
optimizer = optim.SGD(model.classifier[6].parameters(), lr=0.001, momentum=0.9)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Number of epochs
num_epochs = 10

# To track the best model
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Training loop
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            dataloader = trainloader
        else:
            model.eval()   # Set model to evaluation mode
            dataloader = testloader

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Deep copy the model if it's the best one so far
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

print(f'Best val Acc: {best_acc:.4f}')

# Load best model weights
model.load_state_dict(best_model_wts)

# Save the best model
torch.save(model.state_dict(), 'best_model_vgg16.pth')

# Test the model
model.eval()   # Set model to evaluation mode

# Tracking variables
correct = 0
total = 0

# No gradient needed for testing
with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')