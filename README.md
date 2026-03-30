# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
In many real-world applications, images are often corrupted by noise due to factors such as low lighting, sensor errors, or transmission disturbances. This noise reduces image quality and makes further processing or analysis difficult.

To address this issue, a convolutional autoencoder model will be developed for image denoising. A convolutional autoencoder is a type of neural network that learns to compress input images into a lower-dimensional representation and then reconstruct them back to their original form.

The model will be trained using pairs of noisy and clean images, allowing it to learn how to remove noise while preserving important features such as edges and textures. Convolutional layers help in effectively capturing spatial patterns in images.

After training, the model will be tested on new noisy images to evaluate its ability to reconstruct clean images. The objective is to improve image quality by minimizing noise while maintaining essential visual details.
## DESIGN STEPS
### Step 1 :Load and Preprocess Data
Load the MNIST dataset and convert images into tensors using normalization.

### Step 2 :Add Noise to Input Images
Apply random noise to the input images using a noise function to simulate corrupted data.

### Step 3 :Initialize Autoencoder Model
Define the encoder (compression) and decoder (reconstruction) using convolutional layers.

### Step 4 :Forward Propagation
Pass the noisy images through the autoencoder to generate denoised output images.

### Step 5 :Compute Loss and Update Weights
Calculate the loss using Mean Squared Error between original and reconstructed images, then update weights using backpropagation and Adam optimizer.

### Step 6 :Evaluate and Visualize Results
Test the model on unseen data and display original, noisy, and denoised images for comparison.

## PROGRAM

### Name:

### Register Number:

```python
# Autoencoder for Image Denoising using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        # Include your code here
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # Include your code here
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Print model summary
print("Name:SURYANARAYANAN T")
print("Register Number:212224040341")
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    # Include your code here
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            #Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader)}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:SURYANARAYANAN T")
    print("Register Number:212224040341")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)


```

### OUTPUT

### Model Summary
<img src="C:\Users\admin\Downloads\Model Summary.png" /><br>
### Training loss
<img src="training loss.png"/><br>
## Original vs Noisy Vs Reconstructed Image
<img src="image.png"/><br>
## RESULT
Thus , convolutional autoencoder for image denoising application is successfully developed
