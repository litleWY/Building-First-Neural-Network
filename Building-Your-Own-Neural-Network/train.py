# train.py - Train the Model
import os
import torch
import torch.optim as optim
import time
import argparse
from vis import plot_training_curves
from torch.utils.data import DataLoader
from datasets import CustomCIFAR10, train_transform
from model import ResNet18
import torch.nn as nn

# Ensure log directory exists
os.makedirs('./log', exist_ok=True)

# Lists to store training loss and accuracy
train_losses = []
train_accuracies = []

def train(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 99:
                avg_loss = running_loss / total
                accuracy = 100. * correct / total
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
                train_losses.append(avg_loss)
                running_loss = 0.0
        
        epoch_accuracy = 100. * correct / total
        train_accuracies.append(epoch_accuracy)
        end_time = time.time()
        print(f'Epoch [{epoch+1}/{num_epochs}] completed in {end_time - start_time:.2f} seconds with Accuracy: {epoch_accuracy:.2f}%')
        scheduler.step()
    
    # Save the trained model
    torch.save(model.state_dict(), './log/cifar10_resnet.pth')
    
    # Plot training loss and accuracy curves
    plot_training_curves(train_losses, train_accuracies)

# Training script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a ResNet model on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--step_size', type=int, default=5, help='step size for learning rate scheduler (default: 5)')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor (default: 0.5)')
    args = parser.parse_args()

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Apple Metal Performance Shaders (MPS) if available
    else:
        device = torch.device("cpu")
    
    # Load data using CustomCIFAR10
    train_dataset = CustomCIFAR10(train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Load model
    model = ResNet18().to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Train the model
    train(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=args.epochs)