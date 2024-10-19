# evaluate.py - Evaluate the Model
import torch
from torch.utils.data import DataLoader
from datasets import CustomCIFAR10, test_transform
from model import ResNet18
import torch.nn as nn

# Define evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)  # Accumulate test loss
            pred = output.argmax(dim=1, keepdim=True)  # Get predictions
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy

# Evaluation script
if __name__ == "__main__":
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Apple Metal Performance Shaders (MPS) if available
    else:
        device = torch.device("cpu")
    
    # Load data using CustomCIFAR10
    test_dataset = CustomCIFAR10(train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Load model
    model = ResNet18().to(device)
    model.load_state_dict(torch.load('./log/cifar10_resnet.pth', map_location=device))
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate the model
    evaluate(model, test_loader, criterion, device)