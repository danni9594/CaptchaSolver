import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from captcha_solver import CaptchaCNN, CaptchaDataset
import argparse

def train_model(lr, batch_size, num_epochs):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset (exclude input21.jpg and input100.jpg for testing)
    dataset = CaptchaDataset(
        image_dir="input",
        label_dir="output",
        exclude_indices=[100]  # Exclude input21.jpg and input100.jpg for testing
    )
    
    print(f"Total training samples: {len(dataset)}")
    
    # Create data loader using all training data
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training samples: {len(dataset)}")
    
    # Initialize model
    model = CaptchaCNN(num_classes=36)
    model.to(device)
    
    # Loss function and optimizer (better for ResNet-style architecture)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Print epoch statistics
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%')
    
    # Save the trained model
    torch.save(model.state_dict(), 'captcha_model.pth')
    print("Training completed. Model saved as 'captcha_model.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Captcha-solving CNN model.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for.')
    
    args = parser.parse_args()
    
    train_model(args.lr, args.batch_size, args.epochs)
