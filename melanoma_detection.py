import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Check if oneDNN is available (mostly useful for CPU acceleration)
if torch.backends.mkldnn.is_available():
    print("oneDNN backend is available")
else:
    print("oneDNN backend is not available")

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations (ResNet input size is 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset from the directory
dataset = datasets.ImageFolder(root='dataset/', transform=transform)

# Set the number of splits (k) for K-Fold cross-validation
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True)

# Initialize lists to store results
fold_train_losses = []
fold_val_losses = []
fold_accuracies = []

# K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
    print(f'\nFold {fold + 1}/{k_folds}')

    # Create data loaders for the current fold
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)

    # Initialize epoch results
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_accuracies = []

    # Define the ResNet model (ResNet50 in this case)
    resnet = models.resnet50(pretrained=True)
    num_classes = len(dataset.classes)
    
    # Freeze the feature extraction layers and only train the final FC layer
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

    # Only allow training of the last layer
    for param in resnet.fc.parameters():
        param.requires_grad = True

    resnet.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop for each fold
    for epoch in range(20):  # Train for 20 epochs
        resnet.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/20], Loss: {avg_train_loss}")

        # Validation loop
        resnet.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect all labels and predictions for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        epoch_val_losses.append(avg_val_loss)
        epoch_accuracies.append(accuracy)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        class_names = dataset.classes
        conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
        
        print(f'Validation Loss: {avg_val_loss}, Accuracy: {accuracy}%')
        print("Confusion Matrix:")
        print(conf_matrix_df)

        # Step the learning rate scheduler
        scheduler.step()

    # Store fold results
    fold_train_losses.append(epoch_train_losses)
    fold_val_losses.append(epoch_val_losses)
    fold_accuracies.append(epoch_accuracies)

# Calculate average metrics across all folds
avg_train_losses = np.mean(fold_train_losses, axis=0)
avg_val_losses = np.mean(fold_val_losses, axis=0)
avg_accuracies = np.mean(fold_accuracies, axis=0)

# Print final average results
final_avg_train_loss = np.mean(avg_train_losses)
final_avg_val_loss = np.mean(avg_val_losses)
final_avg_accuracy = np.mean(avg_accuracies)

print(f'\nFinal Average Training Loss: {final_avg_train_loss}')
print(f'Final Average Validation Loss: {final_avg_val_loss}')
print(f'Final Average Accuracy: {final_avg_accuracy}%')

# Plotting
epochs = range(1, 21)  # Updated to 20 epochs

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, avg_train_losses, label='Train Loss')
plt.plot(epochs, avg_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, avg_accuracies, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch')
plt.legend()

plt.tight_layout()
plt.show()
