import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data_frame.iloc[idx, 1:].values.reshape(28, 28).astype(np.uint8))
        label = int(self.data_frame.iloc[idx, 0])
        if self.transform:
            image = self.transform(image)
        return image, label

# Define CNN model
class Net(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training section
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, trial):
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Evaluate on validation set every epoch
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            val_loss = running_val_loss / len(val_loader.dataset)
            val_accuracy = correct_val / total_val

            # Save model weights if validation accuracy is improved
            if val_accuracy > best_accuracy:
                torch.save(model.state_dict(), f'best_model_{trial.number}.pth')
                best_accuracy = val_accuracy

            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Return the best accuracy for Optuna to maximize
    return best_accuracy

# Testing section
def test_model(model, test_loader, criterion):
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = running_test_loss / len(test_loader.dataset)
    test_accuracy = correct_test / total_test

    print(f'Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy:.4f}')

def objective(trial):
    # Define hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

    # Initialize model, loss and optimizer
    model = Net(dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Create DataLoader with current batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    # Train the model
    best_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, 20, trial)

    # Return the best accuracy as the objective to maximize
    return best_accuracy

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data Preprocessing
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Define dataset and dataloader
    train_path = '/kaggle/input/fashionmnist/fashion-mnist_train.csv'
    test_path = '/kaggle/input/fashionmnist/fashion-mnist_test.csv'

    train_data = MyDataset(csv_file=train_path, transform=train_transform)
    test_data = MyDataset(csv_file=test_path, transform=test_transform)
    test_len = int(len(test_data) * 0.5)
    val_len = len(test_data) - test_len
    test_data, val_data = random_split(test_data, [test_len, val_len])

    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=8)

    best_trial = study.best_trial
    best_model_weights_path = f'best_model_{best_trial.number}.pth'
    print("Parameters: ", best_trial.params)
    print("Validation Accuracy: ", best_trial.value)

    # Initialize best model
    best_model = Net(dropout_rate=best_trial.params['dropout_rate']).to(device)
    best_model.load_state_dict(torch.load(best_model_weights_path))
    
    # Evaluate best model on the test set
    test_model(best_model, test_loader, nn.CrossEntropyLoss())
