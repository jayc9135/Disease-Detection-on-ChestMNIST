import torch
import torch.nn as nn
import torch.optim as optim
import os
from data.get_data import DataLoaders
from model.central_model import CentralCNN
from torchmetrics import Accuracy


def train(model, loader, loss_function, optimizer, accuracy, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # Add up the loss
        running_loss += loss.item()

        # Update the accuracy
        accuracy.update(outputs, labels.int())

    avg_loss = running_loss / len(loader)
    avg_accuracy = accuracy.compute()

    return avg_loss, avg_accuracy

def main(num_epochs=5, learning_rate=0.001, num_classes=14):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function and optimizer
    model = CentralCNN(n_classes = 14).to(device)
    loader = DataLoaders().get_train_loader()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    accuracy = Accuracy(task='multilabel', num_labels=num_classes).to(device)

    saved_model_path = 'cnn_chestmnist.pth'
    if os.path.exists(saved_model_path):
        print(f"Loading weights from {saved_model_path}")
        model.load_state_dict(torch.load(saved_model_path, weights_only=True))

    for epoch in range(num_epochs):
        accuracy.reset()
        train_loss, train_accuracy = train(model, loader, loss_function, optimizer,accuracy, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

        torch.save(model.state_dict(), 'cnn_chestmnist.pth')


if __name__ == '__main__':
    n_epochs = 10
    l_rate = 0.001
    n_classes = 14
    main(num_epochs=n_epochs,
         learning_rate=l_rate,
         num_classes=n_classes)






