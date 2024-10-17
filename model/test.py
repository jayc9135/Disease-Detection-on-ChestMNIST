import torch
import torch.nn as nn
from data.get_data import DataLoaders
from model.central_model import CentralCNN
from torchmetrics.classification import Accuracy

def test(model, loader, loss_function, accuracy, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            running_loss += loss.item()

            accuracy.update(outputs, labels.int())

    avg_loss = running_loss / len(loader)
    avg_accuracy = accuracy.compute()

    return avg_loss, avg_accuracy

def main(n_classes=14):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loaded model
    model = CentralCNN().to(device)
    model.load_state_dict(torch.load('cnn_chestmnist.pth', weights_only=True))
    # Data loader
    loader = DataLoaders().get_test_loader()
    # Loss function
    loss_function = nn.BCELoss()
    # Accuracy metric helper
    accuracy = Accuracy(task='multilabel', num_labels=n_classes).to(device)

    test_loss, test_accuracy = test(model, loader, loss_function, accuracy, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    main()
