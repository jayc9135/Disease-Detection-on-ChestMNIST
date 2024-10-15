import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random

# Define a simple CNN model for MNIST classification
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define your model layers here
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and preprocess the MNIST dataset
def load_data(poisoned=False, flip_ratio=0.75):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    # Apply data poisoning to the full dataset before splitting
    if poisoned:
        poison_data(full_train_dataset, flip_ratio)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    return train_dataset, val_dataset, test_dataset

# Function to flip labels (data poisoning)
def poison_data(dataset, flip_ratio):
    num_flips = int(len(dataset) * flip_ratio)
    indices_to_flip = random.sample(range(len(dataset)), num_flips)
    for idx in indices_to_flip:
        img, label = dataset[idx]
        new_label = random.choice([l for l in range(10) if l != label])
        dataset.data[idx] = img  # MNIST stores images as 'data'
        dataset.targets[idx] = new_label  # MNIST stores labels as 'targets'

# Define the federated learning client
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, test_loader, device):
        self.cid = cid  # Client ID
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # One local epoch for each round
            for batch in self.train_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        average_loss = total_loss / total  # Average loss per sample
        # Include client ID in the metrics
        return average_loss, total, {"accuracy": accuracy, "loss": average_loss, "cid": self.cid}

# Function to start the client, with an option for poisoned data
def start_client(poisoned=False):
    # Load data
    train_dataset, val_dataset, test_dataset = load_data(poisoned=poisoned, flip_ratio=0.75)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Device configuration (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and move it to the device
    model = CNN().to(device)

    # Generate a client ID
    import sys
    if len(sys.argv) > 1:
        cid = sys.argv[1]
    else:
        cid = "0"

    # Start the federated client
    client = FederatedClient(cid, model, train_loader, test_loader, device)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    # To run a poisoned client, pass 'poisoned' as an argument
    import sys
    poisoned = 'poisoned' in sys.argv
    start_client(poisoned=poisoned)
