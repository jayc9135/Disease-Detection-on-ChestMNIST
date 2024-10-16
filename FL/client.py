import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random
from model.central_model import CentralCNN
from data.get_data import DataLoaders

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
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)  # Ensure labels are Long type

                # Convert one-hot encoded labels to class indices
                if labels.dim() > 1:
                    labels = torch.argmax(labels, dim=1)

                self.optimizer.zero_grad()
                output = self.model(images)  # Output should be raw logits
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
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)  # Ensure labels are Long type

                # Convert one-hot encoded labels to class indices
                if labels.dim() > 1:
                    labels = torch.argmax(labels, dim=1)

                output = self.model(images)  # Output should be raw logits
                loss = self.criterion(output, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        average_loss = total_loss / total  # Average loss per sample
        return average_loss, total, {"accuracy": accuracy, "loss": average_loss, "cid": self.cid}





# Function to start the client, with an option for poisoned data
def start_client(poisoned=False):
    # Create an instance of DataLoaders
    data_loader_instance = DataLoaders()

    # Load data
    train_loader = data_loader_instance.get_train_loader(training_subset_size=1000,
                                                batch_size=10,
                                                poised=False,
                                                flip_ratio=0.75)
    test_loader = data_loader_instance.get_test_loader(testing_subset_size=100,
                                              batch_size=10)

    # Device configuration (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and move it to the device
    model = CentralCNN().to(device)

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