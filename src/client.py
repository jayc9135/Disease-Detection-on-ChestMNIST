import flwr as fl
import torch
import torch.optim as optim
import torch.nn as nn
from model.central_model import CentralCNN
from data.get_data import DataLoaders
from torchmetrics import Accuracy


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, poisoned=False, flip_ratio=0.75):
        # Set client ID
        self.cid = cid

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model initialization
        self.model = CentralCNN(n_classes=14).to(self.device)

        # Data Loaders
        data_loaders = DataLoaders()
        self.train_loader = data_loaders.get_train_loader(
            training_subset_size=1000,
            batch_size=10,
            poisoned=poisoned,
            flip_ratio=flip_ratio
        )
        self.test_loader = data_loaders.get_test_loader()

        # Loss function and optimizer
        self.loss_function = nn.BCELoss()  # Assuming binary cross-entropy loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.accuracy = Accuracy(task='multilabel', num_labels=14).to(self.device)

    def get_parameters(self, config):
        # Get model parameters
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # Set model parameters
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(param) for param in parameters]))
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        # Set model parameters from the server
        self.set_parameters(parameters)

        # # Reset accuracy
        # self.accuracy.reset()

        # Train the model locally
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            self.accuracy.update(outputs, labels.int())

        avg_loss = running_loss / len(self.train_loader)
        avg_accuracy = self.accuracy.compute()

        print(f"Training complete - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        # Return the updated model parameters and metadata
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Set model parameters from the server
        self.set_parameters(parameters)

        # Evaluate the model on the test set
        self.model.eval()
        running_loss = 0.0
        # self.accuracy.reset()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.float().to(self.device)
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                running_loss += loss.item()
                self.accuracy.update(outputs, labels.int())

        avg_loss = running_loss / len(self.test_loader)
        avg_accuracy = self.accuracy.compute()

        print(f"Evaluation complete - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        # Return the evaluation results with the client ID
        return avg_loss, len(self.test_loader.dataset), {"accuracy": avg_accuracy.item(), "loss": avg_loss, "cid": self.cid}


# Start Flower client
if __name__ == "__main__":
    import sys
    poisoned = 'poisoned' in sys.argv

    # Retrieve client ID from command-line arguments or default to "0"
    if len(sys.argv) > 1:
        cid = sys.argv[1]
    else:
        cid = "0"

    client = FlowerClient(cid=cid, poisoned=poisoned)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())
