import flwr as fl
import numpy as np
import csv
import os

# Custom strategy inheriting from FedAvg
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        # Initialize storage for metrics
        self.all_client_losses = {}
        self.all_client_accuracies = {}
        self.avg_losses = []
        self.avg_accuracies = []

    def aggregate_evaluate(self, server_round, results, failures):
        # Extract loss and accuracy from the results
        losses = []
        accuracies = []
        client_ids = []
        for client_id, res in results:
            loss = res.loss
            accuracy = res.metrics["accuracy"]
            cid = res.metrics["cid"]
            losses.append(loss)
            accuracies.append(accuracy)
            client_ids.append(cid)

            # Store per-client metrics
            if cid not in self.all_client_losses:
                self.all_client_losses[cid] = []
                self.all_client_accuracies[cid] = []
            self.all_client_losses[cid].append(loss)
            self.all_client_accuracies[cid].append(accuracy)

        # Calculate average accuracy and loss
        avg_loss = float(np.mean(losses))
        avg_accuracy = float(np.mean(accuracies))

        self.avg_losses.append(avg_loss)
        self.avg_accuracies.append(avg_accuracy)

        # Save metrics to a CSV file
        filename = f'metrics_round_{server_round}.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Client', 'Loss', 'Accuracy'])
            for cid, loss, accuracy in zip(client_ids, losses, accuracies):
                writer.writerow([cid, loss, accuracy])
            writer.writerow(['Average', avg_loss, avg_accuracy])

        # Return aggregated loss and metrics
        return avg_loss, {"accuracy": avg_accuracy}

# Start the server
if __name__ == "__main__":
    # Use the custom strategy
    strategy = CustomFedAvg()

    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),  # Adjust the number of rounds if necessary
    )