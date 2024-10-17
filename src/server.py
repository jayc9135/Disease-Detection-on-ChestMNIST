import flwr as fl
import numpy as np
import csv
import argparse
import os

# Custom strategy inheriting from FedAvg
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, attack, fraction_fit, min_fit_clients, min_available_clients):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
        )
        # Initialize storage for metrics
        self.attack = attack
        self.all_client_losses = {}
        self.all_client_accuracies = {}
        self.avg_losses = []
        self.avg_accuracies = []

    def aggregate_evaluate(self, server_round, results, failures):
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

            if cid not in self.all_client_losses:
                self.all_client_losses[cid] = []
                self.all_client_accuracies[cid] = []
            self.all_client_losses[cid].append(loss)
            self.all_client_accuracies[cid].append(accuracy)

        avg_loss = float(np.mean(losses))
        avg_accuracy = float(np.mean(accuracies))

        self.avg_losses.append(avg_loss)
        self.avg_accuracies.append(avg_accuracy)

        # Determine the directory based on the attack parameter
        if self.attack == 'y':
            directory = 'results/metrics/attack'
        elif self.attack == 'n':
            directory = 'results/metrics/no_attack'

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        filename = f'{directory}/metrics_round_{server_round}.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Client', 'Loss', 'Accuracy'])
            for cid, loss, accuracy in zip(client_ids, losses, accuracies):
                writer.writerow([cid, loss, accuracy])
            writer.writerow(['Average', avg_loss, avg_accuracy])

        return avg_loss, {"accuracy": avg_accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Server Configuration")
    parser.add_argument("--attack", type=str, default="n", help="Is there an attack? Yes[y] or No[n]")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients used in each round")
    parser.add_argument("--min_fit_clients", type=int, default=10, help="Minimum number of clients for training")
    parser.add_argument("--min_available_clients", type=int, default=10, help="Minimum number of available clients to start a round")

    args = parser.parse_args()

    strategy = CustomFedAvg(
        attack=args.attack,
        fraction_fit=args.fraction_fit,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )