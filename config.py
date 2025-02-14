import torch
import torchvision.transforms as transforms
from typing import List, Tuple
from flwr.common import Metrics

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 5
NUM_ROUNDS = 5
BATCH_SIZE = 32
SERVER_ADDRESS = "127.0.0.1:8080"

fraction_fit=1.0
fraction_evaluate=1.0
min_fit_clients=5
min_evaluate_clients=3
min_available_clients=5

def fit_config(server_round):
    return {"epochs": 10, "batch_size": 32, "lr": 0.01 * (0.95 ** server_round)}

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Initialize sums
    train_loss_sum = 0.0
    train_accuracy_sum = 0.0
    num_samples = len(metrics)

    # Sum metrics from all clients
    for _, client_metrics in metrics:
        train_loss_sum += client_metrics.get("train_loss", 0.0)
        train_accuracy_sum += client_metrics.get("train_accuracy", 0.0)

    # Calculate averages
    aggregated_metrics = {
        "train_loss": train_loss_sum / num_samples,
        "train_accuracy": train_accuracy_sum / num_samples,
    }

    return aggregated_metrics


history = {
    "round": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

# Function to update the history
def update_history(server_round, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy):
    history["round"].append(server_round)
    history["train_loss"].append(avg_train_loss)
    history["train_accuracy"].append(avg_train_accuracy)
    history["val_loss"].append(avg_val_loss)
    history["val_accuracy"].append(avg_val_accuracy)