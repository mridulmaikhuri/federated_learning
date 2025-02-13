import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents, start_server
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

from model import Net, test, train

DEVICE = "cpu"

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

class CustomFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Count the number of participating clients
        num_clients = len(results)
        
        # Store number of clients
        if "num_clients" not in history:
            history["num_clients"] = []
        history["num_clients"].append(num_clients)

        # Extract metrics
        train_losses = [fit_res.metrics["train_loss"] for _, fit_res in results if "train_loss" in fit_res.metrics]
        train_accuracies = [fit_res.metrics["train_accuracy"] for _, fit_res in results if "train_accuracy" in fit_res.metrics]

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies) if train_accuracies else 0

        print(f"Round {server_round}: Clients = {num_clients}, Train Loss = {avg_train_loss}, Train Acc = {avg_train_accuracy}")

        update_history(server_round, avg_train_loss, avg_train_accuracy, None, None)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        num_clients = len(results)  # Clients who participated in evaluation
        history["num_clients"].append(num_clients)

        val_losses = [eval_res.metrics["val_loss"] for _, eval_res in results if "val_loss" in eval_res.metrics]
        val_accuracies = [eval_res.metrics["accuracy"] for _, eval_res in results if "accuracy" in eval_res.metrics]

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0

        print(f"Round {server_round}: Clients = {num_clients}, Val Loss = {avg_val_loss}, Val Acc = {avg_val_accuracy}")

        update_history(server_round, None, None, avg_val_loss, avg_val_accuracy)

        return aggregated_loss, aggregated_metrics
    
def fit_config(server_round: int):
    return {"learning_rate": 0.01 * (0.98 ** server_round)}

from typing import Dict, List, Tuple
from flwr.common import Metrics

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate training metrics from all clients.
    """
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

strategy = CustomFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=5,
    min_evaluate_clients=3,
    min_available_clients=5,
    on_fit_config_fn=fit_config,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
)

def filter_none(x_list, y_list):
    """Helper function to remove None values from lists."""
    filtered_x, filtered_y = zip(*[(x, y) for x, y in zip(x_list, y_list) if y is not None])
    return list(filtered_x), list(filtered_y)

def plot_dashboard(history, num_clients_per_round=None):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # === Loss over Rounds ===
    rounds_loss, train_loss = filter_none(history["round"], history["train_loss"])
    rounds_val_loss, val_loss = filter_none(history["round"], history["val_loss"])
    
    axs[0, 0].plot(rounds_loss, train_loss, label="Train Loss", marker="o")
    axs[0, 0].plot(rounds_val_loss, val_loss, label="Validation Loss", marker="s")
    axs[0, 0].set_title("Loss Over Rounds")
    axs[0, 0].set_xlabel("Round")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    
    # === Accuracy over Rounds ===
    rounds_acc, train_acc = filter_none(history["round"], history["train_accuracy"])
    rounds_val_acc, val_acc = filter_none(history["round"], history["val_accuracy"])
    
    axs[0, 1].plot(rounds_acc, train_acc, label="Train Accuracy", marker="o")
    axs[0, 1].plot(rounds_val_acc, val_acc, label="Validation Accuracy", marker="s")
    axs[0, 1].set_title("Accuracy Over Rounds")
    axs[0, 1].set_xlabel("Round")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend()

    # === Loss Gap (Train - Val) ===
    loss_gap = [tl - vl for tl, vl in zip(train_loss, val_loss)]
    axs[1, 0].plot(rounds_loss, loss_gap, label="Train - Val Loss Gap", marker="o", color="red")
    axs[1, 0].set_title("Train vs Validation Loss Gap")
    axs[1, 0].set_xlabel("Round")
    axs[1, 0].set_ylabel("Loss Gap")
    axs[1, 0].legend()
    
    # === Accuracy Gap (Train - Val) ===
    acc_gap = [ta - va for ta, va in zip(train_acc, val_acc)]
    axs[1, 1].plot(rounds_acc, acc_gap, label="Train - Val Accuracy Gap", marker="o", color="purple")
    axs[1, 1].set_title("Train vs Validation Accuracy Gap")
    axs[1, 1].set_xlabel("Round")
    axs[1, 1].set_ylabel("Accuracy Gap")
    axs[1, 1].legend()
    
    # === Best Round Highlight ===
    best_index = val_acc.index(max(val_acc))
    best_round = rounds_val_acc[best_index]
    axs[0, 2].bar(rounds_val_acc, val_acc, color="gray", alpha=0.6)
    axs[0, 2].bar(best_round, val_acc[best_index], color="green")
    axs[0, 2].set_title(f"Best Round: {best_round}\nVal Accuracy: {val_acc[best_index]:.4f}")
    axs[0, 2].set_xlabel("Round")
    axs[0, 2].set_ylabel("Validation Accuracy")

    # === Clients per Round (if provided) ===
    if num_clients_per_round:
        axs[1, 2].bar(history["round"], num_clients_per_round, color="blue", alpha=0.7)
        axs[1, 2].set_title("Clients Participating Each Round")
        axs[1, 2].set_xlabel("Round")
        axs[1, 2].set_ylabel("Clients")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_server(server_address="0.0.0.0:8080", config=flwr.server.ServerConfig(num_rounds=5), strategy=strategy)
    plot_dashboard(history, num_clients_per_round=history["num_clients"])
