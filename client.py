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
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import flwr
from flwr.client import Client, ClientApp, NumPyClient, start_client
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

from model import Net, test, train
import sys

DEVICE = "cpu"

NUM_CLIENTS = 5
BATCH_SIZE = 32

def load_datasets(partition_id: int):
    """Load CIFAR-10 train and test data."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the full training set
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    # Partition training set based on partition_id
    n_clients = 5  # Adjust based on your needs
    partition_size = len(full_trainset) // n_clients
    lengths = [partition_size] * n_clients
    datasets = random_split(full_trainset, lengths)
    trainset = datasets[partition_id]
    
    # Load test set
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(testset, batch_size=32)
    testloader = DataLoader(testset, batch_size=32)
    
    return trainloader, valloader, testloader

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train_loss, train_accuracy = train(self.net, self.trainloader)
        return get_parameters(self.net), len(self.trainloader), {"train_loss": train_loss, "train_accuracy": train_accuracy}


    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        # Evaluate the model
        val_loss, val_accuracy = test(self.net, self.valloader)

        # Return evaluation metrics
        return float(val_loss), len(self.valloader), {"accuracy": float(val_accuracy), "val_loss": float(val_loss)}
    
# client_fn is used to make a Flower client only upon need and then discard it
def client_fn(partition_id: int) -> flwr.client.Client:
    net = Net().to(DEVICE)

    # Load dataset using partition_id directly
    trainloader, valloader, _ = load_datasets(partition_id)

    # Create and return the Flower client
    return FlowerClient(net, trainloader, valloader).to_client()

if __name__ == "__main__":
    partition_id = int(sys.argv[1])  # Get partition ID from CLI args

    # Start the federated client
    flwr.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=client_fn(partition_id)  # Pass partition ID directly
    )