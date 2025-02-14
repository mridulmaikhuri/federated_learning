from collections import OrderedDict
from typing import List
import numpy as np
import torch
import flwr
from flwr.client import NumPyClient
from utils.model import Net, test, train
import sys
from utils.load_dataset import load_datasets
from config import DEVICE, SERVER_ADDRESS

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net):
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
        val_loss, val_accuracy = test(self.net, self.valloader)
        return float(val_loss), len(self.valloader), {"accuracy": float(val_accuracy), "val_loss": float(val_loss)}
    
# client_fn is used to make a Flower client only upon need and then discard it
def client_fn(partition_id: int):
    net = Net().to(DEVICE)
    trainloader, valloader, _ = load_datasets(partition_id)
    return FlowerClient(net, trainloader, valloader).to_client()

if __name__ == "__main__":
    partition_id = int(sys.argv[1]) 

    # Start the federated client
    flwr.client.start_numpy_client(
        server_address=SERVER_ADDRESS, 
        client=client_fn(partition_id)  
    )