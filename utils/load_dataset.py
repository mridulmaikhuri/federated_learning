import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from config import NUM_CLIENTS as num_clients, BATCH_SIZE as batch_size, TRANSFORM as transform 

def partition_dataset(train_set, val_ratio=0.1):
    indices = np.arange(len(train_set))
    np.random.shuffle(indices) 
    client_data = np.array_split(indices, num_clients)

    client_partitions = []
    for client_indices in client_data:
        train_indices, val_indices = train_test_split(client_indices, test_size=val_ratio, random_state=42)
        train_subset = Subset(train_set, train_indices)
        val_subset = Subset(train_set, val_indices)
        client_partitions.append((train_subset, val_subset))

    return client_partitions

def load_datasets(partition_id: int):
    # Load training set
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Divide train_set into num_clients parts
    client_partitions = partition_dataset(train_set)
    train_set, val_set = client_partitions[partition_id]  

    # Load test set (same for all clients)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader

if __name__ == "__main__":
    pass
