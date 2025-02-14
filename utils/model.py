import torch.nn as nn
import torch.nn.functional as F
import torch

DEVICE = "cpu"

##CNN model Class
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(net, trainloader):    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    correct, total, train_loss = 0, 0, 0.0

    for batch in trainloader:
        images, labels = batch  
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, labels)

        # Update model parameters
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item()
        total += labels.size(0)
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
    train_loss /= len(trainloader.dataset)
    train_acc = correct / total

    return train_loss, train_acc


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, val_loss = 0, 0, 0.0
    net.eval()
    
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch  
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)

            val_loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
    val_loss /= len(testloader.dataset)
    val_accuracy = correct / total
    return val_loss, val_accuracy


if __name__ == "__main__":
    pass