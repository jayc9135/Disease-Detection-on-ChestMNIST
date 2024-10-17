import torch
import torch.nn as nn


# Define the Neural Network Class
class CentralCNN(nn.Module):
    def __init__(self, n_classes=14):
        super(CentralCNN, self).__init__()

        # Layers in order as that used in network
        # Input Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=64*4*4, out_features=128)

        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=128)

        # Output Layer
        self.fc3 = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        # Conv Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        # Conv Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Conv Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        # Conv Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)

        # Conv Layer 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Flattening Layer
        x = x.view(x.size(0), -1)

        # FC Layer 1
        x = self.fc1(x)
        x = torch.relu(x)

        # Dropout Layer
        x = self.dropout1(x)

        # FC Layer 2
        x = self.fc2(x)
        x = torch.relu(x)

        # Output Layer
        x = self.fc3(x)
        # x = torch.sigmoid(x)
        x = torch.softmax(x, dim=1)

        return x


