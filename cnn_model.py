import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        # Layer convoluzionali
        self.conv1 = nn.Conv2d(1, 32, 5)  # Input: 1 canale (scala di grigi), Output: 32 feature map, Kernel: 5x5
        self.pool = nn.MaxPool2d(2, 2)   # Max pooling con kernel 2x2
        self.conv2 = nn.Conv2d(32, 64, 5) # Input: 32 feature map, Output: 64 feature map, Kernel: 5x5
        self.conv3 = nn.Conv2d(64, 64, 5) # Input: 64 feature map, Output: 64 feature map, Kernel: 5x5
        
        # Layer completamente connessi
        self.fc1 = nn.Linear(36864, 256) # Calcolato in base alle dimensioni delle immagini (227-5+1) / 2 - 5 + 1 = 53
        self.fc2 = nn.Linear(256, num_classes) # Output: Numero di classi specificato come argomento

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Primo strato conv + ReLU + max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Secondo strato conv + ReLU + max pooling
        x = self.pool(F.relu(self.conv3(x)))  # Terzo strato conv + ReLU + max pooling
        x = x.view(x.size(0), -1)            # Flatten senza specificare la dimensione
        x = F.relu(self.fc1(x))              # Primo strato completamente connesso + ReLU
        x = self.fc2(x)                      # Secondo strato completamente connesso
        return x