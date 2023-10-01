import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from cnn_model import ConvNet
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 20
batch_size = 32
learning_rate = 0.001

#import databases
path = os.getcwd()

def grayscale(image):
    # Converti l'immagine in scala di grigi
    return image.convert('L')

# Definisci la trasformazione per convertire in scala di grigi
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.Lambda(lambda x: grayscale(x)),  # Applica la trasformazione in scala di grigi
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder( path + "/datasets/train",
                                             transform=transform)

test_dataset= torchvision.datasets.ImageFolder( path + "/datasets/val",
                                             transform=transform)


classes = train_dataset.classes


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def imshow(imgs):
    # Sposta il tensore in numpy e converte in formato immagine
    npimgs = imgs.numpy()

    # Trasponi il tensore (canale, altezza, larghezza) in (altezza, larghezza, canale)
    plt.imshow(np.transpose(npimgs, (1, 2, 0)), cmap='gray')
    plt.show()

# one batch of random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images[0:25], nrow=5)

imshow(img_grid)

# Creazione del modello
model = ConvNet(num_classes=len(classes)).to(device)

# Definizione della funzione di perdita e dell'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Numero totale di passaggi
n_total_steps = len(train_loader)

for epoch in range(num_epochs):

    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    print(f'[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}')

print('Finished Training')

PATH = 'cnn.pth'
#torch.save(model.state_dict(), PATH)

loaded_model = ConvNet()
loaded_model.load_state_dict(torch.load(PATH)) # it takes the loaded dictionary, not the path file itself
loaded_model.to(device)
loaded_model.eval()

with torch.no_grad():
    n_correct = 0
    n_correct2 = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        outputs2 = loaded_model(images)
        _, predicted2 = torch.max(outputs2, 1)
        n_correct2 += (predicted2 == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the model: {acc} %')

    acc = 100.0 * n_correct2 / n_samples
    print(f'Accuracy of the loaded model: {acc} %')