import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os 
from cnn_model import ConvNet

def model_predict(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carica l'immagine di test

    def grayscale(image):
    # Converti l'immagine in scala di grigi
        return image.convert('L')
    transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.Lambda(lambda x: grayscale(x)),  # Applica la trasformazione in scala di grigi
    transforms.ToTensor(),
])
    image = transform(image)

    classes = ('Fractured', 'Not Fractured')

    PATH = os.getcwd() + '/cnn.pth'
    loaded_model = ConvNet()
    loaded_model.load_state_dict(torch.load(PATH)) # it takes the loaded dictionary, not the path file itself
    loaded_model.to(device)
    loaded_model.eval()

    # Fai una previsione sull'immagine
    with torch.no_grad():
        image = image.unsqueeze(0)  # Aggiungi una dimensione di batch
        output = loaded_model(image)

    # Calcola la classe predetta
    _, predicted_class = torch.max(output, 1)

    # Ottieni il nome della classe predetta
    predicted_class_name = classes[predicted_class.item()]

    return(f'{predicted_class_name}')