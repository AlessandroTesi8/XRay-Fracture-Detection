import matplotlib.pyplot as plt

def show_img_and_label(img, label):
        # Converti l'immagine in scala di grigi
    img_gray = img.convert('L')
        # Mostra l'immagine
    plt.imshow(img)
    plt.title(f"{label}")
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')  # Rimuovi l'asse delle coordinate
    plt.show()