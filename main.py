from load import model_predict
from img_show import show_img_and_label
import os 
from PIL import Image

if __name__=="__main__":

    quest_path = os.getcwd() + "/quest/"

    elenco_file = os.listdir(quest_path)

    for image in elenco_file:
        try:
            img = Image.open(quest_path + str(image))
            pred = model_predict(img)
            show_img_and_label(img, f"\nthe class predicted for the image {image} is: {pred}")
        except:
            print(f'\n{image} is not an image or has not jpg format')