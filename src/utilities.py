# Utility functions
import cv2
import torch
from PIL import Image
from torchvision.utils import save_image

def read_image(path, size=None, gray=False):
    img = cv2.imread(path)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if size != None:
        img = cv2.resize(img, size)
    return img


def save_tensor_image(img, name, de_normalize=True):

    scale = lambda x : (255 * (x - x.min())) / (x.max() - x.min())
    if de_normalize:
        img = torch.round(torch.dstack((scale(img[0]), scale(img[1]), scale(img[2])))).type(torch.uint8)
        img = Image.fromarray(img.cpu().detach().numpy())
        img.save(name)
    else:
        save_image(img, name)