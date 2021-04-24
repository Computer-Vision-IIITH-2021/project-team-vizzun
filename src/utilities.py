# Utility functions
import cv2
import torch
import argparse
import numpy as np
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

scale = lambda x : (255 * (x - x.min())) / (x.max() - x.min())

def save_tensor_image(img, name, de_normalize=True):

    if de_normalize:
        img = torch.round(torch.dstack((scale(img[0]), scale(img[1]), scale(img[2])))).type(torch.uint8)
        img = Image.fromarray(img.cpu().detach().numpy())
        img.save(name)
    else:
        save_image(img, name)

def Parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="Action to perform [run, train, run_mult, run_alpha]")
    parser.add_argument("content_image", help="Path to Content Image or Training images directory")
    parser.add_argument("style_image", help="Path to Style Image(s) or Training images directory")
    parser.add_argument("-e", "--epoch", help="Number of epochs of training", type=int)
    parser.add_argument("--lr", help="Learning rate")
    parser.add_argument("--wd", help="Weight Decay")
    parser.add_argument("--lmbda", help="Lambda value for training loss", type=float, default=10)
    parser.add_argument("--bs", help="Batch Size", type=int, default=64)
    parser.add_argument("--val", help="Validation split fraction", type=float, default=0.2)
    parser.add_argument("--model", help="Path to pretrained model", default="../models/adain_trained")
    parser.add_argument("--alpha", help="Alpha value to control amount of styleTransfer", type=float, default=1)
    parser.add_argument("-w", "--weights", help="Weights for different styles (Style Interpolation)", default=None)
    return parser