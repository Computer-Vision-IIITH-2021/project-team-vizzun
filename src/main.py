import sys
import torch
import torch.nn as nn

from PIL import Image

# from EncoderDecoder import EncoderDecoder
from StyleTransfer import ContentStyleLoss, StyleTransfer
from train import trainModel
from utilities import save_tensor_image, processTestImage, NameExtract, Parser, getDataset


parser = Parser()
args = parser.parse_args()

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if (args.action == "run"):
        contentImage = args.content_image
        styleImage = args.style_image

        contName = NameExtract(contentImage)
        styleName = NameExtract(styleImage)

        contentImage = processTestImage(Image.open(contentImage)).to(device)
        styleImage = processTestImage(Image.open(styleImage)).to(device)
        
        model = StyleTransfer(device)
        model.load_state_dict(torch.load(args.model))

        styledImage = model(contentImage, styleImage)[0]
        
        save_tensor_image(styledImage, f"../outputs/{contName}_{styleName}.jpg", False)
        print("Style Transfer completed! Please view", f"../outputs/{contName}_{styleName}.jpg")

    

    elif (args.action == "run_multiple_styles"):
        pass
    
    elif (args.action == "run_alpha"):
        pass