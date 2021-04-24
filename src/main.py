import sys
import torch
import torch.nn as nn

from PIL import Image

# from EncoderDecoder import EncoderDecoder
from StyleTransfer import ContentStyleLoss, StyleTransfer
from train import trainModel
from utilities import save_tensor_image, processTestImage, NameExtract, Parser, getDataset


TRAINED_WEIGHTS = "adain_trained"

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

        if args.model:
            TRAINED_WEIGHTS = args.model
        
        model = StyleTransfer(device)
        model.load_state_dict(torch.load(TRAINED_WEIGHTS))

        styledImage = model(contentImage, styleImage)[0]
        
        save_tensor_image(styledImage, f"../outputs/{contName}_{styleName}.jpg")
        save_tensor_image(styledImage, f"../outputs/{contName}_{styleName}_normalized.jpg", False)
        print("Style Transfer completed! Please view", f"../outputs/{contName}_{styleName}.jpg")

    elif (args.action == "train"):

        lmbda = 5 if not args.lmbda else int(args.lmbda)
        model = StyleTransfer(device)
        loss_fn = ContentStyleLoss(lmbda).to(device)

        contentTrainPath = args.content_image
        styleTrainPath = args.style_image

        if args.bs:
            bs = int(args.bs)
        else:
            bs = 64
        
        # if args.lr:
        #     lr = args.lr
        # if args.wd:
        #     wd = args.wd

        model = trainModel(model, loss_fn, *getDataset(contentTrainPath, styleTrainPath, bs=bs), device=device)

    elif (args.action == "run_multiple_styles"):
        pass
    
    elif (args.action == "run_alpha"):
        pass