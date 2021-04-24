import sys
import torch
import torch.nn as nn

from PIL import Image

# from EncoderDecoder import EncoderDecoder
from StyleTransfer import ContentStyleLoss, StyleTransfer, StyleTransferInterpolation
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

        styledImage = model(contentImage, styleImage, args.alpha)[0]
        
        save_tensor_image(styledImage, f"../outputs/{contName}_{styleName}.jpg", False)
        print("Style Transfer completed! Please view", f"../outputs/{contName}_{styleName}.jpg")

    elif (args.action == "train"):

        lmbda = 5 if not args.lmbda else int(args.lmbda)
        model = StyleTransfer(device)
        loss_fn = ContentStyleLoss(lmbda).to(device)

        contentTrainPath = args.content_image
        styleTrainPath = args.style_image

        
        # if args.lr:
        #     lr = args.lr
        # if args.wd:
        #     wd = args.wd

        model = trainModel(model, loss_fn, *getDataset(contentTrainPath, styleTrainPath, val=args.val, bs=args.bs), device=device)

    elif (args.action == "run_multiple_styles"):
        contentImage = args.content_image
        styleImages = args.style_image.split(',')

        contName = NameExtract(contentImage)

        if args.weights is None:
            weights = [1/len(styleImages) for _ in range(len(styleImages))]
        else:
            weights = args.weights.split(',')
            weights = [float(i) for i in weights]

        styleName = "_".join([NameExtract(i) + f"_{j}" for i,j in zip(styleImages, weights)])

        contentImage = processTestImage(Image.open(contentImage)).to(device)
        styleImages = [processTestImage(Image.open(i)).to(device) for i in styleImages]

        model = StyleTransferInterpolation(device)
        model.load_state_dict(torch.load(args.model))

        styledImage = model(contentImage, styleImages, weights, args.alpha)
        save_tensor_image(styledImage, f"../outputs/{contName}_{styleName}.jpg", False)
        print("Style Transfer completed! Please view", f"../outputs/{contName}_{styleName}.jpg")
    
        