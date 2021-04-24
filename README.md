# Arbitrary Style Transfer

## Description
A Pytorch implementation of the 2017 Huang et. al. paper "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" https://arxiv.org/abs/1703.06868 
This Encoder-AdaIN-Decoder architecture - Deep Convolutional Neural Network as a Style Transfer Network (STN) which can receive two arbitrary images as inputs (one as content, the other one as style) and output a generated image that recombines the content and spatial structure from the former and the style (color, texture) from the latter without re-training the network.

## Prerequisites
* Latest version of pytorch,  torchvision and CUDA.
* VGG19 pretrained normalised model
* MS COCO dataset


