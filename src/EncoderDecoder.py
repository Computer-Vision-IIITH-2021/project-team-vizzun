import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder    
        self.style_features = []
        self.style_layers = [1, 6, 11, 20] # relu1_1, relu2_1, relu3_1, relu4_1
        for i in self.style_layers:
            self.encoder._modules[str(i)].register_forward_hook(self.style_feature_hook)

    def style_feature_hook(self, module, input, output):
        self.style_features.append(output)

    def forward(self, image):

        self.content_in = self.encoder(image)
        self.style_features = []

        return self.decoder(self.content_in)