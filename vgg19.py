import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def pooling_func(pooltype):
    if pooltype == "avg":
        return nn.AvgPool2d((2,2), stride=(2,2))
    else:
        return nn.MaxPool2d((2,2), stride=(2,2))

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_model(pooltype, device):

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    cnn = models.vgg19(pretrained=True).features.eval()
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            layer = pooling_func(pooltype)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)    
    
    model.eval().to(device)
    model.requires_grad_(False)

    return model


def get_features(x, model, layers, use_relu=False):
    
    if use_relu:
        fea_layers = ["relu_" + str(element) for element in layers]
    else:
        fea_layers = ["conv_" + str(element) for element in layers]

    features = []

    for name, layer in model.named_children():
        x = layer(x)
        if str(name) in fea_layers:
            features.append(x)

    return features



