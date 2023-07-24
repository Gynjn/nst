import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import color
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import os
import torch.optim as optim

def content_loss(y, y_hat, weight):
    loss = 0
    for i in range(len(y_hat)):
        loss += weight * F.mse_loss(y[i], y_hat[i])
    return loss

def gram_matrix(x, normalize = True):
    b, c, h, w = x.size()
    features = x.view(b*c, h*w)
    G = torch.mm(features, features.t())
    if normalize:
        return G.div(b*c*h*w)
    else:
        return G

def style_loss(y, y_hat, weights, scale):
    loss = 0
    for i in range(len(y_hat)):
        y_gram = gram_matrix(y[i])
        y_hat_gram = gram_matrix(y_hat[i])
        loss += weights[i] * F.mse_loss(y_gram, y_hat_gram)
    return loss * scale

def tv_loss(img, tv_weight):
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss

def a_channel_check(img):
    if img.mode == "RGB":
        return img
    elif img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        background.save(img.filename, quality=100)
        converted_img = Image.open(img.filename)
        return converted_img

def image_loader(image_path, image_size, device):

    if not os.path.exists(image_path):
        raise Exception(f'Path does not exist: {img_path}')

    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)

def get_input_optimizer(input_image):
    optimizer = optim.LBFGS([input_image])
    return optimizer

