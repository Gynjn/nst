import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import PIL
from PIL import Image
import numpy as np
import os
from color import *
from vgg19 import *
from util import *
from tqdm import trange
import cv2 as cv


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR_PATH = "./output"
# SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def style_transfer(content_image, style_image, image_size, content_weight, 
                   style_weight, style_scale, tv_weight, num_iter, content_layer, 
                   style_layer, use_relu, pool_type, preserve_color, init_random=False):
    
    content_img = image_loader(content_image, image_size, device)
    style_img = image_loader(style_image, image_size, device)
    model = get_model(pool_type, device)
    content_features = get_features(content_img, model, content_layer)
    style_features = get_features(style_img, model, style_layer)

    # input_tensor = content_img.clone().requires_grad_().to(device)
    if init_random == False:
        input_tensor = content_img.clone().requires_grad_(True)
    else:
        input_tensor = torch.randn_like(content_img).requires_grad_(True).to(device)
    optimizer = get_input_optimizer(input_tensor)
    
    print('Optimizing..')
    run = [0]
    while run[0] <= num_iter:

        def closure():
            with torch.no_grad():
                input_tensor.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_tensor)
            current_content_features = get_features(input_tensor, model, content_layer)
            current_style_features = get_features(input_tensor, model, style_layer)    

            c_loss = content_loss(current_content_features, content_features, content_weight)
            s_loss = style_loss(current_style_features, style_features, style_weight, style_scale)
            t_loss = tv_loss(input_tensor, tv_weight)

            loss = c_loss + s_loss + t_loss
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} Tv Loss: {:4f}'.format(
                    s_loss, c_loss, t_loss))
                print()

            return s_loss + c_loss + t_loss

        optimizer.step(closure)

    with torch.no_grad():
        input_tensor.clamp_(0, 1)

    output_image = input_tensor.detach().cpu().squeeze().permute(1,2,0).numpy()*255

    if preserve_color = True:
        output_image = color_preserve(output_image, content_img, mask)
        
    output_image = Image.fromarray(output_image.astype('uint8'))
    output_image.save(os.path.join(OUTPUT_DIR_PATH, 'result.png'))



parser = argparse.ArgumentParser(description='Style Transfer')
parser.add_argument('--content_image_path', default = "/hdd/jinnnn/imgs/man.jpg", type=str,
                    help='Path to the image to transform.')

parser.add_argument('--style_image_path', default = "/hdd/jinnnn/imgs/munch.jpg", type=str,
                    help='Path to the style reference image.')

parser.add_argument("--color_mask", type=str, default=None,
                    help='Mask for color preservation')

parser.add_argument("--image_size", dest="image_size", default=512, type=int,
                    help='Minimum image size')

parser.add_argument("--content_weight", dest="content_weight", default=1, type=float,
                    help="Weight of content")

parser.add_argument("--style_weight", dest="style_weight", default=[0.2,0.2,0.2,0.2,0.2], type=float,
                    help="Weight of style, can be multiple for multiple styles")

parser.add_argument("--style_scale", dest="style_scale", default=100000, type=float,
                    help="Scale the weighing of the style")

parser.add_argument("--total_variation_weight", dest="tv_weight", default=1e-3, type=float,
                    help="Total Variation weight")

parser.add_argument("--num_iter", dest="num_iter", default=300, type=int,
                    help="Number of iterations")

parser.add_argument("--content_layer", dest="content_layer", default=[4], type=list,
                    help="Content layer used for content loss.")

parser.add_argument("--style_layer", dest="style_layer", default=[1,2,3,4,5], type=list,
                    help="Content layer used for style loss.")

parser.add_argument("--use_relu", dest="use_relu", default=False, type=bool,
                    help="Use relu activation for style features")

parser.add_argument("--pool_type", dest="pool", default="max", type=str,
                    help='Pooling type. Can be "ave" for average pooling or "max" for max pooling')

parser.add_argument('--preserve_color', dest='color', default="False", type=str,
                    help='Preserve original color in image')

if __name__ == '__main__':
    args = parser.parse_args()

    style_transfer(args.content_image_path, args.style_image_path, args.image_size,
                   args.content_weight, args.style_weight, args.style_scale, args.tv_weight, args.num_iter,
                   args.content_layer, args.style_layer, args.use_relu, args.pool, args.color)
    

    