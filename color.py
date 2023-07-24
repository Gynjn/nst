import numpy as np
from util import *
from scipy import linalg as LG
import cv2


# matching yuv channel
def color_preserve(output_image, content_img, mask=None):
  '''
  output image: numpy array
  content_img: tensor
  mask: numpy array
  if white then mask value is 0, if not mask value is 255
  '''

  content_output = content_img.detach().cpu().squeeze().permute(1, 2, 0).numpy() * 255
  output_yuv = cv2.cvtColor(np.float32(output_image), cv2.COLOR_RGB2YUV)
  content_yuv = cv2.cvtColor(np.float32(output_image), cv2.COLOR_RGB2YUV)
  # output_yuv = cv2.cvtColor(output_image, cv2.COLOR_RGB2YUV)
  # content_yuv = cv2.cvtColor(output_image, cv2.COLOR_RGB2YUV)

  img_size = content_img.shape[1] # assume the width and height is same

  # Only applying color preservation when the pixel is not white
  if mask is not None:
    mask = cv2.resize(mask, dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST)
    output_yuv[:, :, 1:3] = np.where(mask[:, :, np.newaxis] == 255, content_yuv[:, :, 1:3], output_yuv[:, :, 1:3])  

  else:
    output_yuv[:, :, 1:3] = content_yuv[:,:,1:3]

  output_image = cv2.cvtColor(output_yuv, cv2.COLOR_YUV2RGB)
  output_image = np.clip(output_image, 0, 255).astype(np.uint8)

  if mask is not None:
    output_image = np.where(mask[:,:, np.newaxis] == 255, output_image, content_output)

  return output_image