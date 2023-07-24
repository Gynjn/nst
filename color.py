import numpy as np
from util import *
from scipy import linalg as LG

# calculates channel means covariance matrix for image of shape (..., 3) or (3, ...)
def musigma(arr):
  if arr.shape[2] == 3:
    arr = arr.transpose(2, 0, 1)
  narr = arr.reshape(3, -1)
  mu = (narr.sum(axis = 1)/narr.shape[1]).reshape(-1,1)
  sigma = np.matmul((narr - mu), (narr - mu).T) / narr.shape[1]
  return mu, sigma

# calculates transformation matrices for color histogram matching using 3D color matching formulations
def image_analogies(carr, sarr):
  cmu, csig = musigma(carr)
  smu, ssig = musigma(sarr)
  A = np.matmul(LG.fractional_matrix_power(csig,0.5), LG.fractional_matrix_power(ssig,-0.5))
  b = cmu - np.matmul(A, smu)
  return A, b

def style_img_transform(carr, sarr):
  if carr.shape[2] == 3:
    carr = carr.transpose(2,0,1)
  if sarr.shape[2] == 3:
    sarr = sarr.transpose(2,0,1)
    
  A, b = image_analogies(carr, sarr)
  
  ss = sarr.reshape(3,-1)
  n_ss = np.matmul(A, ss) + b
  n_ss = n_ss.reshape(sarr.shape)
  return n_ss.transpose(1,2,0)


def original_color_transform(content, generated, mask=None):
    generated = fromimage(toimage(generated, mode='RGB'), mode='YCbCr')  # Convert to YCbCr color space

    if mask is None:
        generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    else:
        width, height, channels = generated.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    generated[i, j, 1:] = content[i, j, 1:]

    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated