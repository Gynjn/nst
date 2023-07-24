import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.transforms import ToTensor

'''
Just using Pixel Value
'''
# image = cv2.imread("./imgs/man.jpg")

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# height, width = gray_image.shape[:2]
# mask = np.zeros((height, width), dtype=np.uint8)

# for y in range(height):
#     for x in range(width):
#         if gray_image[y, x] < 255:
#             mask[y, x] = 255

# cv2.imwrite("./imgs/man_mask.png", mask)

image = cv2.imread('./imgs/man.jpg')
resized_image = cv2.resize(image, (512, 512))

model = models.segmentation.deeplabv3_resnet50(pretrained=True).eval()

input_tensor = ToTensor()(resized_image).unsqueeze(0)
normalized_input = input_tensor / 255.0
output = model(normalized_input)['out']
segmentation_mask = output.argmax(1)[0].detach().cpu().numpy()

# Create a mask of the segmented regions
segmented_mask = np.zeros_like(resized_image)
segmented_mask[segmentation_mask > 0] = [0, 0, 255]  # Color of segmented regions (e.g., red)

# Overlay the segmented regions on the original image
segmented_image = cv2.addWeighted(resized_image, 0.7, segmented_mask, 0.3, 0)

# Display the original image and segmented output
# cv2.imshow('Original Image', resized_image)
cv2.imwrite('Segmented Output.png', segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# background_class_index = 0  # Assuming the background class index is 0
# background_mask = output.argmax(1)[0] == background_class_index
# background_mask = background_mask.detach().cpu().numpy().astype(np.uint8) * 255

# cv2.imwrite('./imgs/background_mask.png', background_mask)

