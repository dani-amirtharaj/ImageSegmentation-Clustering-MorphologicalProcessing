
# Importing necessary packages for this project
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setting seed for reproducibility
UBIT = 'damirtha'
np.random.seed(sum([ord(c) for c in UBIT]))


# Function to perform erosion on image, given image, structuring element and origin
def erosion(image, struct_elem, origin):
    img_list = []
    image = np.pad(image, int(len(struct_elem)/2),'edge')
    for img_row in range(origin[0], len(image)-len(struct_elem)+origin[0]+1):
        for img_col in range(origin[1], len(image[0])-len(struct_elem[0])+origin[1]+1):
            img_list.append([255 if np.array_equal(image[img_row-origin[0]:img_row-origin[0]+len(struct_elem),
                                         img_col-origin[1]:img_col-origin[1]+len(struct_elem[0])]
                                   , struct_elem) else 0 ])
    return np.array(img_list).reshape(-1,len(image[0])-len(struct_elem[0])+1)

# Function to perform dilation on image, given image, structuring element and origin
def dilation(image, struct_elem, origin):
    img_list = []
    image = np.pad(image, int(len(struct_elem)/2),'edge')
    for img_row in range(origin[0], len(image)-len(struct_elem)+origin[0]+1):
        for img_col in range(origin[1], len(image[0])-len(struct_elem[0])+origin[1]+1):
            img_list.append([0 if np.array_equal(image[img_row-origin[0]:img_row-origin[0]+len(struct_elem),
                                         img_col-origin[1]:img_col-origin[1]+len(struct_elem[0])]
                                   , np.logical_not(struct_elem)) else 255 ])
    return np.array(img_list).reshape(-1,len(image[0])-len(struct_elem[0])+1)

# Function to perform opening on image, given image, structuring element and origin
def opening(image, struct_elem, origin):
    image_eroded = erosion(image, struct_elem, origin)
    return dilation(image_eroded, struct_elem, origin)

# Function to perform closing on image, given image, structuring element and origin
def closing(image, struct_elem, origin):
    image_dilated = dilation(image, struct_elem, origin)
    return erosion(image_dilated, struct_elem, origin)

image=cv2.imread('Images/noise.jpg', cv2.IMREAD_GRAYSCALE)
struct_elem = np.ones((3,3))*255
origin = (1,1)

# Perform opening and closing perations
image_opening = opening(image, struct_elem, origin)
image_closing = closing(image, struct_elem, origin)

# Perform opening on closing of image to denoise
image_denoise1 = opening(image_closing, struct_elem, origin)

# Perform closing on opening of image to denoise
image_denoise2 = closing(image_opening, struct_elem, origin)

# Background of denoised images
image_denoise1_bg = image_denoise1-erosion(image_denoise1, struct_elem, origin)
image_denoise2_bg = image_denoise2-erosion(image_denoise2, struct_elem, origin)

cv2.imwrite('Results/res_noise1.jpg',image_denoise1)
cv2.imwrite('Results/res_noise2.jpg',image_denoise2)
cv2.imwrite('Results/res_bound1.jpg',image_denoise1_bg)
cv2.imwrite('Results/res_bound2.jpg',image_denoise2_bg)
