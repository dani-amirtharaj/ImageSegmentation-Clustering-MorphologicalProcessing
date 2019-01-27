
# Importing necessary packages for this project
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setting seed for reproducibility
UBIT = 'damirtha'
np.random.seed(sum([ord(c) for c in UBIT]))


# Function to draw bounding box for an object detected, given image, binary thresholded image
def getBoundedObject(image, imageT, i):
    
    pixelPos = np.where(imageT==255)

    xmin = np.min(pixelPos[0])
    ymin = np.min(pixelPos[1])
    xmax = np.max(pixelPos[0])
    ymax = np.max(pixelPos[1])
    
    image[xmin:xmax, ymin+1] = [0,0,255]
    image[xmin:xmax, ymax+1] = [0,0,255]
    image[xmin, ymin:ymax+1] = [0,0,255]
    image[xmax, ymin:ymax+1] = [0,0,255]
    
    # Print corner points of the bounding box
    print("(",xmin,",",ymin+i,")")
    print("(",xmin,",",ymax+i,")")
    print("(",xmax,",",ymin+i,")")
    print("(",xmax,",",ymax+i,")")
    
    return image

# Function to draw bounding box for all objects detected, given image, binary thresholded image
def getBoundedImage(image, imageT):
    i = 0
    startPoint = 0
    endPoint = 0
    gap =0
    vertPoints = []
    while (i>=0 and i<len(imageT[0])):
        if (np.max(imageT[:,i])==255):
            gap = 0
            if startPoint==0:
                startPoint = i
        else:
            if gap == 0:
                endPoint = i
            gap+=1
            if (gap>20):
                if (startPoint>0):
                    vertPoints.append([startPoint, endPoint])
                    gap = 0
                startPoint = 0
        i+=1
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in vertPoints:
        print("Object")
        image[:,i[0]:i[1]+1,:] = getBoundedObject(image[:,i[0]:i[1]+1,:],imageT[:,i[0]:i[1]+1],i[0])
    return image

image = cv2.imread('Images/segment.jpg', cv2.IMREAD_GRAYSCALE)
imagePixels = image.flatten()

# Calculation of histogram for gray level intensities of the given image
hist = []
for i in range(1,256):
    hist.append(np.size(imagePixels[imagePixels == i]))

fig, ax=plt.subplots(figsize=(23,8))
ax.plot(hist)
ax.set(xlabel='Intensity', ylabel='Number of pixels')
ax.set_xticks(np.arange(1,256,5))
ax.grid()
plt.savefig('Results/histogram.jpg')

t1 = 139 # found from histogram for filet
t2 = 178 # found from histogram for filet/bones
t3 = 207 # found from histogram for bones

# Using since it is the optimal threshold, detecting the objects of interest
imageT = np.array([255 if image[i,j]> t3 else 0 for i in range(len(image)) for j in range(len(image[0]))]).reshape(-1, len(image[0]))

print("Coordinates of Bounding boxes: ")
imageBound = getBoundedImage(image, imageT)
cv2.imwrite('Results/res_segment.jpg',imageBound)

