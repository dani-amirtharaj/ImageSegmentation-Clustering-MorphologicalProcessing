
# Setting up program
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

# Setting seed for reproducibility
UBIT = 'damirtha'
np.random.seed(sum([ord(c) for c in UBIT]))


# Funnction to return classification vector given X and Mu
def kmeansClass(X, Mu):
    classification = []
    for x in X:
        classification.append(np.argmin(np.sum((Mu - x) **2,axis = 1)))
    return classification         

# Funnction to return Mu vector given X and current classification
def computeMu(X, classification, Mu):
    for i in range(len(Mu)):
        Mu[i] = np.round(np.mean(X[np.array(classification).ravel() == i], axis = 0), 2)

# Function to plot K-means classification
def plotKmeansClass(classification, Mu, obj = 'X'):
    colorX = []
    colorMu = ['r','g','b']

    for i in range(len(classification)):
        colorX.append(colorMu[0] if classification[i] == 0 else colorMu[1] if classification[i] == 1 else colorMu[2])

    plt.xlim(4.3,7)
    plt.ylim(2.6,4.5)
    
    if obj == 'X':
        plt.scatter(X[:,0], X[:,1], 70, edgecolor='b', facecolor = colorX, marker = 10)
        for i in range(len(X)):
            plt.text(X[i][0]-0.06, X[i][1]-0.05, str(tuple(X[i].ravel())), fontsize="x-small")
            
    if obj == 'Mu':
        plt.scatter(Mu[:,0], Mu[:,1], 50, facecolor = colorMu)
        for i in range(len(Mu)):
            plt.text(Mu[i][0]-0.06, Mu[i][1]-0.08, str(tuple(Mu[i].ravel())), fontsize="x-small")

    return plt


# Given data and cluster parameters Mu for k clusters (Task 3.1 to 3.3)
X = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
K = 3
Mu = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])

# Claassifying X based on given Mu cluster centers
classification = kmeansClass(X, Mu)
print('Iteration1 a:',np.array(classification)+1)
plt1 = plt.figure()
plt1 = plotKmeansClass(classification, Mu)
plt1.savefig('Results/task3_iter1_a.jpg')

# Re-calculating cluster centers and classifying X, iteration 1
computeMu(X, classification, Mu)
print('Iteration1 b:',Mu)
classification = kmeansClass(X, Mu)
print('Iteration2 a:',np.array(classification)+1)
plt.figure()
plotKmeansClass(classification, Mu, 'Mu')
plt.savefig('Results/task3_iter1_b.jpg')
plt.figure()
plotKmeansClass(classification, Mu)
plt.savefig('Results/task3_iter2_a.jpg')

# Re-calculating cluster centers, iteration 2
computeMu(X, classification, Mu)
print('Iteration2 b:',Mu)
classification = kmeansClass(X, Mu)
plt.figure()
plotKmeansClass(classification, Mu, 'Mu')
plt.savefig('Results/task3_iter2_b.jpg')


#------Image compression with Color Quantization-----#

# Applying K-means for color quantization
image = cv2.imread('Images/baboon.jpg')
XImg = np.squeeze(image.ravel().reshape(-1,1,3))
K = [3, 5, 10, 20]

for k in K:
    Mu = np.squeeze(np.random.randint(256, size = (k,3)))
    itrMu = np.zeros(Mu.shape)
    count = 0
    while ( not (np.array_equal(Mu,itrMu))):
        itrMu = Mu.copy()
        classification = kmeansClass(XImg, Mu)
        computeMu(XImg, classification, Mu)
        count+= 1
    Xout = np.array([Mu[i].ravel() for i in classification])
    Xout = Xout.reshape(image.shape)
    cv2.imwrite('Results/task3_baboon_'+str(k)+'.jpg',Xout)
