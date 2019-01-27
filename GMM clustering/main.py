
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


# Function to apply GMM clustering given X, Mu, Sigma, K and Pi
def GMMclustering(Mu, Sigma, Pi, K,  X, data = 'X', itr = 5):
    c=0
    for j in range(itr):
        pXgivenC = []
        for i in range(K):
            pXgivenC.append(multivariate_normal.pdf(X, mean = Mu[i], cov = Sigma[i]))
        pXgivenC = np.transpose(np.array(pXgivenC))
        pX = np.array(np.sum(np.multiply(pXgivenC,Pi), axis= 1))
        pCgivenX = np.multiply(np.transpose(np.multiply(pXgivenC,Pi)),pX ** -1)
        for i in range(K):
            Mu[i] = np.average(X, axis = 0, weights = pCgivenX[i])
            Sigma[i] = np.cov(X, rowvar = 0, aweights = pCgivenX[i], ddof = 0)
        Pi = np.sum(pCgivenX, axis = 1)/len(X)
        if data == 'faithful':
            plt.figure()
            plt.plot(X[:,0], X[:,1], 'bo', markersize = 3)
            color = ['r','g','b']
            # Plot a transparent 3 standard deviation covariance ellipse

            for i in range(K):
                plot_cov_ellipse(pos = Mu[i], cov = Sigma[i], nstd=3, alpha=0.5, color= color[i])

            plt.savefig('Results/task3_gmm_iter'+str(j+1)+'.jpg')
        c+=1
    return Mu, Sigma, Pi, c

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

# Initializing GMM parameters 
X = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
K = 3
Mu = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
Sigma = np.array([[[0.5, 0], [0, 0.5]]]*3)
Pi = np.array([1/3, 1/3, 1/3])

# Computing EM algorithm to get soft clusters
Mu, Sigma, Pi, c = GMMclustering(Mu, Sigma, Pi, K,  X, 'X', 1)
print('Mu for 3.5a:')
print(Mu)


# Bonus task 3.5 b, opening dataset and initializing GMM parameters 
file = open('data/faithful.dat','r')
c=0; data=[]
for line in file:
    c+=1
    if c > 26:
        data.append(np.array(line.split()[1:3]))
data = np.array(data)
data = data.astype(np.float)
K = 3
Mu = np.array([[4.0,81],[2.0,57],[4.0, 71]])
Sigma = np.array([[[1.3, 13.98], [13.98, 184.82]]]*K)
Pi = np.array([1/3]*K)

# Computing EM algorithm to get soft clusters
Mu, Sigma, Pi, c = GMMclustering(Mu, Sigma, Pi, K,  data, 'faithful', 5)

