from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#from plotify import Plotify

plt.style.use('ggplot')

# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigen_values in a vector (numpy array) in descending order
#          2) the unit eigen_vectors in a matrix (numpy array) with each column being an 
#             eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigen_values (the projected variance) is decreasing, and
# the eigen_vectors have the same order as their associated eigen_values


def pca(X):
  eigen_values = []
  eigen_vectors = []

  # First we standardize the data, a very important step when doing PCA


  # X_centered = X - mean
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  mean = np.mean(X.T, axis=1)
  print('mean', mean)

  covariance_matrix = np.cov(X.T)

  eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
  key = argsort(eigen_values)[::-1][:2]
  eigen_values, eigen_vectors = eigen_values[key], eigen_vectors[:, key]

  pca = PCA(n_components=2)
  pca.fit(X)


  print('pca.components_', pca.components_)
  print('pca.explained_variance_', pca.explained_variance_)

  rotation_matrix = [[-1,0],[0, -1]]


  plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
  plt.axis('equal')
  
  for vector, value in zip(eigen_vectors, eigen_values):
    plt.arrow(mean[0],
              mean[1],
              np.sqrt(value) * vector[0],
              np.sqrt(value) * vector[1],
              color='r',
              width=0.03)

  plt.show()


  print('eigen_values', eigen_values)
  print('eigen_vectors', eigen_vectors)

  # plotify.scatter_plot(
  #     x_list=[X[:, 0]],
  #     y_list=[X[:, 1]],
  #     linewidth=0.25,
  #     alpha=1,
  #     xlabel='Unemployment rate (%)',
  #     ylabel='murders per year per 1,000,000 inhabitants',
  #     title='Murder rates in correlaction with unemployment',
  #     legend_labels=(''),
  #     vectors=vectors,
  #     mean=pca.mean_,
  #     variance=pca.explained_variance_,
  #     components=pca.components_
  # )

  return eigen_values, eigen_vectors


from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
