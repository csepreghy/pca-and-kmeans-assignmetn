from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from plotify import Plotify

# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigen_values in a vector (numpy array) in descending order
#          2) the unit eigen_vectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigen_values (the projected variance) is decreasing, and the eigen_vectors have the same order as their associated eigen_values

plotify = Plotify()

def pca(data):
  eigen_values = []
  eigen_vectors = []

  # # First we standardize the data, a very important step when doing PCA
  X_normalized = StandardScaler().fit_transform(data)
  print(X_normalized)

  mean_vector = np.mean(X_normalized, axis=0)
  covariance_matrix = (X_normalized - mean_vector).T.dot((X_normalized - mean_vector)) / (X_normalized.shape[0]-1)
  print('Covariance matrix \n%s' % covariance_matrix)
  
  # Or alternatively we can use numpy's covariance function
  covariance_matrix_np = np.cov(X_normalized.T)
  print('Covariance matrix np \n%s' % covariance_matrix_np)

  eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)



  vector_list = []
  for i, v in enumerate(eigen_vectors):
    scaled_v = v * eigen_values[i]
    vector_list.append(scaled_v)

  print('vector_list', vector_list)


  vectors = {
    'origins': [mean_vector, mean_vector],
    'vector_list': vector_list
  }

  plotify.scatter_plot(
      x_list=[X_normalized[:, 0]],
      y_list=[X_normalized[:, 1]],
      linewidth=0.25,
      alpha=1,
      xlabel='Unemployment rate (%)',
      ylabel='murders per year per 1,000,000 inhabitants',
      title='Murder rates in correlaction with unemployment',
      legend_labels=('Non-smokers'),
      vectors=vectors
  )

  return eigen_values, eigen_vectors


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
