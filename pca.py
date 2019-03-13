from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from plotify import Plotify

plotify = Plotify()

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

  # First we standardize the data to get unit vectors for the eigenvectors
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  covariance_matrix = np.cov(X.T)

  eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
  key = np.argsort(eigen_values)[::-1][:2]
  eigen_values, eigen_vectors = eigen_values[key], eigen_vectors[:, key]

  # Only for visual pruposes. Both of them count as eigenvectors
  eigen_vectors[0] = -eigen_vectors[0]

  pca = PCA(n_components=2)
  pca.fit(X)

  print('pca.components_\n', pca.components_) 
  print('pca.explained_variance_\n', pca.explained_variance_)

  arrows = []
  
  for vector, value in zip(eigen_vectors, eigen_values):
    arrow = {
      'x': 0,
      'y': 0,
      'dx': np.sqrt(value) * vector[0],
      'dy': np.sqrt(value) * vector[1],
      'width': 0.03,
      'color': '#4FB99F'
    }

    arrows.append(arrow)

  plotify.scatter_plot(
    x_list=[X[:, 0]],
    y_list=[X[:, 1]],
    linewidth=0.25,
    alpha=1,
    xlabel='Unemployment rate (%)',
    ylabel='murders per year per 1,000,000 inhabitants',
    title='Murder rates in correlaction with unemployment',
    legend_labels=(''),
    arrows=arrows,
    equal_axis=True
  )

  print('eigen_values', eigen_values)
  print('eigen_vectors', eigen_vectors)


  return eigen_values, eigen_vectors

