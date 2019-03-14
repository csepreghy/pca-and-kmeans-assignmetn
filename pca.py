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


def pca(X, show_pc_plots=True):
  eigen_values = []
  eigen_vectors = []

  # First we standardize the data to get unit vectors for the eigenvectors
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  covariance_matrix = np.cov(X.T)

  eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
  key = np.argsort(eigen_values)[::-1][:len(eigen_values)]
  eigen_values, eigen_vectors = eigen_values[key], eigen_vectors[:, key]

  # Only for visual pruposes. Both of them count as eigenvectors
  if len(eigen_values) == 2:
    eigen_vectors[0] = -eigen_vectors[0]

  pca = PCA(n_components=len(eigen_values))
  pca.fit(X)

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
    title='Murder rates in correlaction with unemployment \n (Standardized)',
    legend_labels=(''),
    arrows=arrows,
    equal_axis=True
  )

  print('pca.explained_variance_\n', pca.explained_variance_)
  print('eigen_values \n', eigen_values)

  print('pca.components_[0]\n', pca.components_)
  print('eigen_vectors[0]\n', eigen_vectors)


  # Make a list of (eigenvalue, eigenvector) tuples
  eig_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
              for i in range(len(eigen_values))]

  # Sort the (eigenvalue, eigenvector) tuples from high to low
  eig_pairs.sort()
  eig_pairs.reverse()

  # Visually confirm that the list is correctly sorted by decreasing eigenvalues
  print('Eigenvalues in descending order:')
  for i in eig_pairs:
      print(i[0])
  
  total = sum(eigen_values)
  
  print('total \n', total)

  var_exp = [(i / total)*100 for i in sorted(eigen_values, reverse=True)]

  total_cumulative_explain_varience = 0
  cumulative_explain_variences = []

  for eigen_value in sorted(eigen_values, reverse=True):
    percentage = eigen_value / total * 100
    print('percentage', percentage)
    total_cumulative_explain_varience += percentage
    cumulative_explain_variences.append(total_cumulative_explain_varience)

  print('cumulative_explain_variences', cumulative_explain_variences)
  
  if show_pc_plots == True:
    xticks = []
    for _ in range(len(eigen_values)):
      xticks.append('PC ' + str(_))

    plotify.bar(
      x_list=range(len(eigen_values)),
      y_list=var_exp,
      title='PCA Analysis',
      ylabel='% Variance Explained',
      xticks=xticks,
      rotation=30
    )

    plotify.plot(
      y_list=cumulative_explain_variences,
      title='PCA Analysis',
      ylabel='% Variance Explained',
      xlabel='Number of Features',
    )


  return eigen_values, eigen_vectors

