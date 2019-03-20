# input:   1) datamatrix as loaded by numpy.loadtxt('dataset.txt')
#	         2) an integer d specifying the number of dimensions for the output (most commonly used are 2 or 3)
# output:  1) an N x d numpy array containing the d coordinates of the N original datapoints projected onto the top d PCs

import matplotlib.pyplot as plt
import numpy as np

from pca import pca


def mds(data, d, show_pc_plots):
  eigen_values, eigen_vectors, mean = pca(data, show_pc_plots=show_pc_plots)

  datamatrix = np.dot(np.array(eigen_vectors).T, data.T)

  return datamatrix[:d]
