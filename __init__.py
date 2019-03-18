import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

from pca import pca

murderdata = np.loadtxt('murderdata2d.txt')

eigen_values, eigen_vectors = pca(murderdata, show_pc_plots=False, show_scatter_plot=False)


weed_crop_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
weed_crop_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

eigen_values, eigen_vectors = pca(weed_crop_train, show_pc_plots=False, show_scatter_plot=False)

# print(eigen_values, eigen_vectors)
