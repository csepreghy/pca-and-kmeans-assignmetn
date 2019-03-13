import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

from pca import pca

murderdata = np.loadtxt('murderdata2d.txt')

print(murderdata)

X = murderdata
eigen_values, eigen_vectors = pca(murderdata)