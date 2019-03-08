import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

from pca import pca

murderdata = np.loadtxt('murderdata2d.txt')

print(murderdata)

eigen_values, eigen_vectors = pca(murderdata)


print('eigen_values', eigen_values)
print('eigen_vectors', eigen_vectors)



def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=1,
                      shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


