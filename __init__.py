import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

from pca import pca

murderdata = np.loadtxt('murderdata2d.txt')

print(murderdata)

X = murderdata
eigen_values, eigen_vectors = pca(murderdata)

# mean = np.mean(X.T, axis=1)

# cov_matrix = (X - mean).T.dot((X - mean)) / (X.shape[0]-1)

# plt.scatter(X[:, 0], X[:, 1])

# # plt.axis('equal')
# evals, evecs = np.linalg.eig(cov_matrix)

# # Compute the corresponding standard deviations
# s0 = np.sqrt(evals[0])
# s1 = np.sqrt(evals[1])


# plt.plot([mean[0], mean[0] + s0*evecs[0, 0]], [mean[1], mean[1] + s0*evecs[1, 0]], 'r')
# plt.plot([mean[0], mean[0] + s1*evecs[0, 1]], [mean[1], mean[1] + s1*evecs[1, 1]], 'r')

# plt.show()

# pca = PCA(n_components=2)
# pca.fit(X)

# def draw_vector(v0, v1, ax=None):
#     ax = ax or plt.gca()
#     arrowprops = dict(arrowstyle='->',
#                       linewidth=2,
#                       shrinkA=0, shrinkB=0)
#     ax.annotate('', v1, v0, arrowprops=arrowprops)


# # plot data
# plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 1 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)

# plt.show()
