import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from pca import pca
from mds import mds

from plotify import Plotify
plotify = Plotify()

murderdata = np.loadtxt('murderdata2d.txt')
scaler = StandardScaler()

murderdata_std = scaler.fit_transform(murderdata)

eigenvalues_m_std, eigenvectors_m_std, mean_m = pca(murderdata, with_std=True)

arrows = []

for vector, value in zip(eigenvectors_m_std, eigenvalues_m_std):
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
  x_list=[murderdata_std[:, 0]],
  y_list=[murderdata_std[:, 1]],
  linewidth=0.25,
  alpha=1,
  xlabel='Unemployment rate (%)',
  ylabel='murders per year per 1,000,000 inhabitants',
  title='Murder rates in correlaction with unemployment \n (Standardized)',
  legend_labels=(''),
  arrows=arrows,
  equal_axis=True
)

scaler = StandardScaler(with_mean=True, with_std=False)
murderdata_centered = scaler.fit_transform(murderdata)

arrows = []

eigenvalues_m, eigenvectors_m, mean_m = pca(
    murderdata_centered, with_std=False)

for vector, value in zip(eigenvectors_m, eigenvalues_m):
  arrow = {
    'x': mean_m[0],
    'y': mean_m[1],
    'dx': np.sqrt(value) * vector[0],
    'dy': np.sqrt(value) * vector[1],
    'width': 0.03,
    'color': '#4FB99F'
  }

  arrows.append(arrow)

plotify.scatter_plot(
  x_list=[murderdata_centered[:, 0]],
  y_list=[murderdata_centered[:, 1]],
  linewidth=0.25,
  alpha=1,
  xlabel='Unemployment rate (%)',
  ylabel='murders per year per 1,000,000 inhabitants',
  title='Murder rates in correlaction with unemployment \n (Centered)',
  legend_labels=(''),
  arrows=arrows,
  equal_axis=False,
  tickfrequencyone=False
)


weed_crop_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
weed_crop_train = weed_crop_train[:, :-1]

weed_crop_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

eigenvalues_p, eigenvectors_p, mean_p = pca(weed_crop_train)

datamatrix = mds(weed_crop_train, 2, show_pc_plots=True)

plotify.scatter_plot(
  x_list=[datamatrix[0, :]],
  y_list=[datamatrix[1, :]],
  tickfrequencyone=False,
  xlabel='PC 1',
  ylabel='PC 2',
  title='First 2 PCs of the Pesticide dataset'
)

datamatrix = mds(weed_crop_train, 3, show_pc_plots=False)

plotify.scatter3d(
  x=datamatrix[0, :],
  y=datamatrix[1, :],
  z=datamatrix[2, :],
  xlabel='PC 1',
  ylabel='PC 2',
  zlabel='PC 3',
  title='First 3 PCs of the Pesticide dataset'
)

from sklearn.cluster import KMeans

X_train = weed_crop_train[:, :-1]
y_train = weed_crop_train[:, -1]

X_test = weed_crop_test[:, :-1]
y_test = weed_crop_test[:, -1]

starting_point = np.vstack((X_train[0,], X_train[1,]))


X = datamatrix
y = y_train


kmeans = KMeans(n_clusters=2, n_init=1, init=starting_point, algorithm='full').fit(X_train)


labels = kmeans.labels_

print('kmeans.cluster_centers_', kmeans.cluster_centers_)

colors = []

for l in labels:
  if l == 0:
    colors.append('#4FB99F')
  elif l == 1:
    colors.append('#F2B134')

fig2, ax2 = plt.subplots()

fig2.patch.set_facecolor('#1C2024')
ax2.set_facecolor('#1C2024')

ax2.set_title('KMeans clusters in 2D')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.scatter(X[0, :], X[1, :], c=colors, edgecolor='#333333', alpha=0.8)

plt.show()

fig = plt.figure()
fig.patch.set_facecolor('#1C2024')
ax = Axes3D(fig, rect=[0, 0, 1, 1])
ax.set_facecolor('#1C2024')

ax.scatter(X[0,:], X[1,:], X[2,:], c=colors, edgecolor='#333333', alpha=0.8)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title('KMeans clusters in 3D')
ax.dist = 12

plt.show()
