import numpy as np
from numpy import array
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# from tsne import bh_sne

# X = np.load('./extracted_features/test_sv180_img_feat.npy')
X_img = np.load('./extracted_features/ModelNet40-test-2_cm_sv180_img_feat.npy')
X_pt = np.load('./extracted_features/ModelNet40-test-2_cm_sv180_cloud_feat.npy')

y = np.load('./extracted_features/ModelNet40-test-2_cm_sv180_label.npy')


# tsne = TSNE(n_components=2, random_state=0)
tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)

X_img_2d = tsne.fit_transform(X_img)
X_pt_2d = tsne.fit_transform(X_pt)


target_ids = range(len(y))

# plt.figure(figsize=(6, 12))

fig, (ax1, ax2) = plt.subplots(1,2)

fig.suptitle('Image Features and Point Cloud Features')

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, 40)]
print(len(colors))
print(len(colors))
print(len(colors))
print(len(colors))
print(len(colors))
print(len(colors))
print(len(colors))
colors = ['red', 'blue', 'navy', 'green', 'violet', 'brown', 'gold', 'lime', 'teal', 'olive']
# c=np.random.rand(1,3)
for i in target_ids:
    if i<10:
      ax1.scatter(X_img_2d[y == i, 0], X_img_2d[y == i, 1], c=colors[i])
      ax2.scatter(X_pt_2d[y == i, 0], X_pt_2d[y == i, 1], c=colors[i])
plt.show()
