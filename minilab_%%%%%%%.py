import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.cluster import MeanShift, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import time


#ПРавильная
def generate_data(data_type, n_samples=300, noise=0.05):
    np.random.seed(42)
    if data_type == "circles":
        return make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    elif data_type == "moons":
        return make_moons(n_samples=n_samples, noise=noise)
    elif data_type == "blobs":
        return make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0)
    elif data_type == "anisotropic":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.8)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        return np.dot(X, transformation), y
    elif data_type == "varied":
        return make_blobs(n_samples=n_samples, centers=3, cluster_std=[1.0, 2.5, 0.5])
    elif data_type == "structured":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.5)
        X[y == 1] += [3, -2]
        return X, y


def apply_clustering(X, algorithm):
    X = StandardScaler().fit_transform(X)
    start_time = time.time()

    if algorithm == "MeanShift":
        labels = MeanShift(bandwidth=0.5).fit_predict(X)
    elif algorithm == "Agglomerative Clustering":
        labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    elif algorithm == "DBSCAN":
        labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)

    exec_time = time.time() - start_time
    return labels, exec_time


plt.figure(figsize=(12, 18))
algorithms = ["MeanShift", "Agglomerative Clustering", "DBSCAN"]
data_types = ["circles", "moons", "blobs", "anisotropic", "varied", "structured"]

for i, data_type in enumerate(data_types):
    for j, algo in enumerate(algorithms):
        X, _ = generate_data(data_type)
        labels, exec_time = apply_clustering(X, algo)

        ax = plt.subplot(6, 3, i * 3 + j + 1, aspect='equal')
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0.95, 0.05, f"{exec_time:.3f}s",
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(facecolor='pink', alpha=0.7, edgecolor='purple'))

        if j == 0:
            ax.set_ylabel(data_type, fontsize=10, rotation=0, ha='right', va='center')
        if i == 0:
            ax.set_title(algo, fontsize=10)

plt.tight_layout()
plt.show()