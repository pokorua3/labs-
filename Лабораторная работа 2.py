#Лабораторная работа 2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def generate_circle(centers, radii, points_per_cluster=20):
    data = []
    for center, radius in zip(centers, radii):
        for _ in range(points_per_cluster):
            r = radius * np.sqrt(np.random.random())
            angle = 2 * np.pi * np.random.random()
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            data.append([x, y])
    return np.array(data)


def initialize_krestiki(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_krestiki(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])


def kmeans_with_history(X, k=3, max_iter=100, tol=1e-4):
    krestiki = initialize_krestiki(X, k)
    history = {'centers': [krestiki.copy()], 'labels': []}

    for _ in range(max_iter):
        labels = assign_clusters(X, krestiki)
        history['labels'].append(labels.copy())

        new_krestiki = update_krestiki(X, labels, k)
        history['centers'].append(new_krestiki.copy())

        if np.all(np.abs(new_krestiki - krestiki) < tol):
            break
        krestiki = new_krestiki

    return history


centers = np.array([[2, 3], [8, 10], [5, 15]])
radii = np.array([1.5, 2.0, 2.5])
data = generate_circle(centers, radii, points_per_cluster=30)

history = kmeans_with_history(data, k=3, max_iter=100)

fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.25)

colors = ['red', 'green', 'blue']
cmap = plt.cm.colors.ListedColormap(colors)
scatter = ax.scatter(data[:, 0], data[:, 1], c=history['labels'][0], cmap=cmap)
centers_plot = ax.scatter(history['centers'][0][:, 0], history['centers'][0][:, 1],
                          c='black', marker='X', s=200, label='Центроиды')
ax.set_title('K-средние кластеризация')
ax.set_xlabel('X координата')
ax.set_ylabel('Y координата')
ax.legend()
ax.grid(True)

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Итерация', 0, min(7, len(history['labels']) - 1), valinit=0, valstep=1)


def update(val):
    iteration = int(slider.val)
    scatter.set_array(history['labels'][iteration])
    centers_plot.set_offsets(history['centers'][iteration])
    ax.set_title(f'K-средние кластеризация (Итерация {iteration})')
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()