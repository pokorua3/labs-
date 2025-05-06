import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

plt.figure(figsize=(15, 20))
plt.suptitle("Сравнение методов классификации на различных наборах данных", fontsize=16, y=1.02)

classifiers = [
    ("K-ближайших соседей", KNeighborsClassifier(n_neighbors=3)),
    ("Наивный Байесовский классификатор ", GaussianNB()),
    ("Метод опорных векторов", SVC(kernel='rbf', C=1.0))
]

datasets = [
    ("Окружности", datasets.make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=30)),
    ("Параболы", datasets.make_moons(n_samples=500, noise=0.05, random_state=30)),
    ("Разные дисперсии", datasets.make_blobs(n_samples=500, cluster_std=[1.0, 0.5], random_state=30, centers=2)),
    ("Анизотропные", (np.dot(datasets.make_blobs(n_samples=500, random_state=170, centers=2)[0],
                             [[0.6, -0.6], [-0.4, 0.8]]),
                      datasets.make_blobs(n_samples=500, random_state=170, centers=2)[1])),
    ("Слабо пересекающиеся", datasets.make_blobs(n_samples=500, random_state=30, centers=2))
]

for dataset_idx, (dataset_name, (X, y)) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    for clf_idx, (clf_name, clf) in enumerate(classifiers):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        y_pred = clf.predict(X_test)

        plt.subplot(len(datasets), len(classifiers), dataset_idx * len(classifiers) + clf_idx + 1)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

        for i in range(len(X_train)):
            plt.scatter(X_train[i, 0], X_train[i, 1], marker='o' if y_train[i] else 'x',
                        c='blue', alpha=0.5, s=20)

        for i in range(len(X_test)):
            marker = 'o' if y_test[i] else 'x'
            color = 'green' if y_test[i] == y_pred[i] else 'red'
            if marker == 'o':
                plt.scatter(X_test[i, 0], X_test[i, 1], marker=marker, c=color, edgecolors='k', s=30)
            else:
                plt.scatter(X_test[i, 0], X_test[i, 1], marker=marker, c=color, s=30)

        if dataset_idx == 0:
            plt.title(clf_name)
        if clf_idx == 0:
            plt.ylabel(dataset_name)

        plt.xticks(())
        plt.yticks(())

plt.tight_layout()
plt.show()
