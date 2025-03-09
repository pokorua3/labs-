import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tkinter as tk
from tkinter import messagebox

def cr_points(count, x_range, y_range, category):
    points = [[random.uniform(*x_range), random.uniform(*y_range)] for _ in range(count)]
    return points, [category] * count

x1, y1 = cr_points(50, (0, 5), (0, 5), 0)
x2, y2 = cr_points(50, (6, 10), (6, 10), 1)
x, y = x1 + x2, y1 + y2

def split_data(x, y, p=0.8):
    data = list(zip(x, y))
    random.shuffle(data)
    split_idx = int(len(data) * p)
    train, test = data[:split_idx], data[split_idx:]
    return list(zip(*train)), list(zip(*test))

(x_train, y_train), (x_test, y_test) = split_data(x, y)

def fit(x_train, y_train, x_test, k=3):
    def euclidean(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    y_pred = []
    for test_point in x_test:
        sosedi = sorted(zip(x_train, y_train), key=lambda t: euclidean(test_point, t[0]))[:k]
        y_pred.append(Counter(y for _, y in sosedi).most_common(1)[0][0])
    return y_pred

def run_knn():
    k = int(k_entry.get())
    y_predict = fit(x_train, y_train, x_test, k)
    accuracy = sum(yt == yp for yt, yp in zip(y_test, y_predict)) / len(y_test)
    messagebox.showinfo("Результат", f"Точность модели: {accuracy:.2f}")

    def plot_data():
        colors = ['blue', 'pink']

        for label in set(y_train):
            train_points = [x for x, y in zip(x_train, y_train) if y == label]
            plt.scatter(*zip(*train_points), color=colors[label], label=f"Class {label} (train)", alpha=0.6)

        for point, label in zip(x_test, y_predict):
            plt.scatter(*point, color=colors[label], edgecolors='black', marker="x", s=100)

        plt.legend()
        plt.xlabel("X координаты")
        plt.ylabel("Y координаты")
        plt.title("k-NN метод")
        plt.show()

    plot_data()


root = tk.Tk()
root.title("k-NN Классификатор")

tk.Label(root, text="Введите k:").pack()

k_entry = tk.Entry(root)
k_entry.pack()
k_entry.insert(0, "3")

tk.Button(root, text="Запустить метод", command=run_knn).pack()
root.mainloop()