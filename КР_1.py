#1 вариант метод К-ближайших соседей
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

x1_min, x1_max, y1_min, y1_max = 10, 16, 10, 16
x2_min, x2_max, y2_min, y2_max = 10, 15, 10, 15

data_points = []
labels = []

for i in range(50):
    data_points.append([random.uniform(x1_min, x1_max), random.uniform(y1_min, y1_max)])
    labels.append(0)

for i in range(50):
    data_points.append([random.uniform(x2_min, x2_max), random.uniform(y2_min, y2_max)])
    labels.append(1)

train_x, test_x, train_y, test_y = train_test_split(data_points, labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x, train_y)

pred_y = knn.predict(test_x)

accuracy = accuracy_score(test_y, pred_y)

print("Предсказанные классы:", pred_y)
print("Истинные классы:", test_y)
print(f"Точность: {accuracy:.2f}")

plt.figure(figsize=(10, 6))

for i, point in enumerate(train_x):
    plt.scatter(point[0], point[1], color='blue', marker='o' if train_y[i] == 0 else 'x',
                label='train class 0' if train_y[i] == 0 and 'train class 0' not in plt.gca().get_legend_handles_labels()[1] else
                      'train class 1' if train_y[i] == 1 and 'train class 1' not in plt.gca().get_legend_handles_labels()[1] else "")

for i, point in enumerate(test_x):
    if test_y[i] == pred_y[i]:
        color, marker, label = ('green', 'o', 'test class 0 ВЕРНО') if test_y[i] == 0 else ('green', 'x', 'test class 1 ВЕРНО')
    else:
        color, marker, label = ('red', 'o', 'test class 0 НЕВЕРНО') if test_y[i] == 0 else ('red', 'x', 'test class 1 НЕВЕРНО')

    plt.scatter(point[0], point[1], color=color, marker=marker,
                label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title('Метод K-ближайших соседей sklearn')
plt.xlabel('Х координаты')
plt.ylabel('У координаты')
plt.legend()
plt.grid()
plt.show()