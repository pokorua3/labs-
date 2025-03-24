import random
import math
import matplotlib.pyplot as plt

x1_min, x1_max, y1_min, y1_max = 2, 8, 2, 8
x2_min, x2_max, y2_min, y2_max = 5, 17, 5, 17

data_points = []
labels = []

for i in range(50):
    data_points.append([random.uniform(x1_min, x1_max), random.uniform(y1_min, y1_max)])
    labels.append(0)

for i in range(50):
    data_points.append([random.uniform(x2_min, x2_max), random.uniform(y2_min, y2_max)])
    labels.append(1)

def split_data(points, labels, train_ratio):
    train_size = int(len(points) * train_ratio)
    train_x, train_y, test_x, test_y = [], [], [], []

    indices = list(range(len(points)))
    random.shuffle(indices)

    for i in range(train_size):
        train_x.append(points[indices[i]])
        train_y.append(labels[indices[i]])

    for i in range(train_size, len(points)):
        test_x.append(points[indices[i]])
        test_y.append(labels[indices[i]])

    return train_x, train_y, test_x, test_y

def knn_clas(train_x, train_y, test_x, k=3):
    predictions = []

    for test_point in test_x:
        distances = []

        for i in range(len(train_x)):
            dist = math.sqrt((test_point[0] - train_x[i][0])**2 + (test_point[1] - train_x[i][1])**2)
            distances.append((dist, train_y[i]))

        distances.sort(key=lambda d: d[0])
        nearest_neighbors = distances[:k]

        class_votes = {}
        for _, cls in nearest_neighbors:
            class_votes[cls] = class_votes.get(cls, 0) + 1

        predictions.append(max(class_votes, key=class_votes.get))

    return predictions

def calc_accuracy(true_labels, predicted_labels):
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    return correct / len(true_labels) if true_labels else 0

train_x, train_y, test_x, test_y = split_data(data_points, labels, 0.8)

pred_y = knn_clas(train_x, train_y, test_x)

accuracy = calc_accuracy(test_y, pred_y)

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

plt.title('Метод K-ближайших соседей')
plt.xlabel('Х координаты')
plt.ylabel('У координаты')
plt.legend()
plt.grid()
plt.show()
