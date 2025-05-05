# Вариант 16
# Функция: y=ax2+bx+c, Квадратичная регрессия, Искомые параметры: a, b, c
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random

a_true = 0.5
b_true = -2
c_true = 10


def true_func(x):
    return a_true * x ** 2 + b_true * x + c_true

x_min = -10
x_max = 10
points = 50
x = np.linspace(x_min, x_max, points)
y = np.array([true_func(xi) + random.uniform(-3, 3) for xi in x])

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def get_da(x, y, a, b, c):
    return (2 / len(x)) * np.sum(x**2 * (a * x**2 + b * x + c - y))

def get_db(x, y, a, b, c):
    return (2 / len(x)) * np.sum(x * (a * x ** 2 + b * x + c - y))

def get_dc(x, y, a, b, c):
    return (2 / len(x)) * np.sum(a * x ** 2 + b * x + c - y)

speed = 0.0001
epochs = 1000
a0, b0, c0 = 0.1, 0.1, 0.1

def fit(speed, epochs, a0, b0, c0):
    a_list = [a0]
    b_list = [b0]
    c_list = [c0]

    a, b, c = a0, b0, c0
    for i in range(epochs):
        da = get_da(x, y, a, b, c)
        db = get_db(x, y, a, b, c)
        dc = get_dc(x, y, a, b, c)

        a -= speed * da
        b -= speed * db
        c -= speed * dc

        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    return a_list, b_list, c_list


a_history, b_history, c_history = fit(speed, epochs, a0, b0, c0)
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)

scatter = ax.scatter(x, y, color='blue', label='Исходные данные')

x_plot = np.linspace(x_min, x_max, 100)
line, = ax.plot(x_plot, a_history[0] * x_plot ** 2 + b_history[0] * x_plot + c_history[0],
                'r-', linewidth=2, label='Регрессия')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Квадратичная регрессия с градиентным спуском')
ax.legend()
ax.grid(True)

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Эпоха', 0, epochs, valinit=0, valstep=1)

def update(val):
    epoch = int(slider.val)
    a = a_history[epoch]
    b = b_history[epoch]
    c = c_history[epoch]
    y_pred = a * x_plot ** 2 + b * x_plot + c
    line.set_ydata(y_pred)

    current_mse = mse(a * x ** 2 + b * x + c, y)
    ax.set_title(f'Квадратичная регрессия (эпоха {epoch})\n'
                 f'a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, MSE = {current_mse:.3f}')

    fig.canvas.draw_idle()

slider.on_changed(update)
final_mse = mse(a_history[-1] * x ** 2 + b_history[-1] * x + c_history[-1], y)
print(f"Итоговые значения:")
print(f"a = {a_history[-1]:.5f} (истинное значение: {a_true})")
print(f"b = {b_history[-1]:.5f} (истинное значение: {b_true})")
print(f"c = {c_history[-1]:.5f} (истинное значение: {c_true})")
print(f"Итоговое MSE: {final_mse:.5f}")

plt.show()