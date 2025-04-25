import numpy as np
import matplotlib.pyplot as plt

# Функция: f(x) = x^2 + sin(x)
# Производная: f'(x) = 2x + cos(x)
# Минимум: около x = -0.45 (найден численно)

def gradientDescend(func=lambda x: x ** 2, diffFunc=lambda x: 2 * x,
                    x0=3, speed=0.01, epochs=100):
    xList = []
    yList = []
    x = x0

    for _ in range(epochs):
        xList.append(x)
        yList.append(func(x))
        x = x - speed * diffFunc(x)

    return xList, yList


def plot_results(xList, yList, func):
    x_vals = np.linspace(min(xList) - 1, max(xList) + 1, 400)
    y_vals = func(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='Функция: $f(x) = x^2 + \sin(x)$', color='pink')
    plt.scatter(xList, yList, color='red', label='Точки градиентного спуска')
    plt.plot(xList, yList, '--', color='blue', alpha=0.5, label='Траектория спуска')

    plt.scatter([xList[0]], [yList[0]], color='yellow', s=100, label='Начальная точка')
    plt.scatter([xList[-1]], [yList[-1]], color='purple', s=100, label='Конечная точка')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Градиентный спуск')
    plt.legend()
    plt.grid(True)
    plt.show()


func = lambda x: x ** 2 + np.sin(x)
diffFunc = lambda x: 2 * x + np.cos(x)

xList, yList = gradientDescend(func, diffFunc, x0=3, speed=0.1, epochs=50)

plot_results(xList, yList, func)
print(f"Найденный минимум: x = {xList[-1]:.4f}, f(x) = {yList[-1]:.4f}")


def find_critical_speed(func, diffFunc, x0=3, epochs=100, tol=1e-2):
    low = 0.0
    high = 1.0
    target_min = -0.45

    for _ in range(20):
        mid = (low + high) / 2
        xList, _ = gradientDescend(func, diffFunc, x0, mid, epochs)
        final_x = xList[-1]

        if abs(final_x - target_min) < tol:
            high = mid
        else:
            low = mid

    return (low + high) / 2


critical_speed = find_critical_speed(func, diffFunc)
print(f"Граничное значение (speed): {critical_speed:.4f}")
