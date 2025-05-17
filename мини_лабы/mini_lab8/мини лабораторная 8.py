import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import random

#исходная функция, сгенерировання DeepSeek - f(x) = sin(x) + 0.5*(x**2) - 0.1*x
def f(x):
    return np.sin(x) + 0.5 * x**2 - 0.1 * x

np.random.seed(42)
x = np.linspace(-5, 5, 100)
y = f(x) + np.random.uniform(-0.5, 0.5, 100)

X = x.reshape(-1, 1)

poly_model = make_pipeline(PolynomialFeatures(4), LinearRegression())
poly_model.fit(X, y)
y_poly = poly_model.predict(X)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)
y_ridge = ridge_model.predict(X)

svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X, y)
y_svr = svr_model.predict(X)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.scatter(x, y, color='blue', label='Исходные точки', s=10)
plt.plot(x, f(x), color='green', label='Исходная функция')
plt.plot(x, y_poly, color='red', label='Полиномиальная регрессия')
plt.title(f'Полиномиальная регрессия (MSE: {mean_squared_error(y, y_poly):.4f})')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(x, y, color='blue', label='Исходные точки', s=10)
plt.plot(x, f(x), color='green', label='Исходная функция')
plt.plot(x, y_ridge, color='red', label='Ridge регрессия')
plt.title(f'Ridge регрессия (MSE: {mean_squared_error(y, y_ridge):.4f})')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(x, y, color='blue', label='Исходные точки', s=10)
plt.plot(x, f(x), color='green', label='Исходная функция')
plt.plot(x, y_svr, color='red', label='SVR')
plt.title(f'SVR (MSE: {mean_squared_error(y, y_svr):.4f})')
plt.legend()

plt.tight_layout()
plt.show()

print("Среднеквадратичные ошибки (MSE):")
print(f"Полиномиальная регрессия: {mean_squared_error(y, y_poly):.4f}")
print(f"Ridge регрессия: {mean_squared_error(y, y_ridge):.4f}")
print(f"SVR: {mean_squared_error(y, y_svr):.4f}")
print('Для данной нелинейной функции лучше всего подошла полиномиальная регрессия,\nтк как она смогла наиболее точно аппроксимировать исходную функцию')
