#Команда №6
#Используя функцию
#y = 10 / (1 + е ^ (-0.7x + 2))
#сгенерировать исходные точки для регрессии, добавляя пуассоновскийшум к исходным данным. Сохранить полученные данные в файл
#
#Изучить модель GradientBoostingRegressor из библиотеки sklearn. Разработать модель GradientBoostingRegressor. Настроить обучение модели по эпохам. Подобрать эффективные параметры модели с применением поиска по сетке (Grid Search)
#
#Построить график зависимости значения симметричной средней абсолютной процентной ошибки (sMAPE -- symmetric Mean Absolute Percentage Error) от эпохи обучения. Построить график с ползунком для визуализации изменения итоговой кривой в зависимости от эпохи обучения
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer
from matplotlib.widgets import Slider

np.random.seed(42)
x = np.linspace(-10, 10, 200)
y_clean = 10 / (1 + np.exp(-0.7 * x + 2))
noise = np.random.poisson(1.5, size=x.shape)
y_noisy = y_clean + noise

data = pd.DataFrame({'x': x, 'y': y_noisy})
data.to_csv('noisy_data.csv', index=False)

X = x.reshape(-1, 1)
y = y_noisy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

smape_scorer = make_scorer(smape, greater_is_better=False)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4]
}
gbr = GradientBoostingRegressor()
grid = GridSearchCV(gbr, param_grid, scoring=smape_scorer, cv=3)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
smape_per_epoch = []
for n in range(1, best_model.n_estimators + 1):
    temp_model = GradientBoostingRegressor(
        n_estimators=n,
        learning_rate=best_model.learning_rate,
        max_depth=best_model.max_depth,
        random_state=42
    )
    temp_model.fit(X_train, y_train)
    y_pred = temp_model.predict(X_test)
    smape_per_epoch.append(smape(y_test, y_pred))

# --- 6. График sMAPE от эпох ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, best_model.n_estimators + 1), smape_per_epoch, marker='o')
plt.xlabel("Эпоха обучения")
plt.ylabel("sMAPE (%)")
plt.title("sMAPE от числа эпох")
plt.grid()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
line, = ax.plot(x, best_model.predict(X), label='Epoch 1')
true_line, = ax.plot(x, y_clean, label='Original Function', linestyle='--')
ax.set_title("Изменение предсказаний модели по эпохам")
ax.set_ylim([0, 12])
ax.legend()
ax.grid()

ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Epoch', 1, best_model.n_estimators, valinit=1, valstep=1)

def update(val):
    epoch = int(slider.val)
    model = GradientBoostingRegressor(
        n_estimators=epoch,
        learning_rate=best_model.learning_rate,
        max_depth=best_model.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    line.set_ydata(y_pred)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
