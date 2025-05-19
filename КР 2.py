import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from math import exp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer


def generate_sigmoid_data(num_points=200, x_range=(-5, 10)):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y_clean = [10 / (1 + exp(-0.7 * xi + 2)) for xi in x]
    y_noisy = np.random.poisson(y_clean)
    return pd.DataFrame({'x': x, 'y_clean': y_clean, 'y_noisy': y_noisy})


def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


if __name__ == "__main__":
    data = generate_sigmoid_data()
    data.to_csv('sigmoid_data.csv', index=False)
    print("Данные сохранены в sigmoid_data.csv")

    X = data[['x']].values
    y = data['y_noisy'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0]
    }

    gbr = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        gbr,
        param_grid,
        cv=5,
        scoring=make_scorer(smape, greater_is_better=False),
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f"Лучшие параметры: {best_params}")

    model = GradientBoostingRegressor(**best_params, warm_start=True, random_state=42)

    models = []
    for epoch in range(1, best_params['n_estimators'] + 1):
        model.n_estimators = epoch
        model.fit(X_train_scaled, y_train)
        models.append((epoch, model))


    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    x_plot = np.linspace(-5, 10, 100).reshape(-1, 1)
    x_plot_scaled = scaler.transform(x_plot)

    initial_epoch = 1
    y_plot = models[initial_epoch - 1][1].predict(x_plot_scaled)
    line, = ax.plot(x_plot, y_plot, 'g-', lw=2, label='Model')
    train_scatter = ax.scatter(X_train, y_train, c='b', label='Train data')
    test_scatter = ax.scatter(X_test, y_test, c='r', label='Test data')
    ax.set_title(f'Model (epoch={initial_epoch})')
    ax.legend()
    ax.grid()

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Epoch', 1, best_params['n_estimators'], valinit=1, valstep=1)


    def update(val):
        epoch = int(slider.val)
        y_plot = models[epoch - 1][1].predict(x_plot_scaled)
        line.set_ydata(y_plot)
        ax.set_title(f'Model (epoch={epoch})')
        fig.canvas.draw_idle()


    slider.on_changed(update)
    plt.show()
