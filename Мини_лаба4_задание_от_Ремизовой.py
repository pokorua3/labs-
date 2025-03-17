# Цель: построить модель линейной регрессии, используя библиотеки Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_excel("датасет-1.xlsx")
print(df.head())
print(df.dtypes)

df["price"] = df["price"].astype(str).str.replace(",", ".").astype(float)

plt.scatter(df["area"], df["price"], color="red")
plt.xlabel("Площадь (кв.м.)")
plt.ylabel("Стоимость (млн руб.)")
plt.title("Зависимость стоимости от площади")
plt.show()

reg = LinearRegression()
reg.fit(df[['area']], df['price'])

print("Коэффициент (a):", reg.coef_[0])
print("Свободный член (b):", reg.intercept_)
predicted_price_38 = reg.predict([[38]])
predicted_price_200 = reg.predict([[200]])

print("Предсказанная цена для 38 м²:", predicted_price_38[0])
print("Предсказанная цена для 200 м²:", predicted_price_200[0])

df["predicted_price"] = reg.predict(df[['area']])

print(df.head())
plt.scatter(df["area"], df["price"], color="red", label="Реальные данные")
plt.plot(df["area"], reg.predict(df[['area']]), color="blue", label="Линия регрессии")
plt.xlabel("Площадь (кв.м.)")
plt.ylabel("Стоимость (млн руб.)")
plt.legend()
plt.show()

pred = pd.read_csv("prediction_price.csv", sep=";")
pred["area"] = pred["area"].astype(float)
p = reg.predict(pred[['area']])
pred["predicted_price"] = p
print(pred.head())

pred.to_excel("new_predictions.xlsx", index=False)
