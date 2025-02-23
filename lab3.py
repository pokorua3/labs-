#1.	Напишите программу, строящую график функции. Коэффициенты a,b,c и диапазон задаются с клавиатуры.
#Вариант 7, функция 	f(x)=a·b^x
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def graph():
    a = 5
    b = 3
    x = np.linspace(-10, 10, 100)
    y = a * (b ** x)
    ax.clear()
    ax.plot(x, y)
    ax.grid()
    canvas.draw()

root = tk.Tk()
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

graph()
root.mainloop()