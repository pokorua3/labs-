#2.	Напишите программу построения графика по имеющемуся дискретному набору известных значений. Для этого:
#Данные по выданному варианту поместите в файл.
#С помощью программы прочитайте их.
#Постройте график в Python.
#Вариант 7, данные
# Х	271	327	402	487	627	777	927	1077	1227	1527	1827
#У	240	170	140	130	120	110	100	70	40	30	20

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def read_data(a):
    with open(a, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    x = [float(num) for num in lines[0].strip().split(',')]
    y = [float(num) for num in lines[1].strip().split(',')]
    return x, y


def graphic():
    x, y = read_data('C:/Users/dasha/OneDrive/Рабочий стол/vvod_dannie.txt')
    ax.clear()
    ax.plot(x, y, marker='o', linestyle='-', color='b')
    ax.grid()
    canvas.draw()


root = tk.Tk()
root.title("График из файла")

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

graphic()
root.mainloop()
