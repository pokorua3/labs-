import random
#random.seed(116)
#print(random.randint(1, 20))
#20 задача
#Если 1-й отрицательный элемент массива расположен до минимального, то найти сумму элементов с четными индексами, иначе - с нечетными

file = open('lab2.txt', 'r')
numbers = list(map(int, file.readline().split()))
file.close()
first_otr = -1
min_pol = -1
min_positive = float('inf')
for i in range(len(numbers)):
    if numbers[i] < 0 and first_otr == -1:
        first_otr = i
    if numbers[i] > 0 and numbers[i] < min_positive:
        min_positive = numbers[i]
        min_pol = i

if first_otr != -1 and min_pol != -1:
    if first_otr < min_pol:
        result_sum = sum(numbers[i] for i in range(0, len(numbers), 2))
    else:
        result_sum = sum(numbers[i] for i in range(1, len(numbers), 2))
else:
    result_sum = 0

file = open("lab2_result.txt", "w")
file.write(str(result_sum))
file.close()
