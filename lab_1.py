import random
#random.seed(116)
#print(random.randint(1, 15))
#14 задача

numbers = set(map(str, range(1, 11)))
summ = 0
file = open('lab1.txt', "r")
for i in file.read().split():
    chifr = i.strip(",. ")
    if chifr in numbers:
        summ += int(chifr)
print(f'Сумма чисел в тексте: {summ}')
