import random
#random.seed(116)
#print(random.randint(1, 15))
#14 задача
#Текст содержит слова и целые числа от 1 до 10. Найти сумму включенных в текст чисел

text = 'У меня есть текст, в котором содержатся и цифры 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 и буквы. Ещё есть числа 13 , 12  , 45 .'
numbers = set(map(str, range(1, 11)))
summ = 0
file = open('lab1.txt', "r")
for i in text.split():
    chifr = i.strip(",. ")
    if chifr in numbers:
        summ += int(chifr)
print(f'Сумма чисел в тексте: {summ}')
