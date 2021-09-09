# string = '5１１３２２1970011190０0'
#
# for chart in string:
#     print(int(chart))
import random

for number_index in range(4500):
    number_len = random.randint(1, 18)
    number_random_str = ''
    for index in range(number_len):

        number_random_str+=str(random.sample(range(0, 9), 1)[0])


    print(number_random_str)