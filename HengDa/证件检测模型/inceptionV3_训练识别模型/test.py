#


with open(r'test2.txt','r',encoding='utf-8') as cur:
    data=cur.readlines()[0]
print(data,)
print(type(data),len(data))