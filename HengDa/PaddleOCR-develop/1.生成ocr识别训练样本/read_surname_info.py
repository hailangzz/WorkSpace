import re
import pandas as pd

def read_surname_info():
    all_surname=[]
    surname_cur= open(r'./百家姓.txt','r',encoding='utf-8')
    surname_list=surname_cur.readlines()
    for single_surname in surname_list:
        result = re.findall(".*〔(.*)〕.*", single_surname)
        for surname in result:
            all_surname.append(surname)
    return all_surname

def read_onesurname_info(wordfile='./retrain_word.txt'):
    all_surname = []
    read_cur = open(wordfile, 'r', encoding='utf-8')
    all_onesurname=read_cur.readlines()
    # word_index = 0
    for name in all_onesurname:
        # word_index += 1
        # if word_index > 6622:
        #     print(name,name[:-1],len(name[:-1]))
        all_surname.append(name[:-1])
    return all_surname

def read_secondname_info():
    all_surname = []
    train_word_info = pd.read_excel('labelfile/label_wd_count.xlsx',header=0)
    train_word_info = train_word_info[train_word_info['number']<250]
    print(train_word_info.to_dict('records'))
    all_surname=train_word_info.to_dict('records')
    # read_cur = open('labelfile/label_wd_count.xlsx', 'r', encoding='utf-8')
    # all_onesurname=read_cur.readlines()
    # for name in all_onesurname:
    #     all_surname.append(name[:-1])
    return all_surname

def write_baijiaxing_txt(all_surname):
    write_cur = open('./baijiaxing.txt','w',encoding='utf-8')
    for surname in all_surname:
        for singleword in surname:
            write_cur.write(singleword)
            write_cur.write('\n')
    write_cur.close()

# all_surname = read_surname_info()
# write_baijiaxing_txt(all_surname)
#
#
# print(all_surname)