from collections import Counter
import pandas as pd

all_word_list=[]

label_cur = open(r'labelfile//labels.txt','r',encoding='utf-8')
label_info = label_cur.readlines()
train_cur = open(r'labelfile//train.list','r',encoding='utf-8')
train_info = train_cur.readlines()
train_lable_cur = open(r'labelfile//train_lable.txt','r',encoding='utf-8')
train_lable_info = train_lable_cur.readlines()
train_lable_LSVT_cur = open(r'labelfile//train_lable_LSVT.txt','r',encoding='utf-8')
train_lable_LSVT_info = train_lable_LSVT_cur.readlines()
# print(train_lable_LSVT_info)

for sentens in label_info:
    for word in sentens.split('\t')[-1][:-1]:
        all_word_list.append(word)
print('pass: label_info')
for sentens in train_info:
    for word in sentens.split('\t')[-1][:-1]:
        all_word_list.append(word)
print('pass: train_info')
for sentens in train_lable_info:
    for word in sentens.split('\t')[-1][:-1]:
        all_word_list.append(word)
print('pass: train_lable_info')
for sentens in train_lable_LSVT_info:
    for word in sentens.split('\t')[-1][:-1]:
        all_word_list.append(word)
print('pass: train_lable_LSVT_info')

wd = Counter(all_word_list)
print(wd.most_common())
wd_df = pd.DataFrame(wd.most_common())
wd_df.to_excel('labelfile/label_wd_count.xlsx',index=None,header=['word','number'])