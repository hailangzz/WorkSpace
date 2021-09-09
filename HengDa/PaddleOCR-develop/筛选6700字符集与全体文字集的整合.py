import pandas as pd
import string

useful_word_dict={'useful_word':[]}
chinese_list=[]
for i in range(ord(u'\u4e00'),ord(u'\u9fa5')):
    chinese_list.append(chr(i))

ppocr_cur=open(r'.\\ppocr\utils\ppocr_keys_v1.txt','r',encoding='utf-8')
ppocr_all = ppocr_cur.readlines()
for word in ppocr_all:
    if (word[:-1] in string.digits) or (word[:-1] in string.ascii_letters) or (word[:-1] in chinese_list):
        useful_word_dict['useful_word'].append(word[:-1])
useful_word_df = pd.DataFrame(useful_word_dict)
useful_word_df.to_csv(r'./useful_word_df.txt',index=None,header=None)
useful_word_df.to_csv(r'./useful_word_df.csv',index=None,header=None)
