import pandas as pd
check_differ_df = pd.read_table('./labelfile/differentword',sep='\t',names=['origine','predict'])
origin_word_list=check_differ_df['origine'].values
predict_word_list=check_differ_df['predict'].values

ppocr_key_list=[]
ppocr_key = open('./labelfile/ppocr_keys_v1.txt',encoding='utf-8').readlines()
for single_key in ppocr_key:
    ppocr_key_list.append(single_key.strip())

checkoutword=[]

for word_id in range(len(origin_word_list)):
    for chart in  origin_word_list[word_id]:
        if chart not in predict_word_list[word_id]:
            checkoutword.append(chart)

checkoutword=set(ppocr_key_list) & set(checkoutword)

checkoutword_df=pd.DataFrame(checkoutword,columns=['word'])
label_wd_count_useful=pd.read_excel('./labelfile/label_wd_count_useful.xlsx',header=0)
result_df = label_wd_count_useful.merge(checkoutword_df,on='word',how='inner')
print(result_df)




import pandas as pd
check_differ_df = pd.read_table('./labelfile/differentword',sep='\t',names=['origine','predict'])
origin_word_list=check_differ_df['origine'].values
predict_word_list=check_differ_df['predict'].values

ppocr_key_list=[]
ppocr_key = open('./labelfile/ppocr_keys_v1.txt',encoding='utf-8').readlines()
for single_key in ppocr_key:
    ppocr_key_list.append(single_key.strip())

checkoutword=[]

for word_id in range(len(predict_word_list)):
    for chart in  predict_word_list[word_id]:
        if chart not in origin_word_list[word_id]:
            checkoutword.append(chart)

checkoutword=set(ppocr_key_list) & set(checkoutword)

checkoutword_df=pd.DataFrame(checkoutword,columns=['word'])
label_wd_count_useful=pd.read_excel('./labelfile/label_wd_count_useful.xlsx',header=0)
result_df = label_wd_count_useful.merge(checkoutword_df,on='word',how='inner')
print(result_df)

