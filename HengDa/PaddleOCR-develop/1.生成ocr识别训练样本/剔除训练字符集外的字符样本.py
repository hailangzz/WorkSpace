import pandas as pd

train_wordcount = pd.read_excel(r'./labelfile/label_wd_count.xlsx')

ppocr_word=open(r'./labelfile/ppocr_keys_v1.txt','r',encoding='utf-8').readlines()
print(train_wordcount.shape)
ppocr_word_list=[]
for word in ppocr_word:
    ppocr_word_list.append(word[:-1])

train_wordcount=train_wordcount[~train_wordcount['word'].isin(ppocr_word_list)]
print(train_wordcount.shape)
train_wordcount.to_excel(r'./labelfile/label_wd_count_unuseful.xlsx',index=None)


# 处理ocr汇总的标签文件，将无效字样本删除···
OCRTotalLabel = pd.read_csv(r'./labelfile/OCRTotalLabel.txt',names=['picturename','label'],sep='\t')

print(OCRTotalLabel['label'])