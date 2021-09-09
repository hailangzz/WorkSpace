import os
import PIL.Image as PImage

class CreateRecDataSet:

    def create_save_picture_dir(self,):
        # print(self.originpath+'\\'+self.originpath.split('\\')[-2])
        if not os.path.exists(self.originpath+'\\'+self.originpath.split('\\')[-2]+'_train'):
            os.makedirs(self.originpath+'\\'+self.originpath.split('\\')[-2]+'_train')
        return self.originpath+'\\'+self.originpath.split('\\')[-2]+'_train'

    def __init__(self, originpath=r'G:\BaiduNetdiskDownload\icdar2017rctw_train_v1.2\train'):
        self.originpath = originpath
        self.save_picture_dir = self.create_save_picture_dir()
        self.lable_list = []

    def cut_useful_picture(self,picture_path,PicLabInfoDict):
        for sample_key in PicLabInfoDict:
            pil_im = PImage.open(picture_path)
            try:
                region1 = pil_im.crop(PicLabInfoDict[sample_key]['label_place'])
                # # print(self.originpath + '\\' + picture_path.split('\\')[-1].split('.')[0] + '_' + str(sample_key) +'.jpg')
                # # print(PicLabInfoDict[sample_key]['label'])
                region1.save(self.save_picture_dir +'\\'+ picture_path.split('\\')[-1].split('.')[0]+'_'+str(sample_key)+'.jpg')
                self.lable_list.append(self.save_picture_dir.split('\\')[-1]+'\\'+ picture_path.split('\\')[-1].split('.')[0] + '_' + str(sample_key) +'.jpg' + '\t' + PicLabInfoDict[sample_key]['label'])
                # print(self.save_picture_dir.split('\\')[-1]+'\\'+ picture_path.split('\\')[-1].split('.')[0] + '_' + str(sample_key) +'.jpg' + '\t' + PicLabInfoDict[sample_key]['label'])
            except Exception as e:
                print('出现异常:', e)

    def read_all_lable_info(self,lable_path):
        PicLabInfoDict={}
        txt_file_cur = open(lable_path,'r',encoding='utf-8')
        all_lable_info = txt_file_cur.readlines()
        lable_index=0
        for single_label in [row_label.strip() for row_label in all_lable_info]:
            lable_index+=1
            single_label_split = single_label.split(',')
            if '0' == single_label_split[8]:
                try:
                    PicLabInfoDict[lable_index]={'label_place':[min(int(single_label_split[0]),int(single_label_split[6])),min(int(single_label_split[1]),int(single_label_split[3])),max(int(single_label_split[2]),int(single_label_split[4])),max(int(single_label_split[5]),int(single_label_split[7]))],'label':single_label.split(',')[-1]}
                except :
                    pass

        self.cut_useful_picture(lable_path.replace('.txt','.jpg'), PicLabInfoDict)
        return PicLabInfoDict
                # box = (390,2000,790,2420)
                # region1 = pil_im.crop(box[0])
    def save_label_info(self):
        # print(len(self.lable_list),self.lable_list)
        lable_info_cur = open(self.originpath+'\\train_lable.txt','w',encoding='utf-8')
        for row_info in self.lable_list:
            # print(row_info)
            lable_info_cur.write(row_info)
            lable_info_cur.write('\n')
        lable_info_cur.close()

    def get_all_picture_sample(self):

        for item in os.listdir(self.originpath):
            if '.zip' not in item:
                single_person_dir = os.path.join(self.originpath, item)
                # print(single_person_dir)
                test_num = 0
                for single_picturs_path in os.listdir(single_person_dir):
                    if '.txt' in single_picturs_path:
                        self.read_all_lable_info(os.path.join(single_person_dir, single_picturs_path))
                    test_num +=1
                    # if test_num>10:
                    #     break

CRDS = CreateRecDataSet()
CRDS.get_all_picture_sample()
CRDS.save_label_info()