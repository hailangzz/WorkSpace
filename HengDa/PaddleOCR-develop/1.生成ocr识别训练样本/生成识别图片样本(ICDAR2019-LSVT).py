import json
import os
import PIL.Image as PImage

# 读取json文件内容,返回字典格式
class DecodeJSONInfo:

    def __init__(self, originpath=r'G:\BaiduNetdiskDownload\train_full_labels.json',picturepath=r'G:\BaiduNetdiskDownload\ICDAR 2019-LSVT'):
        self.train_picture_info = {}
        self.save_picture_dir = ''
        self.lable_list = []
        self.originpath = originpath
        self.picturepath = picturepath
        self.save_picture_dir = self.decodejsonfile()


    def decodejsonfile(self,):
        with open(self.originpath,'r',encoding='utf8')as fp:
            json_data = json.load(fp)
            print('这是文件中的json数据：')
            # print(json_data.keys(),len(json_data.keys()))
            for single_picture_name in json_data:
                # print(single_picture_name)
                picture_name = single_picture_name+'.jpg'
                self.train_picture_info[picture_name] = {}
                for lable_info in json_data[single_picture_name]:
                    if lable_info['illegibility'] != True:
                        self.train_picture_info[picture_name][lable_info['transcription']] = [min(lable_info['points'][0][0],lable_info['points'][3][0]),min(lable_info['points'][0][1],lable_info['points'][1][1]),max(lable_info['points'][1][0],lable_info['points'][2][0]),max(lable_info['points'][2][1],lable_info['points'][3][1])]

    def cut_useful_picture(self,picture_path,PicLabInfoDict):
        # print(picture_path,PicLabInfoDict)
        key_index = 0
        for sample_key in PicLabInfoDict:

            pil_im = PImage.open(picture_path)
            try:
                region1 = pil_im.crop(PicLabInfoDict[sample_key])
                region1.save(self.save_picture_dir +'\\'+ picture_path.split('\\')[-1].split('.')[0]+'_'+str(key_index)+'.jpg')
                self.lable_list.append(picture_path.split('\\')[-1].split('.')[0]+'_'+str(key_index)+'.jpg'+'\t'+sample_key)
                key_index += 1
                if len(self.lable_list)%1000==0:
                    print(len(self.lable_list))
            except Exception as e:
                print(picture_path, PicLabInfoDict)
                print('出现异常:', e)

    def cut_train_sample(self,):
        if not os.path.exists(self.picturepath + '\\' + self.picturepath.split('\\')[-1] + '_train'):
            os.makedirs(self.picturepath + '\\' + self.picturepath.split('\\')[-1] + '_train')
            self.save_picture_dir = self.picturepath + '\\' + self.picturepath.split('\\')[-1] + '_train'
        else:
            self.save_picture_dir = self.picturepath + '\\' + self.picturepath.split('\\')[-1] + '_train'

        for item in os.listdir(self.picturepath):
            if '_train' not in item:
                single_person_dir = os.path.join(self.picturepath, item)
                test_num = 0
                for single_picturs_name in os.listdir(single_person_dir):
                    picture_path = os.path.join(single_person_dir, single_picturs_name)
                    # if '.txt' in single_picturs_path:
                    self.cut_useful_picture(picture_path,self.train_picture_info[single_picturs_name])
                    test_num += 1
                    # if test_num>10:
                    #     break

    def save_label_info(self):
        lable_info_cur = open(self.save_picture_dir+'\\train_lable.txt','w',encoding='utf-8')
        for row_info in self.lable_list:
            # print(row_info)
            lable_info_cur.write(row_info)
            lable_info_cur.write('\n')
        lable_info_cur.close()




decodejson = DecodeJSONInfo()
decodejson.cut_train_sample()
decodejson.save_label_info()
