import read_surname_info
import create_name_picture
import random
import os
import json

class CreateCardInfo:
    card_info ={}
    def __init__(self):
        # self.card_info['allsurname'] = read_surname_info.read_surname_info()
        self.card_info['allsurname'] = read_surname_info.read_onesurname_info(wordfile='./baijiaxing.txt')
        self.card_info['allsecondname'] = read_surname_info.read_secondname_info()
        self.card_info['nameinfo'] = []
        self.card_info['backgroundpicture'] = {'lable_info':{},'picture_path':[]}


    def create_card_name_random(self,surnamelist,namelist):
        for secondname in namelist:
            for createwordnum in range(250-secondname['number']):
                singlename = ''.join(random.sample(surnamelist, 1)) + secondname['word']
                self.card_info['nameinfo'].append(singlename)
        # for name_index in range(namenumber):
        #     singlename = ''.join(random.sample(surnamelist, 1)) + ''.join(random.sample(namelist))
        #     self.card_info['nameinfo'].append(singlename)

    def create_card_name(self,surnamelist,namelist):
        for surname in surnamelist:
            for secondname in namelist:
                singlename=surname+secondname
                print(surname+secondname)
                # singlename = ''.join(random.sample(surnamelist, 1)) + ''.join(random.sample(namelist, name_len-1))
                self.card_info['nameinfo'].append(singlename)

    def create_cardname_sample(self,savepath='D://cardnamedataset//'):

        lable_cur = open(savepath + 'train_namelable.txt', 'w+',encoding='utf-8')
        for name_id in range(len(self.card_info['nameinfo'])):
            lable_cur.write(savepath+'name_'+str(len(self.card_info['nameinfo'][name_id]))+'_'+str(name_id)+'.jpg'+'\t'+ self.card_info['nameinfo'][name_id])
            lable_cur.write('\n')
            background_picturepath=random.sample(self.card_info['backgroundpicture']['picture_path'], 1)[0]
            # print(background_picturepath)
            create_name_picture.generator(background_picturepath,self.card_info['nameinfo'][name_id],str(len(self.card_info['nameinfo'][name_id]))+'_'+str(name_id),savepath)

        lable_cur.close()

    def decodejsonfile(self,originpath=r'G:\BaiduNetdiskDownload\train_full_labels.json'):
        with open(originpath,'r',encoding='utf8')as fp:
            json_data = json.load(fp)
            print('这是文件中的json数据：')
            # print(json_data.keys(),len(json_data.keys()))
            for single_picture_name in json_data:
                # print(single_picture_name)
                picture_name = single_picture_name+'.jpg'
                self.card_info['backgroundpicture']['lable_info'][picture_name] = {}
                for lable_info in json_data[single_picture_name]:
                    if lable_info['illegibility'] != True:
                        self.card_info['backgroundpicture']['lable_info'][picture_name][lable_info['transcription']] = [min(lable_info['points'][0][0],lable_info['points'][3][0]),min(lable_info['points'][0][1],lable_info['points'][1][1]),max(lable_info['points'][1][0],lable_info['points'][2][0]),max(lable_info['points'][2][1],lable_info['points'][3][1])]


    def get_background_picture(self,picturepath=r'G:\BaiduNetdiskDownload\ICDAR 2019-LSVT'):

        self.decodejsonfile()
        for item in os.listdir(picturepath):
            if '_train' not in item:
                single_person_dir = os.path.join(picturepath, item)
                test_num = 0
                for single_picturs_name in os.listdir(single_person_dir):
                    test_num+=1
                    picture_path = os.path.join(single_person_dir, single_picturs_name)
                    self.card_info['backgroundpicture']['picture_path'].append(picture_path)
                    # print(picture_path)
                    # if test_num>0:
                    #     break





CCI = CreateCardInfo()
CCI.create_card_name_random(CCI.card_info['allsurname'],CCI.card_info['allsecondname'])
CCI.get_background_picture()
# CCI.create_card_name(CCI.card_info['allsurname'],CCI.card_info['allsecondname'])
CCI.create_cardname_sample()
# print(CCI.card_info['backgroundpicture']['lable_info'].keys())

