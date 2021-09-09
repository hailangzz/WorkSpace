import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import os
import random
import numpy as np
import PIL.Image as PImage
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
class GetPictureSample:
    sample_path_dict={'sample_list':[],
                      'train_picture_path_list':[],
                      'test_picture_path_list':[],
                      'water_image_matrix':[],
                      'unwater_image_matrix':[]
                      }

    def get_picture_path(self,origin_sample_path = r'./water_picture_program_model'):
        for class_dir in os.listdir(origin_sample_path):
            if class_dir in ['yes_water_picture', 'not_water_picture']:
                if class_dir == 'yes_water_picture':

                    single_picture_class_dir = os.path.join(origin_sample_path, class_dir)
                    single_picture_name = os.listdir(single_picture_class_dir)

                    for picturename in single_picture_name:
                        single_sample_path = {'yes_water_picture_path':'','not_water_picture_path':''}
                        single_sample_path['yes_water_picture_path'] = os.path.join(single_picture_class_dir, picturename)
                        single_sample_path['not_water_picture_path'] = os.path.join(os.path.join(origin_sample_path, 'not_water_picture'), picturename)
                        self.sample_path_dict['sample_list'].append(single_sample_path)

    def get_split_sample(self,sample_total_path = sample_path_dict['sample_list'],percent=0.8):
        train_picture_path_list = [[],[]]
        test_picture_path_list = [[],[]]

        train_num = int(len(sample_total_path) * percent)
        test_num = len(sample_total_path) - train_num
        train_picture_path = sample_total_path[:train_num]
        test_picture_path = sample_total_path[-test_num:]


        for index in range(train_num):
            train_picture_path_list[0].append(train_picture_path[index]['yes_water_picture_path'])
            train_picture_path_list[1].append(train_picture_path[index]['not_water_picture_path'])

        for index in range(test_num):
            test_picture_path_list[0].append(test_picture_path[index]['yes_water_picture_path'])
            test_picture_path_list[1].append(test_picture_path[index]['not_water_picture_path'])

        self.sample_path_dict['train_picture_path_list'] = train_picture_path_list
        self.sample_path_dict['test_picture_path_list'] = test_picture_path_list

        return train_picture_path_list,test_picture_path_list

    def get_image_data(self,images_path_list,water_word_num=3):
        water_image_matrix = []
        unwater_image_matrix = []

        for image_file in images_path_list[0]:
            picture_jpg = PImage.open(image_file)
            picture_jpg = picture_jpg.convert("RGB")
            picture_reshape = picture_jpg.resize((36*water_word_num, 36))  # 图片缩放···
            image_std = np.asarray(picture_reshape)/255
            # print(image_std.shape)
            water_image_matrix.append(image_std)


        for image_file in images_path_list[1]: #仅读取10个样本进行测试训练·····
            picture_jpg = PImage.open(image_file)
            picture_jpg = picture_jpg.convert("RGB")
            picture_reshape = picture_jpg.resize((36*water_word_num, 36))  # 图片缩放···
            image_std = np.asarray(picture_reshape)/255
            unwater_image_matrix.append(image_std)

        water_image_matrix = np.array(water_image_matrix)
        unwater_image_matrix = np.array(unwater_image_matrix)
        self.sample_path_dict['water_image_matrix'] = water_image_matrix
        self.sample_path_dict['unwater_image_matrix'] = unwater_image_matrix

        return water_image_matrix,unwater_image_matrix

    def next_batch_sample(self,batch_size=200):

        indices = np.random.choice(len(self.sample_path_dict['water_image_matrix']), batch_size)  # 随机抽取batchsize大小的 组成新数组

        # print(max(indices),min(indices))
        batch_images = self.sample_path_dict['water_image_matrix'][indices]
        batch_labels = self.sample_path_dict['unwater_image_matrix'][indices]
        return batch_images, batch_labels  # 获取到一个batch的样本和标签

    def __init__(self):
        self.get_picture_path()
        self.get_split_sample()
        self.get_image_data(self.sample_path_dict['train_picture_path_list'], water_word_num=3)




