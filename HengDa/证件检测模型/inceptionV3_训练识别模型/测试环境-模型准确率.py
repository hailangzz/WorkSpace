import requests
import json
import numpy as np
from PIL import Image
import time
import os
import copy
import shutil

class ImageCheck():

    def __init__(self):
        self.image_path_dict={'true_path_list':[],'false_path_list':[]}
        self.stand_image_matrix_dict = {'true_matrix_list': [], 'false_matrix_list': []}
        self.model_accuracy={'true_accuracy':0.0,'false_accuracy':0.0}
        self.read_image_path(test_path=r"F:\identification_sample\test")
        self.stand_image_matrix()


    def read_image_path(self,test_path =r"F:\identification_sample\test"):
        for dir_name in os.listdir('F:\\identification_sample\\test'):
            father_path = os.path.join('F:\\identification_sample\\test', dir_name)
            for image_path in os.listdir(father_path):
                if 'true' in dir_name:
                    self.image_path_dict['true_path_list'].append(os.path.join(father_path, image_path))
                else:
                    self.image_path_dict['false_path_list'].append(os.path.join(father_path, image_path))

    def stand_image_matrix(self):
        for path_key in self.image_path_dict:
            for image_path in self.image_path_dict[path_key]:
                image = Image.open(image_path)
                image = image.resize((132, 132))
                re_img = ((np.asarray(image) / 255) - 0.5) * 2
                re_img = re_img[np.newaxis, :, :, :].tolist()
                if 'true' in path_key:
                    self.stand_image_matrix_dict['true_matrix_list'].append(copy.deepcopy(re_img))
                else:
                    self.stand_image_matrix_dict['false_matrix_list'].append(copy.deepcopy(re_img))


    def test_accuracy(self):
        headers = {"content-type": "application/json"}
        for matrix_class_key in self.stand_image_matrix_dict:
            match_num = 0
            matrix_number = len(self.stand_image_matrix_dict[matrix_class_key])
            for matrix_id in range(len(self.stand_image_matrix_dict[matrix_class_key])):

                push_data = json.dumps({"signature_name": "serving_default", "instances": self.stand_image_matrix_dict[matrix_class_key][matrix_id]})
                json_response = requests.post('http://10.71.4.79:8501/v1/models/saved_model:predict', data=push_data,headers=headers)
                predictions = json.loads(json_response.text)['predictions'][0]
                if 'true' in matrix_class_key:
                    if np.argmax(predictions)==1:
                        match_num+=1
                        self.model_accuracy['true_accuracy'] = match_num/matrix_number
                    else:
                        print(self.image_path_dict['true_path_list'][matrix_id])
                        # 将合格图片预测为不合格的样本拷贝出来···
                        shutil.copy(self.image_path_dict['true_path_list'][matrix_id],'F:\\identification_sample\\evaluate_model\\false_true\\'+self.image_path_dict['true_path_list'][matrix_id].split('\\')[-1])

                else:
                    if np.argmax(predictions)==0:
                        match_num+=1
                        self.model_accuracy['false_accuracy'] = match_num / matrix_number
                    else:
                        print(self.image_path_dict['false_path_list'][matrix_id])
                        # 将合格图片预测为不合格的样本拷贝出来···
                        shutil.copy(self.image_path_dict['false_path_list'][matrix_id],'F:\\identification_sample\\evaluate_model\\false_false\\'+self.image_path_dict['false_path_list'][matrix_id].split('\\')[-1])





a=ImageCheck()
print(a.image_path_dict)
a.test_accuracy()
print(a.model_accuracy)