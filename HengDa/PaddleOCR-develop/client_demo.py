# coding: utf8
from datetime import time
import time

import requests
import json
import cv2
import base64
import glob
import os
import time

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


# ????HTTP????
# wsi_mask_path = 'G:/FACE/facetest/front'  # ????????????·??
wsi_mask_path = r'G:\FACE\facetest\check_picture'
fileName = './test1.txt'
file=open(fileName, 'w', encoding='utf8')
push_list = []
for path in os.listdir(wsi_mask_path):
    if path.split('_')[0] not in push_list:
        picturePath = wsi_mask_path + '/' + path
        data = {
            'images': [cv2_to_base64(cv2.imread(picturePath))]}
        headers = {"Content-type": "application/json"}
        # url = "http://10.101.76.2:8868/predict/ocr_system" #测试环境
        # url = "http://10.71.4.82:8868/predict/ocr_system"  # 开发环境
        url = "http://10.71.4.136:8868/predict/ocr_system" # 生产线
        # print(data)
        time_start = time.time()
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        time_end = time.time()
        print(path)
        # print(r.json()["results"],len(r.json()["results"][0]))
        print((time_end-time_start)*1000,r.json())
        try:
            file.write(path + str(r.json()["results"]) + '\n')
        except:
            print('??')
    push_list.append(path.split('_')[0])
file.close()