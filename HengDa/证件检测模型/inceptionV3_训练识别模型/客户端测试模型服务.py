import requests
import json
import numpy as np

# from datetime import time
# import time
# import tensorflow as tf
# import numpy as np
# import os
# from PIL import Image
# image = Image.open(r'F:\\identification_sample\\sample3\\02d241c1-7dde-4c3d-8781-2cc22e1de8dd.jpg')
# image_data = np.asarray(image)
#
# print(image_data)
# url = "http://192.168.234.132:8502/v1/models/pb_models:predict"
# def default(obj):
# 	if isinstance(obj,(np.ndarray,)):
# 		return obj.tolist()
# s = json.dumps({"instances":image_data.tolist(),"signature_name":"serving_default"})    # numpy不能转换成json格式，所以要先把numpy转换成list，再转json
# # print(s)
# r = requests.post(url,data=s)
# predictions = json.loads(r.text)
# # print(predictions)
# print(predictions)


# import json
# import requests
# import numpy as np
# def default(obj):
# 	if isinstance(obj,(np.ndarray,)):
# 		return obj.tolist()
#
# image_test = np.random.randint(0,10,(1,160,160,3))
# data = json.dumps({"signature_name": "serving_default", "instances": str([1,2,3,4,5])})
# print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))
#
#
# headers = {"content-type": "application/json"}
# json_response = requests.post('http://192.168.234.132:8502/v1/models/pb_models:predict', data=data, headers=headers)
# predictions = json.loads(json_response.text)
# print(predictions)
#



# 以下代码可用···
import requests
import json
import numpy as np
from PIL import Image
import time

image = Image.open(r'C:\Users\021206191\Desktop\合格 不合格各五张\\00f3e930-bc62-4e04-a1ff-e593b2ae05f2.JPG')
image = image.resize((299, 299))
re_img = ((np.asarray(image)/255)-0.5)*2  #((x/255)-0.5)*2
# re_img = np.asarray(image)/255
re_img=re_img[np.newaxis,:, : ,:]
np.save("./image_matrix", re_img) #保存图像矩阵数据

print(re_img.shape,type(re_img))
import json
data = json.dumps({"signature_name": "serving_default", "instances": re_img.tolist()})
with open(r'json_string','w') as cur:
    cur.write(data) #保存发送的json字符串数据：
print(type(data))
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

start_time=time.time()
import requests
headers = {"content-type": "application/json"}
json_response = requests.post('http://192.168.234.132:8501/v1/models/pb_models:predict', data=data, headers=headers)

# json_response = requests.post('http://10.71.4.85:8501/v1/models/saved_model:predict', data=data, headers=headers)
# # 负载均衡服务：
# json_response = requests.post('http://10.71.103.237:8501/v1/models/saved_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions'][0]
print(np.argmax(predictions))
end_time=time.time()
print(predictions,end_time-start_time)



