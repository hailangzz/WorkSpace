# 以下代码可用···
import requests
import json
import numpy as np
from PIL import Image
image = Image.open(r'F:\identification_sample\test\false\\2e75f0fc-ca16-48e7-a3f4-0e581c4a011f.jpg')
image = image.resize((299, 299))
re_img = ((np.asarray(image)/255)-0.5)*2   #((x/255)-0.5)*2
# re_img = np.asarray(image)/255
re_img=re_img[np.newaxis,:, : ,:]

import json
data = json.dumps({"signature_name": "serving_default", "instances": re_img.tolist()})
# print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

import requests
headers = {"content-type": "application/json"}
# json_response = requests.post('http://192.168.234.132:8502/v1/models/pb_models:predict', data=data, headers=headers)
json_response = requests.post('http://10.71.4.79:8501/v1/models/saved_model:predict', data=data, headers=headers)
# print(json_response)
predictions = json.loads(json_response.text)['predictions']

print(predictions)