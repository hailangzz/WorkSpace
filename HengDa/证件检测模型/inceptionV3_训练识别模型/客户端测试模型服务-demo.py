import requests
import json
import numpy as np
#
# import numpy as np
# from PIL import Image
# image = Image.open(r'F:\\identification_sample\\sample3\\132bae35f27-fe29-44a6-a631-4c7a0c4e907b.jpg')
# mat= np.array(image)
#
# mat=["99214,17000,17000,13121,99203"]
#
# url = "http://192.168.234.132:8501/v1/models/half_plus_two:predict"
# def default(obj):
# 	if isinstance(obj,(np.ndarray,)):
# 		return obj.tolist()
# s = json.dumps({"instances":{'instances':mat},"signature_name":"serving_default"})    # numpy不能转换成json格式，所以要先把numpy转换成list，再转json
# print(s)
# r = requests.post(url,data=s)
# predictions = json.loads(r.text)
# print(predictions)
# print(predictions.keys(),predictions['error'])


import requests
import numpy as np

SERVER_URL = 'http://192.168.234.132:8501/v1/models/half_plus_two:predict'


def prediction():

	predict_request = '{"instances":%s}' % str([1,2,5])
	response = requests.post(SERVER_URL, data=predict_request)
	prediction = response.json()['predictions']
	print(prediction)

if __name__ == "__main__":
	prediction()


