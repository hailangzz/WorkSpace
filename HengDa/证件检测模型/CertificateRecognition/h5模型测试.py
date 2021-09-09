import os
from keras.models import load_model
import numpy as np
from PIL import Image
image = Image.open(r'F:\工作文档\keras迁移学习\CertificateRecognition\imbalance_train\false_sample\3fa3f976-e695-4ec6-90ca-af66b6bcd79b.jpg')
image = image.resize((132, 132))
re_img = ((np.asarray(image)/255)-0.5)*2   #((x/255)-0.5)*2
# re_img = np.asarray(image)/255
re_img=re_img[np.newaxis,:, : ,:]
#加载模型h5文件
model = load_model("./classify.h5")

predict = model.predict(re_img)
print(predict)