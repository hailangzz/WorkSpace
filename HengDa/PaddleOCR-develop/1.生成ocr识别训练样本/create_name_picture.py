import os
import PIL.Image as PImage
from PIL import ImageFont, ImageDraw
import cv2
import numpy as np
import sys
import random
from scipy.ndimage import filters

if getattr(sys, 'frozen', None):
 base_dir = os.path.join(sys._MEIPASS, 'usedres')
else:
 base_dir = os.path.join(os.path.dirname(__file__), 'usedres')


def picture_gaussian_filter(sample_pictures):
 # 图片高斯模糊···
 gaussian_random_value = random.uniform(0.05, 1.05)
 region1 = filters.gaussian_filter(sample_pictures, gaussian_random_value)
 region1 = PImage.fromarray(region1)
 return region1



#生成身份证图片···
def generator(background_picturepath,name,name_id,savepath):
 # print fname
 # im = PImage.open(os.path.join(base_dir, 'empty.png'))
 im = PImage.open(background_picturepath)
 picture_size = im.size
 name_font = ImageFont.truetype(os.path.join(base_dir, 'hei.ttf'), 72)
 draw = ImageDraw.Draw(im)

 # 获取图像尺寸内随机坐标点
 # x_point = np.random.randint(0, picture_size[0]-73, 1)[0]  # 两个汉字的格式
 # y_point = np.random.randint(0, picture_size[1]-73, 1)[0]
 x_point = np.random.randint(0, picture_size[0] - len(name)*35 , 1)[0] #多个数字的格式
 y_point = np.random.randint(0, picture_size[1] - 73, 1)[0]

 draw.text((x_point, y_point), name, fill=(0, 0, 0), font=name_font) #姓名

 im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
 im = PImage.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

 cut_x_remove=x_point+len(name)*36                       # 确定切割图像中样本字的大小,汉字为73，数字为36
 im = im.crop([x_point, y_point,cut_x_remove,y_point+70])
 im.save(savepath+r'//number_'+str(name_id)+'.jpg')
 # im.convert('L').save('bw.jpg')
 # showinfo(u'成功', u'文件已生成到目录下,黑白bw.png和彩色color.png')


 # for identity_card_id in range(identity_card_num):
 #  name = single_person_info['info']['姓名'].values[0]
# generator(name='你好')
  # print('生成第%d张身份证图片···'% identity_card_id)

