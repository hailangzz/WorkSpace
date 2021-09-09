import warnings
warnings.filterwarnings("ignore")
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import tensorflow.contrib.slim as slim

import os
import numpy as np
import PIL.Image as PImage
from PIL import Image
# Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001         # learning rate


# picture_jpg = PImage.open(r'F:\water_picture_program\yes_water_picture\\993_water_picture.png')
picture_jpg = PImage.open(r'F:\\det_mark_img_2118.png')
picture_jpg = picture_jpg.convert("RGB")
picture_reshape = picture_jpg.resize((36*3, 36))  # 图片缩放···
picture_matrix = np.asarray(picture_reshape)
image_std=picture_matrix/255

image_std_reshape=image_std[np.newaxis,:]
print(image_std_reshape.shape)

matrix_shape = image_std_reshape.shape

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, matrix_shape[1] , matrix_shape[2] ,matrix_shape[3]],name='water_mark')  # value in the range of (0, 1)
tf_y = tf.placeholder(tf.float32, [None, matrix_shape[1] ,matrix_shape[2] , matrix_shape[3]],name='unwater_mark')  # value in the range of (0, 1)


net = slim.conv2d(tf_x, 64, [3,3], 1, padding='SAME', scope='conv1')
print(net.shape)
net = slim.avg_pool2d(net, [2, 2], scope='pool2')
print(net.shape)
net = slim.conv2d(net, 128, [3,3], 1, scope='conv3')
print(net.shape)
net = slim.avg_pool2d(net, [2, 2], scope='pool4')
print(net.shape)
net = slim.conv2d(net, 256, [3,3], 1, scope='conv5')
print(net.shape)
net = slim.avg_pool2d(net, [2, 2], scope='pool5')
print(net.shape)
net = slim.conv2d(net, 512, [3,3], 1, scope='conv6')
print(net.shape)
net = slim.avg_pool2d(net, [2, 2], scope='pool6')
print(net.shape)
net = slim.flatten(net, scope='flat6')
print('flat6',net.shape)
net = slim.fully_connected(net, matrix_shape[1]*matrix_shape[2]*matrix_shape[3], scope='fc7')


# net = slim.conv2d(net, 64, [5, 5], 1, scope='conv5_1')
print(net.shape)
loss = tf.losses.mean_squared_error(labels=slim.flatten(tf_x), predictions=net)
train = tf.train.AdamOptimizer(LR).minimize(loss)

saver = tf.train.Saver()
sess = tf.Session()
with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)



    if os.path.exists('net_model/water_marker_net.ckpt.index'):
        print('model exist!!!')
        saver.restore(sess, 'net_model/water_marker_net.ckpt')
    else:
        print('model not exist!!!')

    decoded_,= sess.run([net], {tf_x: image_std_reshape})
    predict = decoded_*255


    print((image_std_reshape*255).astype(np.int16))
    print(picture_matrix)
    predict=np.squeeze(predict)
    print(predict.astype(np.int16))
    re_predict = predict.reshape(picture_matrix.shape)

    im = Image.fromarray(picture_matrix)
    im.save("your_file.jpeg")
    im = Image.fromarray(np.uint8(re_predict))
    im.save("un_your_file.jpeg")



# print(encoded_)