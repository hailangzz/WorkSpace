import warnings
warnings.filterwarnings("ignore")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import get_picture_sample as gps
import os


# Hyper Parameters
BATCH_SIZE = 120
LR = 0.0001         # learning rate

picturesample = gps.GetPictureSample()
water_image_matrix_x,unwater_image_matrix_y = picturesample.get_image_data(picturesample.sample_path_dict['train_picture_path_list'])
# water_image_matrix_x,unwater_image_matrix_y = picturesample.next_batch_sample(1)
# print(water_image_matrix_x.shape)  # (55000, 28 * 28)
# print(unwater_image_matrix_y.shape)  # (55000, 10)
matrix_shape = water_image_matrix_x.shape
print(matrix_shape)
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

if os.path.exists('net_model/water_marker_net.ckpt.index'):
    print('model exist!!!')
    saver.restore(sess, 'net_model/water_marker_net.ckpt')
else:
    sess.run(tf.global_variables_initializer())

for step in range(8000):
    b_x, b_y = picturesample.next_batch_sample(BATCH_SIZE)
    _, encoded_,loss_ = sess.run([train, net, loss], {tf_x: b_x,tf_y:b_y})

    if step % 4 == 0:  # plotting
        print('train loss: %.4f' % loss_)
        save_path = saver.save(sess, "net_model/water_marker_net.ckpt")
