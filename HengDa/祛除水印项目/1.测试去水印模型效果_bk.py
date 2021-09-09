import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import get_picture_sample as gps
import os
import numpy as np
import PIL.Image as PImage
from PIL import Image
# Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001         # learning rate


# picture_jpg = PImage.open(r'F:\water_picture_program\yes_water_picture\\993_water_picture.png')
picture_jpg = PImage.open(r'F:\\det_mark_img_2117.png')
picture_jpg = picture_jpg.convert("RGB")
picture_reshape = picture_jpg.resize((36*3, 36))  # 图片缩放···
picture_matrix = np.asarray(picture_reshape)
image_std=picture_matrix/255
image_std_reshape = np.array(list(image_std.reshape(-1)))
image_std_reshape=image_std_reshape[np.newaxis,:]
print(image_std_reshape.shape)

matrix_shape = image_std_reshape.shape[1]

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, matrix_shape],name='water_mark')  # value in the range of (0, 1)
tf_y = tf.placeholder(tf.float32, [None, matrix_shape],name='unwater_mark')  # value in the range of (0, 1)

# encoder
en0 = tf.layers.dense(tf_x, 2000, tf.nn.tanh)
en1 = tf.layers.dense(en0, 1000, tf.nn.tanh)
en2 = tf.layers.dense(en1, 500, tf.nn.tanh)
encoded = tf.layers.dense(en2, 250)

# decoder
de0 = tf.layers.dense(encoded, 500, tf.nn.tanh)
de1 = tf.layers.dense(de0, 1000, tf.nn.tanh)
de2 = tf.layers.dense(de1, 2000, tf.nn.tanh)
decoded = tf.layers.dense(de2, matrix_shape, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
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

    decoded_,= sess.run([decoded], {tf_x: image_std_reshape})
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