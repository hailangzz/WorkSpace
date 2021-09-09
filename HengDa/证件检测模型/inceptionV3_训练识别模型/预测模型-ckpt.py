import cv2
import glob
import os.path
import random
import numpy as np
import tensorflow as tf

import tensorflow as tf
with tf.Session() as sess:
    image_data = tf.gfile.FastGFile('F:\\identification_sample\\sample3\\1325d31fc33-7549-4caa-8d5b-533e6d72fc88.jpg',
                                    'rb').read()
    new_saver = tf.train.import_meta_graph('./savedmodel/model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./savedmodel'))
    graph = tf.get_default_graph()
    feed_dict = {'DecodeJpeg/contents:0': np.array(image_data)}
    op_to_restore = graph.get_tensor_by_name("output/prob:0")

    print(sess.run(op_to_restore, feed_dict))
