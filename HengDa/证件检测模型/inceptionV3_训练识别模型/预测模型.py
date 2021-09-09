import cv2
import glob
import os.path
import random
import numpy as np
import tensorflow as tf


def predict():
    check_true_sample_list=[]
    cur = open('check_true_sample.txt','a+')
    strings = ['class_b_false', 'class_b_true']

    def id_to_string(node_id):
        return strings[node_id]

    with tf.gfile.FastGFile('./pbtxt/nn_test.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('output/prob:0')
        # 遍历目录
        for root, dirs, files in os.walk('F:\identification_sample\sample3'):
            for file in files:
                # 载入图片
                image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式
                predictions = np.squeeze(predictions)  # 把结果转为1维数据

                # 打印图片路径及名称
                image_path = os.path.join(root, file)
                print(image_path)

                # 排序
                top_k = predictions.argsort()[::-1]
                print(top_k)
                for node_id in top_k:
                    # 获取分类名称
                    human_string = id_to_string(node_id)
                    # 获取该分类的置信度
                    score = predictions[node_id]
                    print('%s (score = %.5f)' % (human_string, score))


                if predictions[1]>0.95:
                    check_true_sample_list.append(image_path)
                    cur.write(image_path)
                    cur.write('\n')

                if len(check_true_sample_list)==400: #只提取500样本
                    cur.close()
                    return check_true_sample_list
                print(len(check_true_sample_list))
    #             img = cv2.imread(image_path)
    #             cv2.imshow('image', img)
    #             cv2.waitKey(0)
    # cv2.destroyAllWindows()

predict()

