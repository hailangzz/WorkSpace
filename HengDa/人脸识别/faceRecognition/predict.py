from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import os
from utils import file_processing,image_processing
import face_recognition
resize_width = 160
resize_height = 160

def face_recognition_image(model_path,dataset_path, filename,image_path):
    # 加载数据库的数据
    dataset_emb,names_list=load_dataset(dataset_path, filename)
    # 初始化mtcnn人脸检测
    face_detect=face_recognition.Facedetection() #加载人脸检测模型
    # 初始化facenet
    face_net=face_recognition.facenetEmbedding(model_path) #加载人脸编码embedding编码模型

    image = image_processing.read_image_gbk(image_path)
    # 获取 判断标识 bounding_box crop_image
    bboxes, landmarks = face_detect.detect_face(image)
    bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height") #获取人脸图像框体位点信息
    if bboxes == [] or landmarks == []:
        print("-----no face")
        exit(0)
    print("-----image have {} faces".format(len(bboxes)))
    face_images = image_processing.get_bboxes_image(image, bboxes, resize_height, resize_width) #根据人脸框体点位信息扣取人脸图像
    face_images = image_processing.get_prewhiten_images(face_images) #人脸图像矩阵标准化
    pred_emb=face_net.get_embedding(face_images) #获取人脸embedding编码结果128维
    pred_name,pred_score=compare_embadding(pred_emb, dataset_emb, names_list)
    # 在图像上绘制人脸边框和识别的结果
    show_info=[ n+':'+str(s)[:5] for n,s in zip(pred_name,pred_score)]
    image_processing.show_image_bboxes_text("face_recognition", image,bboxes,show_info) #绘制预测人脸对应的人名可能标签值


def load_dataset(dataset_path,filename):
    '''
    加载人脸数据库
    :param dataset_path: embedding.npy文件（faceEmbedding.npy）
    :param filename: labels文件路径路径（name.txt）
    :return:
    '''
    embeddings=np.load(dataset_path)
    names_list=file_processing.read_data(filename,split=None,convertNum=False)
    return embeddings,names_list

def compare_embadding(pred_emb, dataset_emb, names_list,threshold=0.65):
    # 为bounding_box 匹配标签
    pred_num = len(pred_emb)
    dataset_num = len(dataset_emb)
    pred_name = []
    pred_score=[]
    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))  #计算预测头像embedding编码，同原始不同人名embedding编码样例的均方误差值
            dist_list.append(dist)
        min_value = min(dist_list) #均方误差最小的评价值，最接近哪个人的人脸图像
        pred_score.append(min_value)
        if (min_value > threshold):
            pred_name.append('unknow')
        else:
            pred_name.append(names_list[dist_list.index(min_value)])
    return pred_name,pred_score

if __name__=='__main__':
    model_path='models/20180408-102900'
    dataset_path='dataset/emb/faceEmbedding.npy'
    filename='dataset/emb/name.txt'
    image_path='dataset/test_images/huge.jpeg'
    # image_path = 'dataset/1.jpg'
    face_recognition_image(model_path, dataset_path, filename,image_path)