from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import tqdm
import PIL
import six
import glob
import numpy as np
import math
import time
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
paddle.enable_static()

## 读取样本照片数据为train.npy、test.npy
# to_npy 将图片读取完保存在.npy文件中

def read_pictrue_name(pictrue_father_path = './water_picture_program_model/not_water_picture/'):
    # print(os.listdir(pictrue_father_path))
    return os.listdir(pictrue_father_path)

def create_picutre_data_file(train_rate = 0.95,picture_size={'x_size':128,'y_size':128}):
    ratio = 0.95
    image_size = 128
    sample_pictrue_info = { 'not_water_picture_data': [], 'water_picture_data': []}


    pictrue_father_path = './water_picture_program_model/not_water_picture/'
    all_pictrue_name_list = read_pictrue_name(pictrue_father_path)

    for picture_name in all_pictrue_name_list[:]:
        try:
            water_picture_path = './water_picture_program_model/yes_water_picture/'+'/'+picture_name
            not_water_picture_path = pictrue_father_path + '/' + picture_name
            sample_pictrue_info['water_picture_data'].append(cv2.cvtColor(cv2.resize(cv2.imread(water_picture_path),(picture_size['x_size'], picture_size['y_size'])), cv2.COLOR_BGR2RGB))

            sample_pictrue_info['not_water_picture_data'].append(cv2.cvtColor(
                cv2.resize(cv2.imread(not_water_picture_path), (picture_size['x_size'], picture_size['y_size'])),
                cv2.COLOR_BGR2RGB))

        except:
            continue

    sample_pictrue_info['water_picture_data'] = np.array(sample_pictrue_info['water_picture_data'], dtype=np.float32)
    sample_pictrue_info['not_water_picture_data'] = np.array(sample_pictrue_info['not_water_picture_data'],
                                                             dtype=np.float32)
    if not os.path.exists('./npy'):
        os.mkdir('./npy')
        np.save('./npy/x_water.npy', sample_pictrue_info['water_picture_data'])
        np.save('./npy/x_not_water.npy', sample_pictrue_info['not_water_picture_data'])

# create_picutre_data_file()


# 加载数据
def load(dir_='./npy'):
    x_water = np.load(os.path.join(dir_, 'x_water.npy'))
    x_not_water = np.load(os.path.join(dir_, 'x_not_water.npy'))

    return x_water, x_not_water

# L2_loss
def L2_loss(yhat, y):
    loss = np.dot(y-yhat, y-yhat)
    loss.astype(np.float32)
    return loss

# 搭建补全网络
def generator(x):
    print('x', x.shape)
    # conv1
    conv1 = fluid.layers.conv2d(input=x,
                            num_filters=64,
                            filter_size=5,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv1',
                            data_format='NHWC')
    print('conv1', conv1.shape)
    conv1 = fluid.layers.batch_norm(conv1, momentum=0.99, epsilon=0.001)
    conv1 = fluid.layers.relu(conv1, name=None)
    # conv2
    conv2 = fluid.layers.conv2d(input=conv1,
                            num_filters=128,
                            filter_size=3,
                            dilation=1,
                            stride=2,
                            padding='SAME',
                            name='generator_conv2',
                            data_format='NHWC')
    print('conv2', conv2.shape)
    conv2 = fluid.layers.batch_norm(conv2, momentum=0.99, epsilon=0.001)
    conv2 = fluid.layers.relu(conv2, name=None)
    # conv3
    conv3 = fluid.layers.conv2d(input=conv2,
                            num_filters=128,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv3',
                            data_format='NHWC')
    print('conv3', conv3.shape)
    conv3 = fluid.layers.batch_norm(conv3, momentum=0.99, epsilon=0.001)
    conv3 = fluid.layers.relu(conv3, name=None)
    # conv4
    conv4 = fluid.layers.conv2d(input=conv3,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            stride=2,
                            padding='SAME',
                            name='generator_conv4',
                            data_format='NHWC')
    print('conv4', conv4.shape)
    conv4 = fluid.layers.batch_norm(conv4, momentum=0.99, epsilon=0.001)
    conv4 = fluid.layers.relu(conv4, name=None)
    # conv5
    conv5 = fluid.layers.conv2d(input=conv4,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv5',
                            data_format='NHWC')
    print('conv5', conv5.shape)
    conv5 = fluid.layers.batch_norm(conv5, momentum=0.99, epsilon=0.001)
    conv5 = fluid.layers.relu(conv5, name=None)
    # conv6
    conv6 = fluid.layers.conv2d(input=conv5,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv6',
                            data_format='NHWC')
    print('conv6', conv6.shape)
    conv6 = fluid.layers.batch_norm(conv6, momentum=0.99, epsilon=0.001)
    conv6 = fluid.layers.relu(conv6, name=None)

    # 空洞卷积
    # dilated1
    dilated1 = fluid.layers.conv2d(input=conv6,
                            num_filters=256,
                            filter_size=3,
                            dilation=2,
                            padding='SAME',
                            name='generator_dilated1',
                            data_format='NHWC')
    print('dilated1', dilated1.shape)
    dilated1 = fluid.layers.batch_norm(dilated1, momentum=0.99, epsilon=0.001)
    dilated1 = fluid.layers.relu(dilated1, name=None)
    # dilated2
    dilated2 = fluid.layers.conv2d(input=dilated1,
                            num_filters=256,
                            filter_size=3,
                            dilation=4,
                            padding='SAME',
                            name='generator_dilated2',
                            data_format='NHWC') #stride=1
    print('dilated2', dilated2.shape)
    dilated2 = fluid.layers.batch_norm(dilated2, momentum=0.99, epsilon=0.001)
    dilated2 = fluid.layers.relu(dilated2, name=None)
    # dilated3
    dilated3 = fluid.layers.conv2d(input=dilated2,
                            num_filters=256,
                            filter_size=3,
                            dilation=8,
                            padding='SAME',
                            name='generator_dilated3',
                            data_format='NHWC')
    print('dilated3', dilated3.shape)
    dilated3 = fluid.layers.batch_norm(dilated3, momentum=0.99, epsilon=0.001)
    dilated3 = fluid.layers.relu(dilated3, name=None)
    # dilated4
    dilated4 = fluid.layers.conv2d(input=dilated3,
                            num_filters=256,
                            filter_size=3,
                            dilation=16,
                            padding='SAME',
                            name='generator_dilated4',
                            data_format='NHWC')
    print('dilated4', dilated4.shape)
    dilated4 = fluid.layers.batch_norm(dilated4, momentum=0.99, epsilon=0.001)
    dilated4 = fluid.layers.relu(dilated4, name=None)

    # conv7
    conv7 = fluid.layers.conv2d(input=dilated4,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            name='generator_conv7',
                            data_format='NHWC')
    print('conv7', conv7.shape)
    conv7 = fluid.layers.batch_norm(conv7, momentum=0.99, epsilon=0.001)
    conv7 = fluid.layers.relu(conv7, name=None)
    # conv8
    conv8 = fluid.layers.conv2d(input=conv7,
                            num_filters=256,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv8',
                            data_format='NHWC')
    print('conv8', conv8.shape)
    conv8 = fluid.layers.batch_norm(conv8, momentum=0.99, epsilon=0.001)
    conv8 = fluid.layers.relu(conv8, name=None)
    # deconv1
    deconv1 = fluid.layers.conv2d_transpose(input=conv8,
                            num_filters=128,
                            output_size=[64,64],
                            stride = 2,
                            name='generator_deconv1',
                            data_format='NHWC')
    print('deconv1', deconv1.shape)
    deconv1 = fluid.layers.batch_norm(deconv1, momentum=0.99, epsilon=0.001)
    deconv1 = fluid.layers.relu(deconv1, name=None)
    # conv9
    conv9 = fluid.layers.conv2d(input=deconv1,
                            num_filters=128,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv9',
                            data_format='NHWC')
    print('conv9', conv9.shape)
    conv9 = fluid.layers.batch_norm(conv9, momentum=0.99, epsilon=0.001)
    conv9 = fluid.layers.relu(conv9, name=None)
    # deconv2
    deconv2 = fluid.layers.conv2d_transpose(input=conv9,
                            num_filters=64,
                            output_size=[128,128],
                            stride = 2,
                            name='generator_deconv2',
                            data_format='NHWC')
    print('deconv2', deconv2.shape)
    deconv2 = fluid.layers.batch_norm(deconv2, momentum=0.99, epsilon=0.001)
    deconv2 = fluid.layers.relu(deconv2, name=None)
    # conv10
    conv10 = fluid.layers.conv2d(input=deconv2,
                            num_filters=32,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv10',
                            data_format='NHWC')
    print('conv10', conv10.shape)
    conv10 = fluid.layers.batch_norm(conv10, momentum=0.99, epsilon=0.001)
    conv10 = fluid.layers.relu(conv10, name=None)
    # conv11
    x = fluid.layers.conv2d(input=conv10,
                            num_filters=3,
                            filter_size=3,
                            dilation=1,
                            stride=1,
                            padding='SAME',
                            name='generator_conv11',
                            data_format='NHWC')
    print('x', x.shape)
    x = fluid.layers.tanh(x)
    return x



# 搭建内容鉴别器
def discriminator(global_x):
    def global_discriminator(x):
        # conv1
        conv1 = fluid.layers.conv2d(input=x,
                        num_filters=64,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv1',
                        data_format='NHWC')
        print('conv1', conv1.shape)
        conv1 = fluid.layers.batch_norm(conv1, momentum=0.99, epsilon=0.001)
        conv1 = fluid.layers.relu(conv1, name=None)
        # conv2
        conv2 = fluid.layers.conv2d(input=conv1,
                        num_filters=128,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv2',
                        data_format='NHWC')
        print('conv2', conv2.shape)
        conv2 = fluid.layers.batch_norm(conv2, momentum=0.99, epsilon=0.001)
        conv2 = fluid.layers.relu(conv2, name=None)
        # conv3
        conv3 = fluid.layers.conv2d(input=conv2,
                        num_filters=256,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv3',
                        data_format='NHWC')
        print('conv3', conv3.shape)
        conv3 = fluid.layers.batch_norm(conv3, momentum=0.99, epsilon=0.001)
        conv3 = fluid.layers.relu(conv3, name=None)
        # conv4
        conv4 = fluid.layers.conv2d(input=conv3,
                        num_filters=512,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv4',
                        data_format='NHWC')
        print('conv4', conv4.shape)
        conv4 = fluid.layers.batch_norm(conv4, momentum=0.99, epsilon=0.001)
        conv4 = fluid.layers.relu(conv4, name=None)
        # conv5
        conv5 = fluid.layers.conv2d(input=conv4,
                        num_filters=512,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv5',
                        data_format='NHWC')
        print('conv5', conv5.shape)
        conv5 = fluid.layers.batch_norm(conv5, momentum=0.99, epsilon=0.001)
        conv5 = fluid.layers.relu(conv5, name=None)
        # conv6
        conv6 = fluid.layers.conv2d(input=conv5,
                        num_filters=512,
                        filter_size=5,
                        dilation=1,
                        stride=2,
                        padding='SAME',
                        name='discriminator_global_conv6',
                        data_format='NHWC')
        print('conv6', conv6.shape)
        conv6 = fluid.layers.batch_norm(conv6, momentum=0.99, epsilon=0.001)
        conv6 = fluid.layers.relu(conv6, name=None)
        # fc
        x = fluid.layers.fc(input=conv6,
                        size=1024,
                        name='discriminator_global_fc1')
        return x


    global_output = global_discriminator(global_x)
    print('global_output',global_output.shape)
    output = fluid.layers.fc(global_output, size=1,name='discriminator_concatenation_fc1')

    return output


# 定义域损失函数
def calc_g_loss(x, completion):
    loss = L2_loss(x, completion)
    return fluid.layers.reduce_mean(loss)

def calc_d_loss(real, fake):
    alpha = 0.1
    d_loss_real = fluid.layers.reduce_mean(fluid.layers.sigmoid_cross_entropy_with_logits(x=real, label=fluid.layers.ones_like(real)))
    d_loss_fake = fluid.layers.reduce_mean(fluid.layers.sigmoid_cross_entropy_with_logits(x=fake, label=fluid.layers.zeros_like(fake)))
    return fluid.layers.elementwise_add(d_loss_real, d_loss_fake) * alpha


LEARNING_RATE=1e-3 # 学习率
BATCH_SIZE=64 # 样本数
use_gpu=False
picture_size={'x_size':128,'y_size':128}

# 定义program
d_program = fluid.Program()
dg_program = fluid.Program()

# 定义判别器的program
with fluid.program_guard(d_program):
    # 原始数据
    x_water = fluid.layers.data(name='completion',shape=[picture_size['y_size'], picture_size['x_size'], 3],dtype='float32')
    # 全局生成图
    y_not_water = fluid.layers.data(name='y_not_water',shape=[picture_size['y_size'], picture_size['x_size'], 3],dtype='float32')
    real = discriminator(y_not_water)
    # 生成图fc
    fake = discriminator(x_water)

    # 计算生成图片被判别为真实样本的loss
    d_loss = calc_d_loss(real, fake)

# 定义判别生成图片的program
with fluid.program_guard(dg_program):
    # 原始数据
    x_water = fluid.layers.data(name='x_water', shape=[picture_size['y_size'], picture_size['x_size'], 3], dtype='float32')
    y_not_water = fluid.layers.data(name='y_not_water', shape=[picture_size['y_size'], picture_size['x_size'], 3],
                                    dtype='float32')
    # print('input_data',input_data)
    completion = generator(x_water)

    g_program = dg_program.clone()
    g_program_test = dg_program.clone(for_test=True)

    # 得到原图和修复图片的loss
    dg_loss = calc_g_loss(y_not_water, completion)
    print('g_loss_shape:', dg_loss.shape)

opt = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)
opt.minimize(loss=d_loss)
parameters = [p.name for p in g_program.global_block().all_parameters()]
opt.minimize(loss=dg_loss, parameter_list=parameters)



# 数据集标准化
x_water, x_not_water = load()
#print (x_train.shape)
x_water = np.array([a / 127.5 - 1 for a in x_water])
#print (x_train[0])
x_not_water = np.array([a / 127.5 - 1 for a in x_not_water])

# 初始化
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 加载模型
save_pretrain_model_path = 'models/'
fluid.io.load_params(executor=exe, dirname=save_pretrain_model_path, main_program=dg_program)

# 生成器优先迭代次数
NUM_TRAIN_TIMES_OF_DG = 200
# 总迭代轮次
epoch = 2000

step_num = int(len(x_water) / BATCH_SIZE)

for pass_id in range(epoch):
    # 训练生成器
    if pass_id > NUM_TRAIN_TIMES_OF_DG:
        g_loss_value = 0
        for i in tqdm.tqdm(range(step_num)):
            x_batch = x_water[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            y_batch = x_not_water[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            # print(x_batch.shape)
            # print(mask_batch.shape)
            dg_loss_n = exe.run(dg_program,
                                feed={
                                    'x_water': x_batch,
                                    'y_not_water': y_batch,
                                       },
                                fetch_list=[dg_loss])[0]
            g_loss_value += dg_loss_n
        print('Pass_id:{}, Completion loss: {}'.format(pass_id, g_loss_value))


        save_pretrain_model_path = 'models/'
        # 创建保持模型文件目录
        # os.makedirs(save_pretrain_model_path)
        fluid.io.save_params(executor=exe, dirname=save_pretrain_model_path, main_program=dg_program)

    # 生成器判断器一起训练
    else:
        g_loss_value = 0
        d_loss_value = 0
        for i in tqdm.tqdm(range(step_num)):
            x_batch = x_water[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            y_batch = x_not_water[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            dg_loss_n = exe.run(dg_program,
                                feed={
                                    'x_water': x_batch,
                                    'y_not_water': y_batch,
                                       },
                                fetch_list=[dg_loss])[0]
            g_loss_value += dg_loss_n

            completion_n = exe.run(dg_program, feed={'x_water': x_batch,'y_not_water': y_batch,},fetch_list=[completion])[0]


            d_loss_n = exe.run(d_program,
                               feed={
                                   'completion': completion_n,
                                   'y_not_water': y_batch,
                               },
                               fetch_list=[d_loss])[0]
            d_loss_value += d_loss_n

        print('Pass_id:{}, Completion loss: {}'.format(pass_id, g_loss_value))
        print('Pass_id:{}, Discriminator loss: {}'.format(pass_id, d_loss_value))

        save_pretrain_model_path = 'models/'
        # 创建保持模型文件目录
        # os.makedirs(save_pretrain_model_path)
        fluid.io.save_params(executor=exe, dirname=save_pretrain_model_path, main_program=dg_program)

