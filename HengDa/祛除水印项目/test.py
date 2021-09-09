# # import PIL.Image as PImage
# # import numpy as np
# # import torchvision.transforms as transforms
# #
# # picture_jpg = PImage.open(r'F:\行程卡.jpg')
# # picture_jpg = picture_jpg.convert("RGB")
# # # picture_jpg = PImage.open(r'D:\python_program\pythont_deal_picture\sample_picture_data\picture_sample\front\\0_0.jpg')
# # print(picture_jpg.size)
# #
# # picture_reshape = picture_jpg.resize((36*3, 36))  # 图片缩放···
# # print(picture_reshape.size)
# #
# #
# # image_std = np.asarray(picture_reshape)
# # print(image_std,image_std.shape)
# #
# #
# # input_transform = transforms.Compose([transforms.ToTensor(),])
# # image = input_transform(picture_jpg).unsqueeze(0)
# #
# # print(image.shape)
#
#
# # import numpy as np
# # a1 = np.random.choice(a=[1.1,2.1,3.1,4.1,5.1,6], size=3, replace=False, p=None)
# # b1 = np.random.choice(a=[1.1,2.1,3.1,4.1,5.1,6], size=3, replace=False, p=None)
# # print(type(a1),b1)
# # test_list = np.array([1,2,3,4,5,6,7,8])
# # a1 = np.random.choice(len(test_list), size=3, replace=False, p=None)
# # print(test_list[a1])
#
# # coding:utf-8
#
# from PIL import Image, ImageDraw, ImageFont
#
#
# def add_text_to_image(image, text):
#     font = ImageFont.truetype('C:\Windows\Fonts\STXINGKA.TTF', 36)
#
#     # 添加背景
#     new_img = Image.new('RGBA', (3000, 3000), (0, 0, 0, 0)) # 给以3000*3000的固定画布···
#     new_img.paste(image, (305,485)) # 将背景和照片粘起来
#     new_img.show() # 背景是个3倍大的白色背景···
#     # 添加水印  同样三倍尺寸的画布···
#     font_len = len(text)
#     rgba_image = new_img.convert('RGBA')
#     text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
#     image_draw = ImageDraw.Draw(text_overlay)
#
#     image_draw.text((1497, 1497), text, font=font, fill=(0, 0, 0, 50))
#     # for i in range(0, rgba_image.size[0], font_len * 40 + 100):
#     #     for j in range(0, rgba_image.size[1], 200):
#     #         print(i,j)
#     #         image_draw.text((i, j), text, font=font, fill=(0, 0, 0, 50))
#     text_overlay = text_overlay.rotate(45,center=None)
#     image_with_text = Image.alpha_composite(rgba_image, text_overlay)
#     image_with_text.show()
#
#     # 裁切图片
#     image_with_text = image_with_text.crop((image.size[0], image.size[1], image.size[0] * 2, image.size[1] * 2))
#     return image_with_text
#
#
# if __name__ == '__main__':
#     # img = Image.open("F:\img_477.jpg")
#     img = Image.open("F:\\行程卡.jpg")
#     im_after = add_text_to_image(img, u'测试使用')
#     im_after.save(u'F://测试使用.png')


"""
author: AI JUN
function: LeNet-5 by python
date: 2020/3/12
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, losses

# 1.数据集准备
(x, y), (x_val, y_val) = datasets.mnist.load_data()  # 加载数据集，返回的是两个元组，分别表示训练集和测试集
x = tf.convert_to_tensor(x, dtype=tf.float32)/255.  # 转换为张量，并缩放到0~1
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转换为张量（标签）
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 构建数据集对象
train_dataset = train_dataset.batch(32).repeat(10)  # 设置批量训练的batch为32，要将训练集重复训练10遍

# 2.搭建网络
network = Sequential([  # 搭建网络容器
    layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层，6个3*3*1卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 池化层，卷积核2*2，步长2
    layers.ReLU(),  # 激活函数
    layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层，16个3*3*6卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 池化层
    layers.ReLU(),  # 激活函数
    layers.Flatten(),  # 拉直，方便全连接层处理

    layers.Dense(120, activation='relu'),  # 全连接层，120个节点
    layers.Dense(84, activation='relu'),  # 全连接层，84个节点
    layers.Dense(10)  # 输出层，10个节点
])
network.build(input_shape=(None, 28, 28, 1))  # 定义输入,batch_size=32,输入图片大小是28*28,通道数为1。
network.summary()  # 显示出每层的待优化参数量

# 3.模型训练（计算梯度，迭代更新网络参数）
optimizer = optimizers.SGD(lr=0.01)  # 声明采用批量随机梯度下降方法，学习率=0.01

acc_meter = metrics.Accuracy()  # 新建accuracy测量器
for step, (x, y) in enumerate(train_dataset):  # 一次输入batch组数据进行训练
    with tf.GradientTape() as tape:  # 构建梯度记录环境
        x = tf.reshape(x, (32, 28, 28, 1))  # 将输入拉直，[b,28,28]->[b,784]
        # x = tf.extand_dims(x, axis=3)
        out = network(x)  # 输出[b, 10]
        y_onehot = tf.one_hot(y, depth=10)  # one-hot编码
        loss = tf.square(out - y_onehot)
        loss = tf.reduce_sum(loss)/32  # 定义均方差损失函数，注意此处的32对应为batch的大小
        grads = tape.gradient(loss, network.trainable_variables)  # 计算网络中各个参数的梯度
        optimizer.apply_gradients(zip(grads, network.trainable_variables))  # 更新网络参数
        acc_meter.update_state(tf.argmax(out, axis=1), y)  # 比较预测值与标签，并计算精确度（写入数据，进行求精度）

    if step % 200 == 0:  # 每200个step，打印一次结果
        print('Step', step, ': Loss is: ', float(loss), ' Accuracy: ', acc_meter.result().numpy())  # 读取数据
        acc_meter.reset_states()  # 清零测量器l