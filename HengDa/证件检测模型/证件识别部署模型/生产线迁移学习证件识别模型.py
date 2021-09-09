import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
# 数据准备
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  归一化到±1之间
    # rotation_range=30,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # rotation_range=30,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(directory=r'F:\\identification_sample\\checkpictrue\\total_train',
                                  target_size=(299,299),#Inception V3规定大小
                                  batch_size=64)
val_generator = val_datagen.flow_from_directory(directory=r'F:\\identification_sample\\checkpictrue\\val_sample',
                                target_size=(299,299),
                                batch_size=64)

# 构建基础模型
base_model = InceptionV3(weights='imagenet',include_top=False)

def create_model(base_model):
    # 增加新的输出层
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
    x = Dense(1024,activation='relu')(x)
    x = Dense(528,activation='relu')(x)
    predictions = Dense(2,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=predictions)
    # plot_model(model,'tlmodel.png')
    return model

'''
这里的base_model和model里面的iv3都指向同一个地址
'''
def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 2 # max_pooling_2d_2  #前2层神经网络参数不变···
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


model=create_model(base_model)
setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=20,#800
                    epochs=1,#2
                    validation_data=val_generator,
                    validation_steps=12,#12
                    )
tf.saved_model.save(model, './saved_model/1/')



























