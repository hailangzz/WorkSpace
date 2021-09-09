import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
import os
from keras.optimizers import Adagrad

origin_train_path = r'./total_train'
def get_class_weight(origin_train_path):
    class_weight={}
    sample_class_num={}
    sample_directory = os.listdir(origin_train_path)
    for dir_index in range(len(sample_directory)):
        sample_class_num[dir_index]=len(os.listdir(os.path.join(origin_train_path,sample_directory[dir_index])))
        class_weight[dir_index] = 1

    for class_key in class_weight:
        class_weight[class_key]=(1/sample_class_num[class_key]*sum(sample_class_num.values()))

    return class_weight

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    zoom_range=0.2,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    zoom_range=0.2,
)

# imbalance_train
train_generator = train_datagen.flow_from_directory(directory=origin_train_path,
                                  target_size=(299,299),
                                  batch_size=32)
val_generator = val_datagen.flow_from_directory(directory=r'./val_sample',
                                target_size=(299,299),
                                batch_size=32)


base_model = InceptionV3(weights='imagenet',include_top=False)

def create_model(base_model):
    # 增加新的输出层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(2,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=predictions)
    return model

def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 10
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


model=create_model(base_model)
setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    # steps_per_epoch=200,#800
                    epochs=1,#2
                    validation_data=val_generator,
                    class_weight=get_class_weight(origin_train_path),
                    validation_steps=12,#12,
                    )

model.save('classify.h5')
tf.saved_model.save(model, './saved_model/1/')



























