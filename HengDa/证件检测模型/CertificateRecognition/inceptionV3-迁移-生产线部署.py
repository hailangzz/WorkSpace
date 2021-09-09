import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model

from keras.optimizers import Adagrad

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    zoom_range=0.2,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    zoom_range=0.2,
)

train_generator = train_datagen.flow_from_directory(directory=r'./total_train',
                                  target_size=(299,299),
                                  batch_size=64)
val_generator = val_datagen.flow_from_directory(directory=r'./val_sample',
                                target_size=(299,299),
                                batch_size=64)


base_model = InceptionV3(weights='imagenet',include_top=False)

def create_model(base_model):
    # 增加新的输出层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(528,activation='relu')(x)
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
                    steps_per_epoch=8000,#800
                    epochs=10,#2
                    validation_data=val_generator,
                    validation_steps=12,#12
                    )
tf.saved_model.save(model, './saved_model/1/')



























