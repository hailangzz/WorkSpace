# -*- coding: utf-8 -*-
# =================================================

"""
2. Model Prepare:
    Train a neural network model to classify image of clothing, like sneakers and shirts.
    Finally, output a '.pb' model
"""

import tensorflow as tf

# =========================
# ===== Load dataset ======
# =========================
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print("train_images: {}, train_labels: {}".format(train_images.shape, train_labels.shape))
print("test_images: {}, test_labels: {}".format(test_images.shape, test_labels.shape))

# =========================
# ====== Preprocess =======
# =========================
train_images = train_images / 255.0
test_images = test_images / 255.0

# =========================
# ==== Build the model ====
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# =========================
# ==== Train the model ====
# =========================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))
# =========================
# ==== Save the model ====
# =========================
model.save('./saved_model/clothing/1/', save_format='tf')  # save_format: Defaults to 'tf' in TF 2.X, and 'h5' in TF 1.X.

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import tempfile
import os
MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
# export_path='.//saved_model2//clothing'
print(export_path)
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None
)

print('\nSaved model:')
