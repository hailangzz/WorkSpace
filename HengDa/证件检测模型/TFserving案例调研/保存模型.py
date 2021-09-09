from nets.ssd_net import SSD300
from keras import backend as K
import tensorflow as tf
import os

def save_model_for_serving(verion=1, path="./serving_model/commodity/"):
    # 2、导出模型过程
    # 路径+模型名字："./model/commodity/"
    export_path = os.path.join(
        tf.compat.as_bytes(path),
        tf.compat.as_bytes(str(verion)))

    print("正在导出模型到 %s" % export_path)

    # 模型获取
    model = SSD300((300, 300, 3), num_classes=9)
    model.load_weights("./ckpt/fine_tuning/weights.13-5.18.hdf5")

    with K.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'images': model.input},
            outputs={t.name: t for t in model.outputs}
        )


if __name__ == '__main__':
    save_model_for_serving(verion=1, path="./serving_model/commodity/")
