# # -*- coding: utf-8 -*-
# import tensorflow as tf
# from tensorflow.summary import FileWriter
#
# sess = tf.Session()
# tf.train.import_meta_graph("./savedmodel/model.ckpt.meta")
# FileWriter("logs/1", sess.graph)
# sess.close()
#
#
#
# # -*- coding: utf-8 -*-
# import tensorflow as tf
# from tensorflow.python import saved_model
#
# export_path = "pb_models/1"   #存储TFserving模型的目录·····
#
# graph = tf.Graph()
# saver = tf.train.import_meta_graph("./savedmodel/model.ckpt.meta", graph=graph)
# with tf.Session(graph=graph) as sess:
#     saver.restore(sess, tf.train.latest_checkpoint("./savedmodel"))
#     saved_model.simple_save(session=sess,
#                             export_dir=export_path,
#                             inputs={"contents": graph.get_operation_by_name('DecodeJpeg/contents').outputs[0]},
#                             outputs={"prob": graph.get_operation_by_name('output/prob').outputs[0]})


# -*- coding: utf-8 -*-
# @Time        : 2019/12/27 10:43
# @Author      : tianyunzqs
# @Description :

import tensorflow as tf


def restore_and_save(input_checkpoint, export_path):
    checkpoint_file = tf.train.latest_checkpoint(input_checkpoint)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print(graph.get_name_scope())
            # for node in graph.as_graph_def().node:
            #     print(node.name)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # 建立签名映射，需要包括计算图中的placeholder（ChatInputs, SegInputs, Dropout）和
            # 我们需要的结果（project/logits,crf_loss/transitions）
            """
            build_tensor_info
            建立一个基于提供的参数构造的TensorInfo protocol buffer，
            输入：tensorflow graph中的tensor；
            输出：基于提供的参数（tensor）构建的包含TensorInfo的protocol buffer

            get_operation_by_name
            通过name获取checkpoint中保存的变量，能够进行这一步的前提是在模型保存的时候给对应的变量赋予name
            """

            contents = tf.saved_model.utils.build_tensor_info(
                graph.get_operation_by_name("DecodeJpeg/contents").outputs[0])

            predict = tf.saved_model.utils.build_tensor_info(
                graph.get_operation_by_name("output/prob").outputs[0])

            """
            # signature_constants
            # SavedModel保存和恢复操作的签名常量。
            # 在序列标注的任务中，这里的method_name是"tensorflow/serving/predict"
            # """
            # 定义模型的输入输出，建立调用接口与tensor签名之间的映射
            labeling_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        "contents": contents
                    },
                    outputs={
                        "predict": predict
                    },
                    method_name="tensorflow/serving/predict"
                ))

            """
            tf.group
            创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
            """
            # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            """
            add_meta_graph_and_variables
            建立一个Saver来保存session中的变量，输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
            对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
            对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
            """
            # 建立模型名称与模型签名之间的映射
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        labeling_signature
                })

            builder.save()
            print("Build Done")


# 模型格式转换
restore_and_save(
    r'./savedmodel',
    r'pb_models/1'
)
