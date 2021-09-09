import tensorflow as tf # 以下所有代码默认导入
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
# 保存模型路径
PATH = './models'
# 创建一个变量
one = tf.Variable(2.0)
# 创建一个占位符,在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
num = tf.placeholder(tf.float32,name='input')
# 创建一个加法步骤,注意这里并没有直接计算
sum = tf.add(num,one,name='output')
# 初始化变量，如果定义Variable就必须初始化
init = tf.global_variables_initializer()
# 创建会话sess
with tf.Session() as sess:
	sess.run(init)
	# #保存SavedModel模型
	builder = tf.saved_model.builder.SavedModelBuilder(PATH)
	signature = predict_signature_def(inputs={'input':num}, outputs={'output':sum})
	builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],signature_def_map={'predict': signature})
	builder.save()
