import tensorflow as tf
from flask import Flask
from flask import request

app = Flask(__name__)

with tf.compat.v1.gfile.FastGFile(r'F:\\工作文档\\tensorflow服务模型\\model\\linear.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
sess = tf.compat.v1.Session()
output = sess.graph.get_tensor_by_name('outputY:0')


@app.route('/predict', methods=["GET"])
def testPredict():
    inputX = request.args.get("inputX")

    return str(sess.run(output, feed_dict={'inputX:0': inputX}))

if __name__ == '__main__':
    app.run()
