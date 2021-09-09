from flask import Flask,request,jsonify

import dill as pickle

app = Flask(__name__)

# /invocation是路由地址，methods是支持http方法，可以分为POST和GET

@app.route('/invocation',methods=["POST"])

def invocation():

    # 获得post传过来的数据

    data = request.get_json(force=True) # 读取字符串中的json数据
    print(data)
    # 加载MinMaxScalerModel和LogisticRegressionModel
    with open(r'F:\\工作文档\\tensorflow服务模型\\鸢尾花识别\\MinMaxScalerModel.pkl', 'rb') as file1:
        scaler = pickle.load(file1)
    # scaler = joblib.load('MinMaxScalerModel.pkl')
    with open(r'F:\\工作文档\\tensorflow服务模型\\鸢尾花识别\\LogisticRegressionModel.pkl', 'rb') as file2:
        model = pickle.load(file2)
    # model = joblib.load('LogisticRegressionModel.pkl')

    #数据归一化及预测

    sc_data = scaler.transform(data)

    pro = model.predict_proba(sc_data)

    info = {'result me': str(pro)}

    return info
    #return jsonify(info)#返回结果 序列化json数据，即带数据结构类型的json

#flask提供了jsonify函数供用户处理返回的序列化json数据，

#而python自带的json库中也有dumps方法可以序列化json对象.(序列化即是将结构对象变为字符串··)

if __name__ == '__main__':

    app.run(host='127.0.0.1', port=8080)
