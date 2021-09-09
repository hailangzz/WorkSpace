# -*- coding: utf-8 -*-
# =================================================

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np


class PredictModelGrpc(object):
    def __init__(self, model_name, input_name, output_name, socket='0.0.0.0:8500'):
        self.socket = socket
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.request, self.stub = self.__get_request()

    def __get_request(self):
        channel = grpc.insecure_channel(self.socket, options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                                                              ('grpc.max_receive_message_length',
                                                               1024 * 1024 * 1024)])  # 可设置大小
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()

        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"

        return request, stub

    def inference(self, frames):
        self.request.inputs[self.input_name].CopyFrom(tf.make_tensor_proto(frames, dtype=tf.float32))  # images is input of model
        result = self.stub.Predict.future(self.request, 10.0)
        res = tf.make_ndarray(result.result().outputs[self.output_name])[0]
        return res

if __name__ == '__main__':
    model = PredictModelGrpc(model_name='03_win_day')
    import time
    for i in range(10):
        s = time.time()
        data = np.random.uniform(0, 1, (1, 12, 3, 3, 11))
        result = model.inference(data)
        print(result.shape)
        e = time.time()
        print(e-s)