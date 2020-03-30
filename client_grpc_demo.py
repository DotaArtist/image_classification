#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'
"""client """

import grpc
from preprocessor import BatchPreprocessor
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


SERVER_URL = '172.17.21.16:8500'
multi_scale = None

data = BatchPreprocessor(dataset_file_path="",
                         num_classes=15,
                         output_size=[224, 224],
                         horizontal_flip=True,
                         shuffle=True, multi_scale=multi_scale
                         , is_load_img=False).process_single_img(
    "../img/bd842007-e439-468d-9e4e-2436cf626743.jpg")


def main(_):
    channel = grpc.insecure_channel(SERVER_URL)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'image_feature'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['image_input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data.tolist(), shape=[1, 224, 224, 3]),
    )
    request.inputs['is_train'].CopyFrom(
        tf.contrib.util.make_tensor_proto(False),
    )
    result = stub.Predict(request, 10)  # 10 secs timeout
    # print(result)


if __name__ == '__main__':
    tf.app.run()
    # main(_="")
