#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'
"""client """


# import json
import requests
import numpy as np
from preprocessor import BatchPreprocessor


# class PythonObjectEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)


SERVER_URL = 'http://172.17.21.16:8501/v1/models/image_feature:predict'

multi_scale = None

data_input = BatchPreprocessor(dataset_file_path="",
                               num_classes=15,
                               output_size=[224, 224],
                               horizontal_flip=True,
                               shuffle=True, multi_scale=multi_scale
                               , is_load_img=False).process_single_img(
    # "../img/bd842007-e439-468d-9e4e-2436cf626743.jpg")
    "../img/gu_2.jpg")


def soft_max(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def main():
    data = dict()
    # data["signature_name"] = "predict"
    data["inputs"] = {"image_input": data_input.tolist(), "is_train": False}
    response = requests.post(SERVER_URL, json=data)
    print("response time,ms", response.elapsed.microseconds)

    response.raise_for_status()
    prediction = response.json()

    cate3 = prediction['outputs']["y_pred"][0]
    print(len(cate3))
    cate3 = soft_max(cate3)
    cate3 = list(zip(range(86), cate3))

    data = sorted(cate3, key=lambda x: float(x[1]), reverse=True)
    print(data[:5])
    print(prediction['outputs']['image_pooling'][0])


if __name__ == '__main__':
    main()
