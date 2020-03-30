#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'
"""find most similar images"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class SearchImage(object):
    def __init__(self, image_relation_path):
        self.image_relation_path = image_relation_path
        self.relation = None
        self.idx = None
        self.relation_mat = None
        self.kd_tree = None

    def load_relation(self):
        data = pd.read_csv(self.image_relation_path, sep="\t", header=None)
        data.columns = ["image_path", "feature"]
        data["feature_data"] = data["feature"].apply(lambda x: eval(x))
        self.relation = data[["image_path", "feature_data"]]
        del data

    def build_kd_tree(self):
        idx = self.relation["image_path"].values.tolist()
        x = np.array(self.relation["feature_data"].values.tolist())
        self.kd_tree = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine').fit(x)
        self.idx = idx
        self.relation_mat = x

    def search_nearest(self, index_num):
        distances, indices = self.kd_tree.kneighbors(self.relation_mat[index_num:index_num + 1])
        print(indices)


if __name__ == '__main__':
    a = SearchImage(image_relation_path="../feature_cate2/5.txt")
    a.load_relation()
    a.build_kd_tree()
    a.search_nearest(5)
