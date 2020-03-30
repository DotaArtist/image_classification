# coding=utf-8
import os
"""遍历文件夹下文件列表"""

path = 'C:/Users/YP_TR/Downloads/图像人工分类20200211/预筛其他数据20200211/入院311'

file_list = os.listdir(path)
for i in file_list:
    print('{}@{}'.format("/".join([path, i]), '3'))
