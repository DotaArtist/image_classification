#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'
"""get image data"""


import os
import json
import pymssql
import logging
# import imagehash
import pandas as pd
# from PIL import Image
from multiprocessing import Pool
from datetime import date, datetime
from sql_script import GET_IMAGE_DATA, GET_IMAGE_HASH


now = datetime.now()
day = now.strftime('%Y-%m-%d')

IMG_PATH = "/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/img"  # image path
LOG_PATH = "/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/logs"
PID_DETAIL = "/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/config/tmp.txt"
DATA_PATH = "/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/data"  # data
DATA_FILTER_PATH = "/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/data/data_filter.txt"  # filter data
TYPE_MAP = "/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/data/map.txt"  # map
TYPE_FILTER_MAP = "/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/data/map_filter.txt"  # filter map
HASH_PATH = "/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/config/hashmap.txt"


logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

FH = logging.FileHandler("{0}/{1}.log".format(LOG_PATH, day))
FH.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
FH.setFormatter(formatter)

logger.addHandler(FH)


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def get_pid_detail():
    """
    :return:
    """
    pidurl = dict()

    with pymssql.connect('C2CSearchEngine-readonly.db.ymatou.com',
                         'C2CSearchEngineuser_r',
                         'qom4p2sIIhfFyKn1BvZl',
                         'integratedproduct', charset='utf8') as conn:
        with conn.cursor(as_dict=True) as cursor:
            sql = GET_IMAGE_DATA
            cursor.execute(sql)
            for row in cursor:
                if row["sproductid"] not in pidurl.keys():
                    pidurl[row["sproductid"]] = row

    with open("{0}".format(PID_DETAIL), "w", encoding="utf-8") as f:
        json.dump(pidurl,  default=json_serial, fp=f)

    logging.info('get data from sql server!')
    return pidurl


def get_hash_data():
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        with pymssql.connect('C2CSearchEngine-readonly.db.ymatou.com',
                             'C2CSearchEngineuser_r',
                             'qom4p2sIIhfFyKn1BvZl',
                             'integratedproduct', charset='utf8') as conn:
            with conn.cursor(as_dict=True) as cursor:
                sql = GET_IMAGE_HASH
                cursor.execute(sql)
                for row in cursor:
                    f.writelines("{0}\t{1}\n".format(row["pid"], row["imgmd5"]))


def get_img_data():
    with open(PID_DETAIL, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
    return data


def get_train_data(data, tp):

    NoneType = type(None)

    with open("{0}/data.txt".format(DATA_PATH), "w", encoding="utf-8") as f:
        with open(HASH_PATH, "w", encoding="utf-8") as f1:
            type_dict = dict()
            counter = 0

            for pid, value in data.items():

                if tp == 1:
                    if value["firstcat"] not in type_dict.keys():
                        type_dict[value["firstcat"]] = counter
                        counter += 1
                    # d = imagehash.dhash(Image.open("{0}/{1}.jpg".format(IMG_PATH, pid)))
                    # f1.writelines("{0}\t{1}\n".format(pid, str(d)))
                    f.writelines("{0}/{1}.jpg {2}\n".format(IMG_PATH, pid, type_dict[value["firstcat"]]))

                elif tp == 2:
                    if value["secondcat"] not in type_dict.keys():
                        type_dict[value["secondcat"]] = counter
                        counter += 1
                    f.writelines("{0}/{1}.jpg {2}\n".format(IMG_PATH, pid, type_dict[value["secondcat"]]))

                elif tp == 3:
                    if value["thirdcat"] not in type_dict.keys():
                        type_dict[value["thirdcat"]] = counter
                        counter += 1
                    f.writelines("{0}/{1}.jpg {2}\n".format(IMG_PATH, pid, type_dict[value["thirdcat"]]))

                elif tp == 4:

                    if isinstance(value["sbrand"], NoneType):
                        sbrand = ""
                    else:
                        sbrand = value["sbrand"]

                    if isinstance(value["sbranden"], NoneType):
                        sbranden = ""
                    else:
                        sbranden = value["sbranden"]

                    brand = "|".join([sbrand, sbranden])
                    if brand not in type_dict.keys():
                        type_dict[brand] = counter
                        counter += 1
                    f.writelines("{0}/{1}.jpg {2}\n".format(IMG_PATH, pid, type_dict[brand]))

    with open(TYPE_MAP, "w", encoding="utf-8") as ff:
        for key, value in type_dict.items():
            ff.writelines("{0}\t{1}\n".format(key, value))

    return counter


def download_img(pid, pidurl):
    try:
        if pidurl[-4:] not in [".jpg", "jpeg", ".JPG", ".png"]:
            os.system("rm \'{0}/{1}.jpg\'".format(IMG_PATH, pid))
            print("delete gif")
            # pass
        elif not os.path.isfile("{0}/{1}.jpg".format(IMG_PATH, pid)):
            os.system("wget -O \'{0}/{1}.jpg\' {2}".format(IMG_PATH, pid, pidurl))
        else:
            pass
    except:
        logger.exception("Exception Logged")


def main():
    # get_pid_detail()
    data = get_img_data()

    p = Pool(8)
    for pid, value in data.items():
        p.apply_async(download_img, args=(pid, value["spicurl"],))

    logger.info('Waiting for all subprocesses done...')
    p.close()
    p.join()
    logger.info('All subprocesses done.')


def add_filter(limit):
    image_cate = pd.read_csv("{0}/data.txt".format(DATA_PATH), sep=" ", header=None)
    image_cate.columns = ["image_path", "origin_label"]
    image_cate["origin_label"].astype(int)

    label_statistics = image_cate["origin_label"].value_counts()
    label_list = label_statistics[label_statistics >= limit].index.tolist()
    label_new_index = range(1, len(label_list) + 1)
    label_map = dict(zip(label_list, label_new_index))

    cate_map = pd.read_csv(TYPE_MAP, sep="\t", header=None)
    cate_map.columns = ["type_name", "origin_label"]
    cate_map["origin_label"].astype(int)

    image_cate["new_label"] = image_cate[["origin_label"]].applymap(lambda x: label_map[x] if x in label_map.keys() else 0)
    cate_map["new_label"] = cate_map[["origin_label"]].applymap(lambda x: label_map[x] if x in label_map.keys() else 0)

    image_cate = image_cate[image_cate["new_label"] != -1]
    cate_map = cate_map[cate_map["new_label"] != -1]

    image_cate[["image_path", "new_label"]].to_csv(DATA_FILTER_PATH, header=None, sep=" ", index=None)
    cate_map[["type_name", "new_label"]].to_csv(TYPE_FILTER_MAP, header=None, sep="\t", index=None)


if __name__ == "__main__":
    # download pid info
    get_pid_detail()

    # read pid info
    data = get_img_data()

    # set image category type
    counter = get_train_data(data, tp=3)

    # download one image
    # download_img("ffffcfb5-3867-4837-9b40-7cf0620e93f5", data)

    # download all image file
    main()

    # category filter
    add_filter(limit=1000)
