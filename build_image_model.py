#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'
"""build model"""

import json
import os
import tensorflow as tf
import datetime
import numpy as np
from model_v2 import ResNetModel
from preprocessor import BatchPreprocessor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('training_file', '../data/mytrain.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/myval.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

tf.app.flags.DEFINE_integer('is_builder_model', 1, 'build model:1')
tf.app.flags.DEFINE_integer('is_get_feature', 0, 'get feature:1')

FLAGS = tf.app.flags.FLAGS

MODEL_VERSION = '1'
MODEL_PATH = '../export_base/'


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('resnet_depth={}\n'.format(FLAGS.resnet_depth))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('tensorboard_root_dir={}\n'.format(FLAGS.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    is_training = tf.placeholder('bool', [])
    # is_training = tf.constant(True, dtype=tf.bool)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = ResNetModel(is_training, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)

    _p, avg_pool, _f, _ = model.inference(x)
    saver = tf.train.Saver()

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    batch_preprocess = BatchPreprocessor(dataset_file_path="",
                                         num_classes=FLAGS.num_classes,
                                         output_size=[224, 224],
                                         horizontal_flip=True,
                                         shuffle=True, multi_scale=multi_scale
                                         , is_load_img=False)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())

        # Directly restore (your model should be exactly the same with checkpoint)
        saver.restore(sess, "../training/resnet_20191017_113334/checkpoint/model_epoch11.ckpt")

        if FLAGS.is_get_feature == 1:
            counter = 0

            image_list = os.listdir("/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/img")
            image_list = ["/data1/yupeng/cnn_finetune/tensorflow-cnn-finetune/img/" + i for i in image_list]
            inputs_batch = batch_preprocess.process_batch_img(image_list)

            for batch in inputs_batch:
                inputs, file_list = batch

                if (not isinstance(inputs, type(None))) and inputs.shape == (64, 224, 224, 3):
                    s, prob = sess.run([avg_pool, _p], feed_dict={x: inputs, is_training: False})

                    for file_index in range(len(file_list)):
                        _feature = s[file_index]
                        _feature = [round(_i, 6) for _i in _feature]
                        _prob = prob[file_index].tolist()
                        _pred = _prob.index(max(_prob))

                        with open("../feature_5_data_cate3/{0}.txt".format(str(_pred)), mode="a") as f1:
                            f1.writelines("{0}\t{1}\n".format(str(file_list[file_index][:-4]),
                                                              str(s[file_index].tolist()),
                                                              # str(prob[file_index].tolist())
                                                              ))
                            counter += 1

                    if counter % 1000 == 0:
                        print(counter)

        if FLAGS.is_builder_model == 1:
            # Export model
            # export_path_base = FLAGS.export_path_base
            # export_path = os.path.join(
            #     tf.compat.as_bytes(export_path_base),
            #     tf.compat.as_bytes(str(FLAGS.model_version)))
            # print('Exporting trained model to', export_path)

            # Build the signature_def_map.
            # output == tf.nn.xw_plus_b(x, weights, biases)
            builder = tf.saved_model.builder.SavedModelBuilder("".join([MODEL_PATH, MODEL_VERSION]))
            image_input = tf.saved_model.utils.build_tensor_info(x)
            # image_pooling = tf.saved_model.utils.build_tensor_info(avg_pool)
            is_train = tf.saved_model.utils.build_tensor_info(is_training)
            # fc_weight = tf.saved_model.utils.build_tensor_info(_w)
            # fc_bias = tf.saved_model.utils.build_tensor_info(_b)
            y_pred = tf.saved_model.utils.build_tensor_info(_p)
            # feature_center = tf.saved_model.utils.build_tensor_info(_f)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                  inputs={'image_input': image_input, 'is_train': is_train},
                  outputs={
                      # 'image_pooling': image_pooling,
                      # 'fc_weight': fc_weight,
                      # 'fc_bias': fc_bias,
                      'y_pred': y_pred
                      },
                  method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
            )

            builder.save()
            print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
