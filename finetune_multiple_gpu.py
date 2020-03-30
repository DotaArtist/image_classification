# -*- coding: utf-8 -*-
import os
import sys
import datetime
import numpy as np
import tensorflow as tf
from model_multiple_gpu import loss
from model_multiple_gpu import inference
from model_multiple_gpu import load_original_weights
from model_multiple_gpu import NUM_BLOCKS

sys.path.insert(0, '../utils')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 15, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('training_file', '../data/mytrain.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/myval.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')
tf.app.flags.DEFINE_integer('num_threads', 16, 'queue threads num')
tf.app.flags.DEFINE_integer('train_size', 700000, 'train data size')
tf.app.flags.DEFINE_integer('val_size', 70000, 'validation data size')
tf.app.flags.DEFINE_float('min_frac', 0.4, 'min_fraction_of_examples_in_queue')
tf.app.flags.DEFINE_integer('gpu_num', 1, 'gpu numbers')

FLAGS = tf.app.flags.FLAGS


def read_data(_filename_queue, multi_scale):
    reader = tf.TextLineReader()
    _, line_op = reader.read(_filename_queue)

    record_defaults = [[""], [1]]
    col_img, col_label = tf.decode_csv(line_op, record_defaults=record_defaults, field_delim=" ")

    image_content = tf.read_file(col_img)

    # precessing
    image_op = tf.image.decode_png(image_content, channels=3)
    image_op = tf.cast(image_op, tf.float32)

    if not multi_scale:
        pass
    else:
        new_size = np.random.randint(multi_scale[0], multi_scale[1], 1)[0]
        image_op = tf.random_crop(image_op, [new_size, new_size, 3])

    image_op = tf.image.resize_images(image_op, size=[256, 256], method=0)
    r_shape = tf.reshape(tf.reduce_mean(image_op, [0, 1]), [1, 1, 3])
    # image_op = image_op / r_shape * 128
    image_op = image_op - r_shape

    return image_op, col_label


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

    min_queue_examples = int(FLAGS.train_size * FLAGS.min_frac)
    min_queue_val = int(FLAGS.val_size * FLAGS.min_frac)

    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    # create queue
    filename_queue = tf.train.string_input_producer([FLAGS.training_file], num_epochs=None, shuffle=True)
    image_op, col_label = read_data(filename_queue, multi_scale)
    train_x, train_y = tf.train.shuffle_batch([image_op, col_label],
                                              batch_size=FLAGS.batch_size,
                                              num_threads=FLAGS.num_threads,
                                              capacity=min_queue_examples + 3 * FLAGS.batch_size,
                                              min_after_dequeue=min_queue_examples)

    filename_queue_val = tf.train.string_input_producer([FLAGS.val_file], num_epochs=None, shuffle=True)
    image_op_val, col_label_val = read_data(filename_queue_val, multi_scale)
    val_x, val_y = tf.train.shuffle_batch([image_op_val, col_label_val],
                                          batch_size=FLAGS.batch_size,
                                          num_threads=FLAGS.num_threads,
                                          capacity=min_queue_val + 3 * FLAGS.batch_size,
                                          min_after_dequeue=min_queue_val)

    is_training = tf.constant(True, tf.bool)
    no_training = tf.constant(False, tf.bool)

    # model
    train_layers = FLAGS.train_layers.split(',')

    train_batches_per_epoch = int(FLAGS.train_size / FLAGS.batch_size)
    val_batches_per_epoch = int(FLAGS.val_size / FLAGS.batch_size)

    train_loss, train_op = loss(train_x, train_y, FLAGS.learning_rate, train_layers, FLAGS.gpu_num,
                                is_training, NUM_BLOCKS[FLAGS.resnet_depth], FLAGS.num_classes)

    val_loss, _ = loss(val_x, val_y, FLAGS.learning_rate, train_layers, FLAGS.gpu_num,
                       no_training, NUM_BLOCKS[FLAGS.resnet_depth], FLAGS.num_classes)

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(inference(train_x,
                                                is_training=is_training,
                                                num_blocks=NUM_BLOCKS[FLAGS.resnet_depth],
                                                num_classes=FLAGS.num_classes), 1), tf.argmax(train_y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    val_correct_pred = tf.equal(tf.argmax(inference(val_x,
                                                    is_training=no_training,
                                                    num_blocks=NUM_BLOCKS[FLAGS.resnet_depth],
                                                    num_classes=FLAGS.num_classes), 1), tf.argmax(val_y, 1))

    val_accuracy = tf.reduce_mean(tf.cast(val_correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', train_loss)
    tf.summary.scalar('train_accuracy', accuracy)

    tf.summary.scalar('val_loss', val_loss)
    tf.summary.scalar('val_accuracy', val_accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        load_original_weights(sess, num_classes=FLAGS.num_classes, depth=FLAGS.resnet_depth)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))

        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))
            step = 1

            # Start training
            while step < train_batches_per_epoch:
                sess.run(train_op)

                # Logging
                if step % FLAGS.log_step == 0:
                    s = sess.run(merged_summary)
                    train_writer.add_summary(s, epoch * train_batches_per_epoch + step)

                step += 1

            # Epoch completed, start validation
            print("{} Start validation".format(datetime.datetime.now()))
            test_acc = 0.
            test_count = 0

            for _ in range(val_batches_per_epoch):
                acc = sess.run(val_accuracy)
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            s = tf.Summary(value=[
                tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
            ])
            val_writer.add_summary(s, epoch + 1)
            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            # save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch' + str(epoch + 1) + '.ckpt')
            saver.save(sess, checkpoint_path)

            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))


if __name__ == '__main__':
    tf.app.run()
