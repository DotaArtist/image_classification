# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

sys.path.insert(0, '../utils')

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
IMAGE_SIZE = 24
NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
FC_WEIGHT_STDDEV = 0.01


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def distorted_inputs(batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    train_data = pd.read_csv(FLAGS.training_file, header=None, sep=" ")
    train_data.columns = ["path", "label"]
    filenames = train_data["path"].tolist()
    labels = train_data["label"].tolist()

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope('data_augmentation'):
        # Read examples from files in the filename queue.
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def _distorted_inputs():
    images, labels = distorted_inputs(batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label


def process_img():
    pass


"""
Helper methods
"""


def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay"

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)


def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, dtype='float', initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)


def stack(x, is_training, num_blocks, stack_stride, block_filters_internal):
    for n in range(num_blocks):
        block_stride = stack_stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, is_training, block_filters_internal=block_filters_internal, block_stride=block_stride)
    return x


def block(x, is_training, block_filters_internal, block_stride):
    filters_in = x.get_shape()[-1]

    m = 4
    filters_out = m * block_filters_internal
    shortcut = x

    with tf.variable_scope('a'):
        a_conv = conv(x, ksize=1, stride=block_stride, filters_out=block_filters_internal)
        a_bn = bn(a_conv, is_training)
        a = tf.nn.relu(a_bn)

    with tf.variable_scope('b'):
        b_conv = conv(a, ksize=3, stride=1, filters_out=block_filters_internal)
        b_bn = bn(b_conv, is_training)
        b = tf.nn.relu(b_bn)

    with tf.variable_scope('c'):
        c_conv = conv(b, ksize=1, stride=1, filters_out=filters_out)
        c = bn(c_conv, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or block_stride != 1:
            shortcut_conv = conv(x, ksize=1, stride=block_stride, filters_out=filters_out)
            shortcut = bn(shortcut_conv, is_training)

    return tf.nn.relu(c + shortcut)


def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    return tf.nn.xw_plus_b(x, weights, biases)


def contains(target_str, search_arr):
    rv = False

    for search_str in search_arr:
        if search_str in target_str:
            rv = True
            break

    return rv


def main(_):
    tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
    tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
    tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training')
    tf.app.flags.DEFINE_integer('num_classes', 26, 'Number of classes')
    tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
    tf.app.flags.DEFINE_string('train_layers', 'fc', 'Finetuning layers, seperated by commas')
    tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
    tf.app.flags.DEFINE_string('training_file', '../data/mytrain.txt', 'Training dataset file')
    tf.app.flags.DEFINE_string('val_file', '../data/myval.txt', 'Validation dataset file')
    tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
    tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')
    tf.app.flags.DEFINE_integer('num_gpus', 4, """How many GPUs to use.""")

    FLAGS = tf.app.flags.FLAGS

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

    train_data = pd.read_csv(FLAGS.training_file, header=None, sep=" ")
    train_data.columns = ["path", "label"]
    filenames = train_data["path"].tolist()
    labels = train_data["label"].tolist()

    images = tf.convert_to_tensor(filenames, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    input_queue = tf.train.slice_input_producer([images, labels], num_epochs=FLAGS.num_epochs, shuffle=True)
    image, label = read_images_from_disk(input_queue)

    image = tf.image.resize_images(image, size=[128, 128, 3])

    image_batch, label_batch = tf.train.batch([image, label], batch_size=FLAGS.batch_size)

    is_training = tf.placeholder('bool', [])
    num_blocks = NUM_BLOCKS[FLAGS.resnet_depth]

    def inference(x):
        with tf.variable_scope('scale1'):
            s1_conv = conv(x, ksize=7, stride=2, filters_out=64)
            s1_bn = bn(s1_conv, is_training=is_training)
            s1 = tf.nn.relu(s1_bn)

        # Scale 2
        with tf.variable_scope('scale2'):
            s2_mp = tf.nn.max_pool(s1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            s2 = stack(s2_mp, is_training=is_training, num_blocks=num_blocks[0], stack_stride=1, block_filters_internal=64)

        # Scale 3
        with tf.variable_scope('scale3'):
            s3 = stack(s2, is_training=is_training, num_blocks=num_blocks[1], stack_stride=2, block_filters_internal=128)

        # Scale 4
        with tf.variable_scope('scale4'):
            s4 = stack(s3, is_training=is_training, num_blocks=num_blocks[2], stack_stride=2, block_filters_internal=256)

        # Scale 5
        with tf.variable_scope('scale5'):
            s5 = stack(s4, is_training=is_training, num_blocks=num_blocks[3], stack_stride=2, block_filters_internal=512)

        # post-net
        avg_pool = tf.reduce_mean(s5, reduction_indices=[1, 2], name='avg_pool')

        with tf.variable_scope('fc'):
            prob = fc(avg_pool, num_units_out=FLAGS.num_classes)
        return prob

    def loss(batch_x, batch_y=None):
        y_predict = inference(batch_x)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([cross_entropy_mean] + regularization_losses)
        return loss
