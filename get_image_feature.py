"""training"""
import os
import numpy as np
import tensorflow as tf
import datetime
from model_v2 import ResNetModel
from preprocessor import BatchPreprocessor
from sklearn.metrics import classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 4, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_string('is_pretrain', '1', 'use pretrain model')
tf.app.flags.DEFINE_float('alpha_label_smooth', 0., 'label smooth')
tf.app.flags.DEFINE_string('train_layers', 'fc,scale5/block3', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('train_file', '../data/train_13645_type7.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/dev_3000_type7.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 5, 'Logging period in terms of iteration')

FLAGS = tf.app.flags.FLAGS


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
    y = tf.placeholder(tf.int32, [None, FLAGS.num_classes])
    y_label = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder('bool', [])
    # is_training = tf.constant(True, dtype=tf.bool)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = ResNetModel(is_training, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)

    prob, avg_pool,  _a, _b = model.inference(x)
    loss = model.loss(x, y)
    train_op = model.optimize(FLAGS.learning_rate, train_layers)

    # Training accuracy of the model
    softmax_prob = tf.nn.softmax(prob)
    predict_label = tf.argmax(softmax_prob, 1)
    correct_pred = tf.equal(tf.argmax(model.prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', loss)
    # tf.summary.histogram('softmax_prob', softmax_prob)
    # tf.summary.histogram('fc_weights', _a)
    tf.summary.scalar('train_accuracy', accuracy)

    # tf.summary.histogram('avg_pool', avg_pool)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver(max_to_keep=20)

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    # print(tf.trainable_variables())
    # assert 1 == 2

    train_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.train_file,
                                           num_classes=FLAGS.num_classes,
                                           output_size=[224, 224],
                                           horizontal_flip=False,
                                           shuffle=True, multi_scale=multi_scale, alpha_label_smooth=FLAGS.alpha_label_smooth)

    val_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.val_file,
                                         num_classes=FLAGS.num_classes,
                                         output_size=[224, 224], alpha_label_smooth=FLAGS.alpha_label_smooth)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)

    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        if FLAGS.is_pretrain == '1':
            model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "../training/resnet_20200214_222540/checkpoint/model_epoch15.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))

        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1

            # Start training
            while step < train_batches_per_epoch:
                batch_xs, batch_ys, batch_y_label = train_preprocessor.next_batch(FLAGS.batch_size)
                ss_loss, _ = sess.run([loss, train_op], feed_dict={x: batch_xs, y: batch_ys, is_training: True, y_label: batch_y_label})

                # Logging
                if step % FLAGS.log_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, is_training: False, y_label: batch_y_label})
                    train_writer.add_summary(s, epoch * train_batches_per_epoch + step)
                    print("{} Epoch number: {}, Step number: {}, Loss: {}".format(
                        datetime.datetime.now(), epoch + 1, step, ss_loss))
                step += 1

            # Epoch completed, start validation
            print("{} Start validation".format(datetime.datetime.now()))
            test_acc = 0.
            test_count = 0.001
            test_y_list, test_y_pre_list = [], []

            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty, batch_ty_label = val_preprocessor.next_batch(FLAGS.batch_size)
                _prob, _predict, _softmax_prob, acc = sess.run([prob, predict_label, softmax_prob, accuracy], feed_dict={x: batch_tx, y: batch_ty, is_training: False, y_label: batch_ty_label})
                test_y_pre_list.extend(_predict)
                test_y_list.extend(batch_ty_label)
                # print(_predict, batch_ty_label, acc)
                test_acc += acc
                test_count += 1

            test_acc /= test_count
            s = tf.Summary(value=[
                tf.Summary.Value(tag="validation_total_accuracy", simple_value=test_acc),
                tf.Summary.Value(tag="validation_accuracy", simple_value=acc),
            ])
            val_writer.add_summary(s, epoch+1)
            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
            print("{} Validation F1 values:".format(datetime.datetime.now()), classification_report(y_true=test_y_list, y_pred=test_y_pre_list))

            # Reset the dataset pointers
            val_preprocessor.reset_pointer()
            train_preprocessor.reset_pointer()

            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

            # save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            saver.save(sess, checkpoint_path)

            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))


if __name__ == '__main__':
    tf.app.run()
