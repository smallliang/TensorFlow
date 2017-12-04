import models.tutorials.image.cifar10.cifar10 as cifar10
import models.tutorials.image.cifar10.cifar10_input as cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

cifar10.maybe_download_and_extract()

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
labels_holder = tf.placeholder(tf.int32, [batch_size])

# first conv
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=0.05, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
#
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
#
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# second conv
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=0.05, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# full connect
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# full connect
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# last layer
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

def loss_function(logits_data, label_data):
    labels = tf.cast(label_data, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_data, name='cross_entropy_per_sample')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

loss = loss_function(logits, labels_holder)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op = tf.nn.in_top_k(logits, labels_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()

image_batch, label_batch = sess.run([images_train, labels_train])
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder, logits=logits, name='cross_entropy_per_sample')
print(sess.run(reshape, feed_dict={image_holder: image_batch, labels_holder: label_batch}))

# # train
# for step in range(max_steps):
#     start_time = time.time()
#     image_batch, label_batch = sess.run([images_train, labels_train])
#     _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, labels_holder: label_batch})
#     duration = time.time() - start_time
#     if (step+1) % 10 == 0:
#         example_per_sec = batch_size / duration
#         sec_per_batch = float(duration)
#
#         format_str = ('step %d, loss = %.2f (%.1f example/sec; %.3f sec/batch)')
#         print(format_str % (step+1, loss_value, example_per_sec, sec_per_batch))
#
# # test
# num_examples = 10000
# import math
# num_iter = int(math.ceil(num_examples / batch_size))
# true_count = 0
# total_sample_count = num_iter * batch_size
# step = 0
# while step < num_iter:
#     image_batch, label_batch = sess.run([images_test, labels_test])
#     predictions = sess.run(top_k_op, feed_dict={image_holder: image_batch, labels_holder: label_batch})
#     true_count += np.sum(predictions)
#     step += 1
# precision = true_count / total_sample_count
# print('precision %.3f' % precision)

