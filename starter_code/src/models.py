import os
import sys
import math

import numpy as np
import tensorflow as tf

from src.utils import count_model_params

from src.data_utils import create_batches

class Hparams:
  # number of output classes. Must be 10 for CIFAR-10
  num_classes = 10

  # size of each train mini-batch
  batch_size = 128

  # size of each eval mini-batch
  eval_batch_size = 100

  learning_rate = 0.05

  # l2 regularization rate
  l2_reg = 1e-4

  # number of training steps
  train_steps = 50000 # 1000


def conv_net(images, labels, *args, **kwargs):
  """A conv net.

  Args:
    images: dict with ['train', 'valid', 'test'], which hold the images in the
      [N, H, W, C] format.
    labels: dict with ['train', 'valid', 'test'], holding the labels in the [N]
      format.

  Returns:
    ops: a dict that must have the following entries
      - "global_step": TF op that keeps track of the current training step
      - "train_op": TF op that performs training on [train images] and [labels]
      - "train_loss": TF op that predicts the classes of [train images]
      - "valid_acc": TF op that counts the number of correct predictions on
       [valid images]
      - "test_acc": TF op that counts the number of correct predictions on
       [test images]

  """

  # YOUR CODE HERE
  hparams = Hparams()
  images, labels = create_batches(images, labels, batch_size=hparams.batch_size,
                                  eval_batch_size=hparams.eval_batch_size)
    

  x_train, y_train = images["train"], labels["train"]
  x_valid, y_valid = images["valid"], labels["valid"]
  x_test, y_test = images["test"], labels["test"]

  H, W, C = (x_train.get_shape()[1].value,
             x_train.get_shape()[2].value,
             x_train.get_shape()[3].value)

  def _get_logits(input_layer, flag):
    
    if flag:
        dropout0 = tf.layers.dropout(inputs=input_layer, rate=0.2)
    else:
        dropout0 = input_layer
    
    conv1 = tf.layers.conv2d(
      inputs=dropout0,
      filters=96,
      kernel_size=[3, 3],
      padding="same",
      strides=1,
      activation=tf.nn.relu,
      reuse=tf.AUTO_REUSE,
      name="conv1")
    
    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=96,
      kernel_size=[3, 3],
      padding="same",
      strides=1,
      activation=tf.nn.relu,
      reuse=tf.AUTO_REUSE,
      name="conv2")

    if flag:
        dropout1 = tf.layers.dropout(inputs=conv2, rate=0.5)
    else:
        dropout1 = conv2

    pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=[3, 3], strides=2, name="pool1")
    
    conv3 = tf.layers.conv2d(
      inputs=pool1,
      filters=192,
      kernel_size=[3, 3],
      padding="same",
      strides=1,
      activation=tf.nn.relu,
      reuse=tf.AUTO_REUSE,
      name="conv3")

    conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=192,
      kernel_size=[3, 3],
      padding="same",
      strides=1,
      activation=tf.nn.relu,
      reuse=tf.AUTO_REUSE,
      name="conv4")
    
    if flag:
        dropout2 = tf.layers.dropout(inputs=conv4, rate=0.5)
    else:
        dropout2 = conv4
    
    pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=[3, 3], strides=2, name="pool2")

    conv5 = tf.layers.conv2d(
      inputs=pool2,
      filters=192,
      kernel_size=[3, 3],
      padding="valid",
      strides=1,
      activation=tf.nn.relu,
      reuse=tf.AUTO_REUSE,
      name="conv5")
    
    conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=192,
      kernel_size=[1, 1],
      padding="valid",
      strides=1,
      activation=tf.nn.relu,
      reuse=tf.AUTO_REUSE,
      name="conv6")
    
    conv7 = tf.layers.conv2d(
      inputs=conv6,
      filters=10,
      kernel_size=[1, 1],
      padding="valid",
      strides=1,
      activation=tf.nn.relu,
      reuse=tf.AUTO_REUSE,
      name="conv7")
    
    avg = tf.layers.average_pooling2d(inputs=conv7, pool_size=5, padding="valid", strides=5)

    logits = tf.reshape(avg, [-1, 10])
    
    return logits

  x_train_logits = _get_logits(x_train, True)
  x_valid_logits = _get_logits(x_valid, False)
  x_test_logits = _get_logits(x_test, False)

  global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
    
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=x_train_logits, labels=y_train)

  train_loss = tf.reduce_mean(log_probs)

  optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=hparams.learning_rate)
  
  train_op = optimizer.minimize(train_loss, global_step=global_step)
    
  # predictions and accuracies
  def _get_preds_and_accs(logits, y):
    preds = tf.argmax(logits, axis=1)
    preds = tf.to_int32(preds)
    acc = tf.equal(preds, y)
    acc = tf.to_int32(acc)
    acc = tf.reduce_sum(acc)
    return preds, acc

  valid_preds, valid_acc = _get_preds_and_accs(x_valid_logits, y_valid)
  test_preds, test_acc = _get_preds_and_accs(x_test_logits, y_test)

  # put everything into a dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "valid_acc": valid_acc,
    "test_acc": test_acc,
  }
  return ops


def feed_forward_net(images, labels, *args, **kwargs):
  """A feed_forward_net.

  Args:
    images: dict with ['train', 'valid', 'test'], which hold the images in the
      [N, H, W, C] format.
    labels: dict with ['train', 'valid', 'test'], holding the labels in the [N]
      format.

  Returns:
    ops: a dict that must have the following entries
      - "global_step": TF op that keeps track of the current training step
      - "train_op": TF op that performs training on [train images] and [labels]
      - "train_loss": TF op that predicts the classes of [train images]
      - "valid_acc": TF op that counts the number of correct predictions on
       [valid images]
      - "test_acc": TF op that counts the number of correct predictions on
       [test images]

  """

  # YOUR CODE HERE
  hparams = Hparams()
  images, labels = create_batches(images, labels, batch_size=hparams.batch_size,
                                  eval_batch_size=hparams.eval_batch_size)

  x_train, y_train = images["train"], labels["train"]
  x_valid, y_valid = images["valid"], labels["valid"]
  x_test, y_test = images["test"], labels["test"]

  H, W, C = (x_train.get_shape()[1].value,
             x_train.get_shape()[2].value,
             x_train.get_shape()[3].value)

  hidden1_units = 500
  hidden2_units = 150

  dim = H * W * C
    
  w1 = tf.Variable(tf.truncated_normal([dim, hidden1_units], stddev=1.0 / math.sqrt(float(dim))), name='w1')
  b1 = tf.Variable(tf.zeros([hidden1_units]), name='b1')

  w2 = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],stddev=1.0 / math.sqrt(float(hidden1_units))), name='w2')
  b2 = tf.Variable(tf.zeros([hidden2_units]), name='b2')

  wo = tf.Variable(tf.truncated_normal([hidden2_units, hparams.num_classes], stddev=1.0 / math.sqrt(float(hidden2_units))), name='wo')
  bo = tf.Variable(tf.zeros([hparams.num_classes]), name='bo')

  def _get_logits(x):
    x = tf.reshape(x, [-1, H * W * C])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
    yo = tf.matmul(y2, wo) + bo
    return yo
  
    
  x_train_logits = _get_logits(x_train)
  x_valid_logits = _get_logits(x_valid)
  x_test_logits = _get_logits(x_test)
  # create train_op and global_step
  global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

  print(x_train_logits.shape.as_list())
  print(y_train.shape.as_list())
  
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=x_train_logits, labels=y_train)

  train_loss = tf.reduce_mean(log_probs)
  # train_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_train, logits=x_train_logits)

  optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=hparams.learning_rate)
  
  train_op = optimizer.minimize(train_loss, global_step=global_step)
    
  # predictions and accuracies
  def _get_preds_and_accs(logits, y):
    preds = tf.argmax(logits, axis=1)
    preds = tf.to_int32(preds)
    acc = tf.equal(preds, y)
    acc = tf.to_int32(acc)
    acc = tf.reduce_sum(acc)
    return preds, acc

  valid_preds, valid_acc = _get_preds_and_accs(x_valid_logits, y_valid)
  test_preds, test_acc = _get_preds_and_accs(x_test_logits, y_test)

  # put everything into a dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "valid_acc": valid_acc,
    "test_acc": test_acc,
  }
  return ops


def softmax_classifier(images, labels, name="softmax_classifier"):
  """A softmax classifier.

  Args:
    images: dict with ['train', 'valid', 'test'], which hold the images in the
      [N, H, W, C] format.
    labels: dict with ['train', 'valid', 'test'], holding the labels in the [N]
      format.

  Returns:
    ops: a dict that must have the following entries
      - "global_step": TF op that keeps track of the current training step
      - "train_op": TF op that performs training on [train images] and [labels]
      - "train_loss": TF op that predicts the classes of [train images]
      - "valid_acc": TF op that counts the number of correct predictions on
       [valid images]
      - "test_acc": TF op that counts the number of correct predictions on
       [test images]

  """

  hparams = Hparams()
  images, labels = create_batches(images, labels, batch_size=hparams.batch_size,
                                  eval_batch_size=hparams.eval_batch_size)

  x_train, y_train = images["train"], labels["train"]
  x_valid, y_valid = images["valid"], labels["valid"]
  x_test, y_test = images["test"], labels["test"]

  H, W, C = (x_train.get_shape()[1].value,
             x_train.get_shape()[2].value,
             x_train.get_shape()[3].value)

  # create model parameters
  with tf.variable_scope(name):
    w_soft = tf.get_variable("w", [H * W * C, hparams.num_classes])

  # compute train, valid, and test logits
  def _get_logits(x):
    x = tf.reshape(x, [-1, H * W * C])
    logits = tf.matmul(x, w_soft)
    return logits

  train_logits = _get_logits(x_train)
  valid_logits = _get_logits(x_valid)
  test_logits = _get_logits(x_test)

  # create train_op and global_step
  global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                            name="global_step")
  log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=train_logits, labels=y_train)
  train_loss = tf.reduce_mean(log_probs)
  optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=hparams.learning_rate)
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions and accuracies
  def _get_preds_and_accs(logits, y):
    preds = tf.argmax(logits, axis=1)
    preds = tf.to_int32(preds)
    acc = tf.equal(preds, y)
    acc = tf.to_int32(acc)
    acc = tf.reduce_sum(acc)
    return preds, acc

  valid_preds, valid_acc = _get_preds_and_accs(valid_logits, y_valid)
  test_preds, test_acc = _get_preds_and_accs(test_logits, y_test)

  # put everything into a dict
  ops = {
    "global_step": global_step,
    "train_op": train_op,
    "train_loss": train_loss,
    "valid_acc": valid_acc,
    "test_acc": test_acc,
  }
  return ops

