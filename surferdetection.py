##
## Surfer Detection
## surferdetection.py
##
## Original Copyright 2015 The TensorFlow Authors. All Rights Reserved.
##
## Originally licensed under the Apache License, V. 2.0 (the "License");
## you may not use this file except in compliance with the License.
##
## All modifications attributed to Justin Fung, 2017.
##
## ====================================================================

"""Builds the SURFERDETECTION network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to
 # run evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Compute accuracy on batch.
 accuracy = accuracy(logits, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import surferdetection_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'surferdetection_data',
                           """Path to the SURFERDETECTION data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the SURFERDETECTION data set.
IMAGE_SIZE = surferdetection_input.IMAGE_SIZE # =80
NUM_CLASSES = surferdetection_input.NUM_CLASSES # =5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = \
               surferdetection_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN # =10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = \
               surferdetection_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL # =997

# Paramters for the training op
alpha = 5.5 # leaky RELUs, lines 219, 239 

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 6      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0025  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

#  Compressed Tarball name.
FILE_NAME = 'surferdetection-bin.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for SURFERDETECTION training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'surferdetection-batches-bin')
  images, labels = surferdetection_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for SURFERDETECTION evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'surferdetection-batches-bin')
  images, labels = surferdetection_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images):
  """Build the SURFERDETECTION model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().

  # CONVOLUTIONAL LAYER 1
  # (results in 8 (80x80) feature maps)
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[9, 9, 3, 8],
                                         stddev=0.09,
                                         wd=0.001)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.2))
    bias = tf.nn.bias_add(conv, biases)
    #conv1 = tf.nn.relu(bias, name=scope.name)
    conv1 = tf.maximum(alpha*bias, bias, name=scope.name) # leaky RELU
    _activation_summary(conv1)

  # MAX POOLING LAYER 1
  # (results in 8 (40x40) feature maps)
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # LOCAL RESPONSE NORMALIZATION
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # CONVOLUTIONAL LAYER 2
  # (results in 8 (40x40) feature maps)
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 8, 8],
                                         stddev=0.09,
                                         wd=0.001)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [8], tf.constant_initializer(0.2))
    bias = tf.nn.bias_add(conv, biases)
    #conv2 = tf.nn.relu(bias, name=scope.name)
    conv2 = tf.maximum(alpha*bias,bias,name=scope.name) #leaky RELU
    _activation_summary(conv2)

  # LOCAL RESPONSE NORMALIZATION
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, 
                    name='norm2')

  # MAX POOLING LAYER 2
  # (results in 8 (20x20) feature maps)
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # FULLY CONNECTED LAYER 1
  with tf.variable_scope('fc1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 128],
                                          stddev=0.025, wd=0.001)
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.2))
    #fc1 = tf.nn.dropout(tf.matmul(reshape, weights) + biases, keep_prob=0.75, 
                        #name=scope.name)
    fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc1)

  # FULLY CONNECTED LAYER 2
  with tf.variable_scope('fc2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[128, 128],
                                          stddev=0.025, wd=0.001)
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.2))
    #fc2 = tf.nn.dropout(tf.matmul(fc1, weights) + biases, keep_prob=0.75,
                        #name=scope.name)
    fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
    _activation_summary(fc2)

  # SOFTMAX LAYER, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [128, NUM_CLASSES],
                                          stddev=0.09, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def accuracy(logits, labels):
  """Add summary for training batch accuracy.

  Calculates accuracy on training batch.
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs()

  Returns:
    Accuracy as a float.
  """
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels,tf.int64))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  tf.scalar_summary("accuracy", accuracy)
  return accuracy


def batch_distribution(labels):
  """Calculates a mean of labels per batch.

  Useful as a debugging call for unbalanced datasets.
  Args:
    labels: Labels from distorted_inputs().

  Returns:
    Average of the labels as a float.
  """
  mean = tf.reduce_mean(tf.cast(labels,"float"))
  return mean


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in SURFERDETECTION model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train SURFERDETECTION model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate stepwise based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads_and_vars = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads_and_vars, 
                                          global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads_and_vars:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the surferdetection-batches-bin.tar.gz tarball.
     If you have already downloaded surferdetection-bin.tar.gz, please place it
     in the 'surferdetection_data' subdirectory of your current working dir.
     This tarball is NOT currently hosted on the web.  You will need to obtain
     through the author.
  """

  dest_directory = FLAGS.data_dir

  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  #filename = DATA_URL.split('/')[-1]
  filename = FILE_NAME.split('/')[-1]
  filepath = os.path.join(dest_directory,filename)
  extracted_filename = filename.split('.')[0]
  extracted_filepath = os.path.join(dest_directory,extracted_filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                      float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  if not os.path.exists(extracted_filepath):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

