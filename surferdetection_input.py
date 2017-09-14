##
## Surfer Detection
## surferdetection_input.py
##
## Original Copyright 2015 The TensorFlow Authors. All Rights Reserved.
##
## Originally licensed under the Apache License, V. 2.0 (the "License");
## you may not use this file except in compliance with the License.
##
## All modifications attributed to Justin Fung, 2017.
##
## ====================================================================

"""Routine for decoding the SURFERDETECTION binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from PIL import Image

import surferdetection_augmentation

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('distortion', True,
                            """Whether or not to distort input.""")
tf.app.flags.DEFINE_boolean('oversample', False,
                            """Whether or not to oversample input.""")
tf.app.flags.DEFINE_integer('num_bins', 5,
                            """number of binary bins for training set""")

# Process images of this size. Note that... if one alters this number, 
# then the entire model architecture will change and any model would 
# need to be retrained.
IMAGE_SIZE = 80

# Global constants describing the SURFERDETECTION data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 500

NUM_TOT_IMAGES = 10500
NUM_TOT_IMG_PER_CLASS = [5250,5250]

NUM_TRAIN_IMAGES = 10000
NUM_TRAIN_IMG_PER_CLASS = [5000,5000]

NUM_EVAL_IMAGES = 500
NUM_EVAL_IMG_PER_CLASS = [250,250]

# If using unbalanced data
NUM_EXAMPLES_PER_BALANCED_CLASS = \
       int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / NUM_CLASSES)
OVERSAMPLE_FACTOR_PER_CLASS = \
       [int(round(NUM_EXAMPLES_PER_BALANCED_CLASS/j))
        for j in NUM_TRAIN_IMG_PER_CLASS]


def read_surferdetection(filename_queue):
  """Reads and parses examples from SURFERDETECTION data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (80)
      width: number of columns in the result (80)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..4.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class SURFERDETECTIONRecord(object):
    pass
  result = SURFERDETECTIONRecord()

  # Dimensions of the images in the SURFERDETECTION dataset.
  # See README for a description of the input format.
  label_bytes = 1
  result.height = 80
  result.width = 80
  result.depth = 3
  image_bytes = result.height * result.width * result.depth # =19200
  
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes # =19201

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the SURFERDETECTION format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image. 'value' represents
  # the image which we reshape from [depth * height * width] to 
  # [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                          [result.depth, result.height, result.width])
  
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(images, labels, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels that shuffles the examples,
  and then read 'batch_size' images + labels from the example queue.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  # A shuffling queue into which tensors from tensors are enqueued.
  num_preprocess_threads = 16

  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [images, labels],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        enqueue_many = False)
  else:
    images, label_batch = tf.train.batch(
        [images, labels],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        enqueue_many = False)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  # the returned images are dequeued from the batch queue above
  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for SURFERDETECTION training using the Reader
  ops.

  Args:
    data_dir: Path to the SURFERDETECTION data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
            size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, FLAGS.num_bins+1)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create queue that produces the files to read, ready for an input pipeline.
  filename_queue = tf.train.string_input_producer(filenames) 

  # Use the Reader Ops to read examples from files in the filename queue.
  read_input = read_surferdetection(filename_queue)
  
  # Cast types:
  reshaped_image = tf.cast(read_input.uint8image, tf.uint8) #[80,80,3]
  label = tf.cast(read_input.label, tf.int32) #1d tensor of dtype int32
  
  # If OVERSAMPLE the training set due to unbalanced labels:
  if FLAGS.oversample == True:
    reshaped_image = tf.expand_dims(reshaped_image, 0) # [1,80,80,3]
    [os_images, os_labels] = [reshaped_image,label]

    # Boolean for the label:
    pred0 = tf.reshape(tf.equal(label, tf.convert_to_tensor([0])), [])
    pred1 = tf.reshape(tf.equal(label, tf.convert_to_tensor([1])), [])

    # Vertically stack tensors in a batch:
    def f0(): return tf.concat(
        0,
        [reshaped_image]*OVERSAMPLE_FACTOR_PER_CLASS[0]), \
        tf.concat(0, [label]*OVERSAMPLE_FACTOR_PER_CLASS[0])
    def f1(): return tf.concat(
        0,
        [reshaped_image]*OVERSAMPLE_FACTOR_PER_CLASS[1]), \
        tf.concat(0, [label]*OVERSAMPLE_FACTOR_PER_CLASS[1])
  
    [os_images, os_labels] = tf.cond(pred0, f0, lambda: [os_images, os_labels])
    [os_images, os_labels] = tf.cond(pred1, f1, lambda: [os_images, os_labels])

  # If DISTORT the training set for data augmentation:
  if (FLAGS.distortion == True) and (FLAGS.oversample == False):
    def distort(x): return tf.reshape(
                               tf.py_func(surferdetection_augmentation.augment,
                                          [x], [tf.uint8], stateful=True),
                           [80,80,3])

    os_images = distort(reshaped_image)
    os_labels = label

  if (FLAGS.distortion == True) and (FLAGS.oversample == True):
    def distort(x): return tf.reshape(
                               tf.py_func(surferdetection_augmentation.augment,
                                          [x], [tf.uint8], stateful=True),
                           [80,80,3])

    os_images = tf.map_fn(distort, os_images)

  # Divide by the range of the pixels to normalize [0,1], cast to float32.
  os_images = tf.div(tf.to_float(os_images),
                     tf.constant(255, dtype=tf.float32))
  
  # Check data for issues
  tf.check_numerics(os_images, message="bad data!", name=None)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.50
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue) # =5000
  print ('Filling queue with %d SURFERDETECTION images before starting to train.'
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(os_images, 
                                         os_labels,
                                         min_queue_examples, 
                                         batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for SURFERDETECTION evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the SURFERDETECTION data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, FLAGS.num_bins)] 
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'eval_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_surferdetection(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Divide by the range of the pixels to normalize [0,1], cast to float32.
  float_image = tf.div(reshaped_image,tf.constant(255,dtype=tf.float32))

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 1.0
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, 
                                         read_input.label,
                                         min_queue_examples, 
                                         batch_size,
                                         shuffle=False)
