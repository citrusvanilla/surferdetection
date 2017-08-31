##
## Surfer Detection
## surferdetection_augmentation.py
##
## Copyright 2017 Justin Fung. All rights reserved.
##
## ========================================================

"""Routine for making predictions of a directory of tiles.

Summary of available functions:

  # predict()
  predictions = predict(test_directory, threshold, prob)

Usage:
  predict() accepts a directory path to a set of jpeg images that have already
  been "tiled", that is broken up into 80x80 patches. It returns a 1D numpy
  array of positive identifications in the order that the tiles are passed to
  the function.  predict() can also return positive identifications if and only
  if the softmax for a positive identification exceeds a passed threshold.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import re

import numpy as np
import tensorflow as tf

import surferdetection

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'surferdetection_restore',
                           """Directory where to read model checkpoints.""")


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def predict(test_directory, threshold, prob = False):
  """Predict label for images.

  Args:
    test_directory: location of jpeg images to predict.
    threshold: returns positive instances only for softmax exceeding this level.
    prob: positive instances must exceed 'threshold' value if set to True.
   """ 
  with tf.Graph().as_default() as g:

    # Get a list of tiles that will be predicted:
    tiles = [image for image in natural_sort(os.listdir(test_directory))
             if not image.startswith('.')]
    for tile in range(len(tiles)): tiles[tile] =
                      os.path.join(test_directory, tiles[tile])
    num_tiles = len(tiles)

    # Make a queue from the tile names.
    tiles_queue = tf.train.string_input_producer(tiles, shuffle=False)

    # Make a reader to reader the tiles into TF.
    reader = tf.WholeFileReader()

  	# Read the tiles into TF
    key, value = reader.read(tiles_queue)

  	# Decode the reader value and assign it to 'tile'.
    tile = tf.image.decode_jpeg(value, channels=3)
    tile = tf.reshape(tile,[80, 80, 3])
    tile = tf.expand_dims(tile, 0)

	  # Preprocess the tile as in training:
    float_image = tf.div(tf.to_float(tile),tf.constant(255, dtype=tf.float32))

    # Compute the logits prediction from the inference model.
    logit = surferdetection.inference(float_image)

    # Restore the moving average version of the learned variables for test.
    variable_averages = tf.train.ExponentialMovingAverage(
                            surferdetection.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return

      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      num_iter = num_tiles
      step = 0
      
      # Initialize numpy array to hold predictions.
      prediction_map = np.zeros(num_tiles)

      while step < num_iter and not coord.should_stop():
      	
        # If confidence in prediction exceeds threshold,
        # return 1 for positive idenitication.
        if prob  == True:
          probabilities = sess.run(tf.nn.softmax(logit))
          
          if np.argmax(probabilities, axis=1) == 1: 
              prediction_map[step] = 1 \
              if np.amax(probabilities, axis=1) >= threshold else 0
      	  else: prediction_map[step] = 0
        
      	else:
          logit_pair = sess.run(logit)
          prediction_map[step] = np.argmax(logit_pair, axis=1)
        
        step += 1

      coord.request_stop()
      coord.join(threads)

  return prediction_map.astype(int)
