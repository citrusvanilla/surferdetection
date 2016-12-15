# NOTE THESE TEMPLATES COME FROM TENSORFLOW TUTORIALS,
# WHICH HAVE THE FOLLOWING LICENSE RESTRICTIONS
# 
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""A binary to train SURFERDETECTION using a single GPU.

Accuracy:
surferdetection_train.py achieves ~94% accuracy after 130K steps (13 epochs of
data) as judged by surferdetection_eval.py.

Speed: With online training.

System                       | Step Time (sec/image)  |     Accuracy
------------------------------------------------------------------
1 CPU 2.6 GHz Intel Core i5  | 0.03                   | ~94% at 130K steps (1 hour)

Usage:
Please see the README for how to download the surferdetection
data set, compile the program and train the model.

Training automatically begins a new training.
To restore a trained model, set FLAGS.restore to True.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import surferdetection

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'surferdetection_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of images to run.""") # one batch is 100 images
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('restore', False,
                           """If True, restore a pretrained model""")
tf.app.flags.DEFINE_string('restore_file', 'surferdetection_restore/restore_model.ckpt',
                           """Directory where to restore parameters""")
tf.app.flags.DEFINE_integer('starting_step', 1,
                           """Initial step of training""")

def train():
  """Train SURFERDETECTION for max_steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(FLAGS.starting_step, trainable=False)

    # Get images and labels for SURFERDETECTION.
    images, labels = surferdetection.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = surferdetection.inference(images)

    # Calculate loss.
    loss = surferdetection.loss(logits, labels)

    # Calculate Accuracy.
    accuracy = surferdetection.accuracy(logits, labels)

    # Calculate Batch Label Distribution.
    distribution_mean = surferdetection.batch_distribution(labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = surferdetection.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Restore a session.
    if FLAGS.restore: saver.restore(sess, FLAGS.restore_file)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.starting_step,FLAGS.max_steps):
      start_time = time.time()
      _, loss_value, accuracy_value, batch_mean = sess.run([train_op, loss, accuracy, distribution_mean])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Only prints for every 50th step... more-frequent printing suggested for "babysitting" training.
      if step % 50 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch), training accuracy = %.2f, batch mean = %.2f')
        print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch, accuracy_value, batch_mean))

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  surferdetection.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
