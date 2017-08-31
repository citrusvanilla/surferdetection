# Surfer Detection using a Convolutional Neural Network (CNN) in Google's Tensorflow

>*NOTE: Source data is NOT included in this repository.  Please contact author for access.*

![Surfer Detection Example](http://i.imgur.com/QaWJIU3.jpg?1)

This project contains a Convolutional Neural Network framework for detecting surfers, 
built in Python 2 using Google's Tensorflow.  You can follow Google's awesome
[Tensorflow](http://tensorflow.org) project, by referencing the Deep CNN tutorial found [here](https://www.tensorflow.org/tutorials/deep_cnn/).

The model contained here is trained on scenes from one of the 150 cameras from [Surfline](http://surfline.com)-
specifically, the [Manasquan Inlet](http://www.surfline.com/surf-report/manasquan-inlet-mid-atlantic_4278/) camera
from Manasquan, New Jersey.  You can view the results of a fully-trained surfer detector on a full day's worth of scenes at [Vimeo](https://vimeo.com/citrusvanilla/surferdetection).
Password is 'rockrockrockrockawaybeach'.

### Software and Library Requirements
* Python 2.7.11
* Jupyter Notebook 4.2.2
* Tensorflow 0.12
* Numpy 1.11.2
* PIL Image 1.1.7
* scikit-image 0.12.3
* matplotlib 1.5.2

## Goals
The goal is to detect surfers towards the quantification of crowd sizes.

## Key Processes
1. Train a Surfer Detector from scratch using 80x80 image patches from a Surfline camera
2. Restore a fully-trained Surfer Detector
3. Visualize predictions on unseen Scenes

## Highlights of the CNN
* Lean architecture of 2 convolutional layers and 2 fully connected layers, requiring only 13.6 million multiply/add operations per image patch
* Trainable in under an hour on a CPU
* 5% error rate on evaluation set
* Leaky RELUs, Local Response Normalization, decaying learning rate

## Model Architecture
The model in this demo is a multi-layer architecture consisting of alternating convolutions and non-linearities.
These layers are followed by fully connected layers leading into a softmax classifier. 
The model follows closely the architecture described by [Alex Krizhevsky](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

This model achieves a peak performance of about 95% accuracy in about an hour on a CPU. 

## Code Organization

File | Purpose
------------ | -------------
surferdetection_input.py |	Reads the SURFERCOUNTING binary file format.
surferdetection.py |	Builds the SURFERCOUNTING model.
surferdetection_train.py	| Trains a SURFERCOUNTING model on a CPU.
surferdetection_eval.py |	Evaluates the predictive performance of a SURFERCOUNTING model.
surferdetection_augmentation.py	| Data augmentation routine for SURFERCOUNTING images.
surferdetection_predict.py |	Makes predictions on unseen images.
surferdetection_predictscene.ipynb |	iPython Notebook for building full scene visualizations of predictions.

## SURFERDETECTION Model

The SURFERDETECTION network is largely contained in surferdetection.py. 

* Model inputs: inputs() and distorted_inputs() add operations that read and preprocess SURFERDETECTION images for evaluation and training, respectively.
* Model prediction: inference() adds operations that perform inference, i.e. classification, on supplied images.
* Model training: loss(), accuracy(), and train() add operations that compute the loss, training accuracy, gradients, variable updates and visualization summaries.


**Model Inputs**

The input part of the model is built by the functions inputs() and distorted_inputs() which read images from the SURFERDETECTION binary data files. 
These files contain fixed byte length records, so we use tf.FixedLengthRecordReader as the input reader.


**Data** 

You will need to obtain the tarball from the author before beginning training. 

**PLEASE PLACE THE TAR.GZ FILE INSIDE THE 'SURFERCOUNTING_DATA' DIRECTORY!**

Following TF's lead, we augment images inside 16 separate threads which continuously fill a TensorFlow queue.
However, augmentation uses external Numpy functions in surferdetector_augmentation.py. 

* Images are normalized to [0,1].
* Images are randomly flipped from left to right.
* Images are randomly rotated through [-12,12} degrees.
* X and Y axis are randomly scaled independently through [0.80, 1.0] percent.
* Rescaled images are randomly translated within a patch, while still preserving all original pixels.


**Model Prediction**

The prediction part of the model is constructed by the inference() function which adds operations to compute the logits of the predictions. 
This part of the model is organized as follows:

Layer Name | Description
------------ | -------------
conv1	| convolution and "leaky" rectified linear activation.
pool1	| overlapping max pooling.
norm1	| local response normalization.
conv2	| convolution and "leaky" rectified linear activation.
norm2	| local response normalization.
pool2	| overlapping max pooling.
local3 | fully connected layer with rectified linear activation.
local4	| fully connected layer with rectified linear activation.
softmax_linear	| linear transformation to produce logits.


**Model Training**

We train the Sufter Detection model "online" (that is, one image at a time as opposed to "batch
processing").  Exploding gradients are checked with leaky RELUs and small positive bias initialization.

For regularization, we apply weight decay losses to all learned variables as well as local response normalization
after the convolutional layers. 

The Surfer Detection model uses binary softmax regression ("surfer present" or "surfer absent"). 

The objective function for the model is the sum of the cross entropy loss and 
all these weight decay terms,  as returned by the loss() function.

Training can be "babysat" using Google's [Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/),
by entering the the following command at the commandline after training has commenced:

    tensorboard --logdir=/surferdetection/surferdetection_train


## Launching and Training the Model

Launch training from the commandline with the script surferdetection_train.py.

    python surferdetection_train.py

You should see output like this:

    Filling queue with 5000 SURFERCOUNTING images before starting to train. This will take a few minutes.
    2016-11-04 11:45:45.927302: step 0, loss = 0.57 (47.3 examples/sec; 0.024 sec/batch)
    2016-11-04 11:45:49.133065: step 50, loss = 0.86 (52.8 examples/sec; 0.025 sec/batch)
    
The script reports the loss and accuracy on the image every 50 steps, as well as the speed at which the last image was processed.

_surferdetection_train.py_ saves all model parameters in checkpoint files every 1000 steps but it does not evaluate the model. 
The checkpoint file will be used by surferdetection_eval.py to measure the predictive performance.

Launch periodic evaluation (set to evaluate the full validation set every 2 minutes) from the commandline after training has commenced:

    python surferdetection_eval.py

This evaluation simply gives accuracy on the evaluation set as a percentage.  You should see an output such as this:

    2016-11-06 08:30:44.391206: precision @ 1 = 0.860

The training script calculates the moving average version of all learned variables. The evaluation script substitutes all learned model 
parameters with the moving average version. This substitution boosts model performance at evaluation time.

**Visualizing Test Scenes**

The Jupyter Notebook file 'surferdetection_predictscene.ipynb' has been provided to help you visualize prediction on unseen scenes.
This notebook utilizes the surferdetection_predict.py file that makes use of a fully-trained surfer detector model in the surferdetection_restore directory.

Launch the notebook from the commandline with the following:

    ipython notebook surferdetection_predictscene.ipynb

And that's it!  Please contact the author for gaining access to source data, troubleshooting or general comments.
