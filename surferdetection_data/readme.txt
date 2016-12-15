Training and Evaluation data are absent from this repository.
Please contact citrusvanilla@gmail.com for requests.

-----------

Binary Version of Surferdetection images:

The binary version writes the files data_batch_1.bin, data_batch_2.bin, ..., data_batch_5.bin, as well as eval_batch.bin. Each of these files is formatted as follows:

<1 x label><19200 x pixel>
...
<1 x label><19200 x pixel>

In other words, the first byte is the label of the first image, which is a number in the range 0-5. The next 19200 bytes are the values of the pixels of the image. The first 6400 bytes are the red channel values, the next 6400 the green, and the final 6400 the blue. The values are stored in row-major order, so the first 80 bytes are the red channel values of the first row of the image. 

Each file contains 2000 such 19201-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be about 38402000 bytes long (~38.4 MB).

Please note that this method was inspired by the CIFAR-10 dataset.  For more information, please see http://www.cs.toronto.edu/~kriz/cifar.html.

To create the tarball from the Binary files from the command line, use: