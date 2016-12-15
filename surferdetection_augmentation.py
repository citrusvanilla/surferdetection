"""Routine for augmenting a SURFERDETECTION image.

Summary of augmentation functions:

 # RANDOM COLOR TEMPERATURE
 img, color_temp_val = color_temp(img)

 # RANDOM CONTRAST AND BRIGHTNESS
 img, exposure_range = contrast_brightness(img)
        
 # RANDOM HORIZONTAL FLIP
 img, flipped = horizontal_flip(img)

 # RANDOM ROTATION OF IMAGE
 img, rotation_angle = rotate(img,max_rotation_angle)

 # RESIZE IMAGE BACK TO OUTPUT SIZE
 img = resize(img,output_size)
        
 # RANDOM RESCALE/ZOOM IMAGE
 img, scale_xy = rescale(img,min_xy_scale_factor)

 # RANDOM TRANSLATION OF IMAGE
 img, translation_xy= translate(img,output_size,scale_xy)

 # COMBINATION OF ALL DISTORTIONS
 img, distortion_stats = augment(input, 
                                max_rotation_angle=0, 
                                min_xy_scale_factor=1.0, 
                                output_size = (80,80), 
                                rand_hor_flip = False, 
                                rand_color_temp = False, 
                                rand_contr_bright = False)

Usage:
augment() combines all individual distortions and operates on one image at a time.
Please note that the individual distortion functions are not commutative.  
Most but not all functions rely on SkImage library and therefore 
the input and output image dtypes are different for certain functions.  
Please see function documentation for more information.
"""

import numpy as np
import random
import math
from skimage import transform #docs at http://scikit-image.org/docs/dev/
from skimage import exposure #docs at http://scikit-image.org/docs/dev/
from PIL import Image
import tensorflow as tf

kelvin_table = {
    0000: (255,255,255),
    3000: (255,180,107),
    6000: (255,243,239),
    9000: (214,225,255),
    12000: (191,211,255)}

# You can load a local image for debugging purposes here.
#file = "/{your_sample_image}.jpeg"


def color_temp(image):
    """Adjusts Color "warmth" of image randomly using Kelvin Scale.
       Please note this function is not commutative.  See augment() function.

      Args:
        image: a 3D PIL Image.

      Returns:
        image: an augmented 3D PIL image, same dtype as input.
        temp: scalar value of randomly chosen kelvin value.
    """
    # adjust the color temp of the image
    temp = random.choice(list(kelvin_table.keys()))
    r, g, b = kelvin_table[temp]
    kelvin_matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    
    return image.convert('RGB', kelvin_matrix), temp


def contrast_brightness(img_as_array):
    """Adjusts exposure range of 8-bit RGB image randomly. Minimum random 
       range is 50.
       Please note this function is not commutative.  See augment() function.

      Args:
        img_as_array: 3D image as a numpy array.

      Returns:
        image: an augmented 3D image as a numpy array, same dtype as input.
        intensity: lower range value and upper range value as tuple.
    """
    # perform random contrast adjustment
    intensity_min = random.randint(0, 155)
    intensity_max = random.randint(intensity_min+100,255)

    return exposure.rescale_intensity(img_as_array,out_range=(intensity_min,intensity_max)), (intensity_min, intensity_max)


def horizontal_flip(img_as_array):
    """Randomly flips image horizontally.
       Please note this function is not commutative.  See augment() function.

      Args:
        img_as_array: image 3D numpy array.

      Returns:
        image: augmented 3D numpy array, same dtype as input.
        flipped: boolean indicating if image was flipped.
    """

    # random bernoulli for a horiztonal flip
    if random.random() < 0.5:
        flipped = True
        img_as_array = np.fliplr(img_as_array)
    else:
        flipped = False

    return img_as_array, flipped


def rotate(img_as_array, max_rotation_angle):
    """Randomly rotates image and resizes image to the extent of the rotated image
       so that no pixels are lost.
       Please note this function is not commutative.  See augment() function.

      Args:
        img_as_array: image as 3D numpy array.
        max_rotation_angle: int or float indicating the maximum absolute angle of rotation.

      Returns:
        image: augmented and resized image as 3D numpy uint8 array.
        rotation_angle: randomly determined rotation angle as int.
    """

    rotation_angle = random.randint(-max_rotation_angle, max_rotation_angle)
    return transform.rotate(img_as_array, rotation_angle, resize=True, center=None, 
           order=1, mode='constant', cval=140, clip=True, preserve_range=True).astype(np.uint8), rotation_angle


def resize(img_as_array,output_size):
    """Resizes input image to desired output image size.
       Please note this function is not commutative.  See augment() function.

      Args:
        img_as_array: image as 3D numpy array.
        output_size: tuple of height, width values indicating desired output image size.

      Returns:
        image: resized image as 3D numpy uint8 array.
    """

    return transform.resize(img_as_array, output_size, order=1, mode='constant', 
                            cval=140, clip=True, preserve_range=True).astype(np.uint8)


def rescale(img_as_array,min_xy_scale_factor):
    """Randomly downscales the x and y ranges of an image while zero-padding image to
       retain the input image size shape.  Newly downscaled image occupies the
       upper-left most region of the output image.
       Please note this function is not commutative.  See augment() function.
       
      Args:
        img_as_array: image as 3D numpy array.
        min_xy_scale_factor: lower bound of downscaling factor for x and y values.
        
      Returns:
        image: rescaled imaged as 3D numpy uint8 array.
        scale_xy: randomly determined downscale factors as tuple of x and y factors.
    """

    # perform randomized affine_transformation
    scale_x = random.uniform(min_xy_scale_factor,1.0)
    scale_y = random.uniform(min_xy_scale_factor,1.0)
    scale_xy = (scale_x, scale_y)
    scale_xy_inverted = (1/scale_x, 1/scale_y)
    
    affine_transform=transform.AffineTransform(scale=scale_xy_inverted)

    return transform.warp(img_as_array, affine_transform, map_args={}, 
                          output_shape=None, order=1, mode='constant', 
                             cval=140.0, clip=True, preserve_range=True).astype(np.uint8), scale_xy


def translate(img_as_array,output_size,scale_xy):
    """Translates an image and pads with zeros.  The provided x and y rescaling factors
       from a previous rescaling operation ensures that a randomly chosen translation 
       does not cause the output image to lose any pixels.
       Please note this function is not commutative.  See augment() function.

      Args:
        img_as_array: image as 3D numpy array.
        output_size: tuple of height, width values indicating desired output image size.

      Returns:
        image: image the size of output_size as 3D numpy uint8 array.
        translation_xy: randomly determined x and y translations of the image as tuple.
    """

    scaled_width = (int(output_size[0]*scale_xy[0]))
    scaled_height = (int(output_size[1]*scale_xy[1]))

    translation_x=random.randint(0,(output_size[0]-scaled_width))
    translation_y=random.randint(0,(output_size[1]-scaled_height))
    translation_xy = (-translation_x,-translation_y)

    affine_transform=transform.AffineTransform(translation=(translation_xy))
    
    return transform.warp(img_as_array, affine_transform, map_args={}, 
                          output_shape=None, order=1, mode='constant', 
                             cval=140.0, clip=True, preserve_range=True).astype(np.uint8), translation_xy


def augment(input, random_a = 0.75, max_rotation_angle=12, min_xy_scale_factor=0.75, output_size = (80,80),
            rand_hor_flip = True, rand_color_temp = False, rand_contr_bright = False):
    """Combines all augmentation operations.  Subfunctions are not commutative due
       to the nature of the SURFERDETECTION dataset.  See README for more information.

      Args:
        input: image as NUMPY array from main or sample image file for debugging.
        random: random percentage to perform augmentation
        max_rotation_angle: maximum accepted angle of image rotation, as int or float.
        min_xy_scale_factor: mimimum accepted rescaling factor for image x and y, as float.
        output_size: output size of image as tuple.  recommended same is input.
        rand_hor_flip: boolean for random horizontal flip.
        rand_color_temp: boolean for random color temperature adjustment.
        rand_contr_bright: boolean for random contrast adjustment.

      Returns:
        image: augmented image as 3D numpy uint8 array.
        distortion_stats: statistics of the augmentation routine as a dict (debugging)
    """

    # Aug
    img = input
    if random.random() < random_a:


        # CONVERT INPUT IMAGE TO PIL IMAGE
        img = Image.fromarray(input)

        # RANDOM COLOR TEMPERATURE
        if rand_color_temp == True:
            img, color_temp_val = color_temp(img)
        else:
            color_temp_val = None

        # CONVERT TO NUMPY ARRAY
        img = np.asarray(img)
        img_size = (img.shape[0],img.shape[1])
        
        # RANDOM CONTRAST AND BRIGHTNESS
        if rand_contr_bright == True:
            img, exposure_range = contrast_brightness(img)
        else:
            exposure_range = None
            
        # RANDOM HORIZONTAL FLIP
        if rand_hor_flip == True:
            img, flipped = horizontal_flip(img)
        else:
            flipped = None

        # RANDOM ROTATION OF IMAGE
        if max_rotation_angle != 0:
            img, rotation_angle = rotate(img,max_rotation_angle)
        else:
            rotation_angle = 0

        # RESIZE IMAGE BACK TO OUTPUT SIZE
        if rotation_angle != 0:
            img = resize(img,output_size)
            
        # RANDOM RESCALE/ZOOM IMAGE
        if rotation_angle != 0:
            img, scale_xy = rescale(img,min_xy_scale_factor)
        else:
            scale_xy = (0,0)

        # RANDOM TRANSLATION OF IMAGE
        if rotation_angle != 0:
            img, translation_xy= translate(img,output_size,scale_xy)
        else:
            translation_xy = (0,0)
            
        # FINAL DISTORTION STATS
        distortion_stats = {
            "color_temp_val": color_temp_val,
            "exposure_range": exposure_range,
            "flipped?": flipped,
            "rotation_angle": rotation_angle,
            "scale_xy": scale_xy,
            "translation_xy": translation_xy}
    
    return img #, distortion_stats


