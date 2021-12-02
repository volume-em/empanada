"""

Preprocessing functions that are applied to images that
come off of a microscope in real time.

"""

import cv2
import math
import numpy as np

__all__ = [
    'invert_intensity', 'bin2d', 'resize2d', 'pad2d',
    'create_image_pyramid'
]

def invert_intensity(image):
    return np.invert(image)

def bin2d(image, kernel_size):
    assert kernel_size < min(image.shape), "Image too small for kernel!"
    
    h, w = image.shape
    
    # padding on all sides is half of kernel size, with round up
    ps = int(math.ceil(kernel_size / 2))
    image = np.pad(image, (ps, ps), mode='reflect')

    # windows never overlap
    stride = kernel_size 
    h_stride, w_stride = image.strides
    
    # set stride shape and strides
    stride_shape = (
        (h - kernel_size) // stride + 1,
        (w - kernel_size) // stride + 1,
        kernel_size, kernel_size
    )
    strides = (stride * h_stride, stride * w_stride, h_stride, w_stride)

    image_strided = np.lib.stride_tricks.as_strided(image, stride_shape, strides)
    image_binned = image_strided.mean(axis=(2, 3))
    
    # convert the the binned image to it's original type
    return np.round(image_binned).astype(image.dtype)

def resize2d(
    image, 
    scale_factor=None, 
    output_shape=None,
    is_mask=False
):
    assert any([scale_factor is not None, output_shape is not None])
    assert not all([scale_factor is not None, output_shape is not None])
    if scale_factor == 1:
        return image
    
    if scale_factor is not None:
        output_shape = tuple([size // scale_factor for size in image.shape])
        
    # cv2 swaps width and height
    out_w, out_h = output_shape[::-1]
    interpolator = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    
    return cv2.resize(image, (out_w, out_h), interpolation=interpolator)

def pad2d(image, percent=0.1):
    """Zero pads image by some percentage of the image size."""
    padding = tuple([round(percent * size) for size in image.shape])
    return np.pad(image, padding, mode='constant')

def create_image_pyramid(image, scale_factors=[1, 2, 4]):
    return [resize2d(image, scale_factor=sf) for sf in scale_factors]