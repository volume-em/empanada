import cv2
import numpy as np
from preprocess import resize2d

__all__ = [
    'xcorr_transform'
]

def shift_to_affine(shifts):
    T = np.eye(len(shifts)+1)
    T[0, 2] = shifts[0] # x translation
    T[1, 2] = shifts[1] # y translation

    return T

def apply_affine(image, transform):
    h, w = image.shape
    image = cv2.warpAffine(image, transform[:2], (w, h))

    return image

def phase_correlation(target_image, moving_image, downsampling=2):
    # downsample images for faster correlation
    target_image = resize2d(target_image, scale_factor=downsampling)
    moving_image = resize2d(moving_image, scale_factor=downsampling)
    
    # convert to float 32
    target_image = target_image.astype(np.float32)
    moving_image = moving_image.astype(np.float32)
    
    h, w = target_image.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, phase = cv2.phaseCorrelate(moving_image, target_image, window)
    
    return shift

def xcorr_transform(target_image, moving_image, downsampling=2):
    # compute xy shifts to align images
    shift = phase_correlation(
        target_image, moving_image, downsampling
    )
    
    # return shifts in an affine matrix
    return shift_to_affine([s * downsampling for s in shift])