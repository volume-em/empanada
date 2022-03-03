import cv2
import math
import albumentations as A

__all__ = [
    'resize_by_factor',
    'FactorPad'
]

def resize_by_factor(image, scale_factor=1):
    # do nothing
    if scale_factor == 1:
        return image

    # cv2 expects (w, h) for image size
    h, w = image.shape
    dh = math.ceil(h / scale_factor)
    dw = math.ceil(w / scale_factor)

    image = cv2.resize(image, (dw, dh), cv2.INTER_LINEAR)

    return image

class FactorPad(A.Lambda):
    def __init__(self, factor=128):

        def pad_func(x, **kwargs):
            return factor_pad(x, factor=factor)

        super().__init__(image=pad_func, mask=pad_func)
