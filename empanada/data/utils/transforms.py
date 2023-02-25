import cv2
import math
import numpy as np

__all__ = [
    'resize_by_factor'
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

def factor_pad(image, factor=128):
    h, w = image.shape[:2]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if image.ndim == 3:
        padding = ((0, pad_bottom), (0, pad_right), (0, 0))
    elif image.ndim == 2:
        padding = ((0, pad_bottom), (0, pad_right))
    else:
        raise Exception

    padded_image = np.pad(image, padding)
    return padded_image

try:
    # only necessary for model training,
    # inference-only empanada doesn't need it
    import albumentations as A
            
    class FactorPad(A.Lambda):
        def __init__(self, factor=128):
            super().__init__(image=self.pad_func, mask=self.pad_func)
            self.factor = factor

        def pad_func(self, x, **kwargs):
            return factor_pad(x, factor=self.factor)
    
    __all__.append('FactorPad')
    
except ImportError:
    pass

try:
    import random
    from volumentations.augmentations import functional as F
    from volumentations.core.transforms_interface import ImageOnlyTransform, to_tuple
    
    class RandomBrightnessContrast3d(ImageOnlyTransform):
        def __init__(
            self,
            brightness_limit=0.2,
            contrast_limit=0.2,
            brightness_by_max=True,
            always_apply=False,
            p=0.5,
        ):
            super(RandomBrightnessContrast3d, self).__init__(always_apply, p)
            self.brightness_limit = to_tuple(brightness_limit)
            self.contrast_limit = to_tuple(contrast_limit)
            self.brightness_by_max = brightness_by_max

        def apply(self, img, alpha=1.0, beta=0.0, **params):
            return F.brightness_contrast_adjust(img.astype(np.uint8), alpha, beta, self.brightness_by_max).astype(np.float32)

        def get_params(self, **data):
            return {
                "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
                "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
            }

        def get_transform_init_args_names(self):
            return ("brightness_limit", "contrast_limit", "brightness_by_max")
        
    __all__.append('RandomBrightnessContrast3d')
            
except ImportError:
    pass