import os
import cv2
import torch
import numpy as np
import albumentations as A
from skimage import io
from skimage import measure

from empanada.data._base import _BaseDataset
from empanada.data.utils import heatmap_and_offsets

__all__ = [
    'SingleClassInstanceDataset'
]

class SingleClassInstanceDataset(_BaseDataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        heatmap_sigma=6,
        weight_gamma=0.3,
        has_confidence=False
    ):
        super(SingleClassInstanceDataset, self).__init__(
            data_dir, transforms, heatmap_sigma, weight_gamma, has_confidence
        )

    def __getitem__(self, idx):
        # transformed and paste example
        f = self.impaths[idx]
        image = cv2.imread(f, 0)
        mask = cv2.imread(self.mskpaths[idx], -1)
        
        # add channel dimension if needed
        if image.ndim == 2:
            image = image[..., None]
        
        if self.transforms is not None:
            output = self.transforms(image=image, mask=mask)
        else:
            output = {'image': image, 'mask': mask}
        
        mask = output['mask']
        heatmap, offsets = heatmap_and_offsets(mask, self.heatmap_sigma)
        output['ctr_hmp'] = heatmap
        output['offsets'] = offsets

        fdirs = f.split('/')
        # subdir is 1 before images dir
        subdir = fdirs[fdirs.index('images') - 1]
        output['fname'] = f

        # confidences are 1-5, subtract 1 to have 0-4 (for cross entropy loss)
        if self.has_confidence:
            output['conf'] = self.confidences_dict[subdir][os.path.basename(f)] - 1

        # the last step is to binarize the mask for semantic segmentation
        if isinstance(mask, torch.Tensor):
            output['sem'] = (mask > 0).float()
        elif isinstance(mask, np.ndarray):
            output['sem'] = (mask > 0).astype(np.float32)
        else:
            raise Exception(f'Invalid mask type {type(mask)}. Expect tensor or ndarray.')

        return output