import os
import cv2
import torch
import numpy as np
import albumentations as A
from skimage import io
from skimage import measure

from empanada.data._base import _BaseDataset
from empanada.data.utils import heatmap_and_offsets, copy_paste_class

__all__ = [
    'SingleClassInstanceDataset'
]

@copy_paste_class
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

        # used for copy-paste
        self._split_transforms()

    def load_example(self, idx):
        # get image and mask
        f = self.impaths[idx]
        image = cv2.imread(f, 0)
        mask = cv2.imread(self.mskpaths[idx], -1)

        data = {}
        # add channel dimension if needed
        if image.ndim == 2:
            image = image[..., None]

        # convert all of the target segmentations to masks
        # bboxes are expected to be (y1, x1, y2, x2, category_id, mask_id)
        masks = []
        bboxes = []
        for ix, rp in enumerate(measure.regionprops(mask)):
            rp_mask = np.zeros_like(mask)
            rp_mask[tuple(rp.coords.T)] = 1
            masks.append(rp_mask)
            rpb = rp.bbox
            # category_id is default 1 for mitos
            coco_bbox = (rpb[1], rpb[0], rpb[3], rpb[2], 1, ix)
            bboxes.append(coco_bbox)

        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }

        return self.transforms(**output)

    def __getitem__(self, idx):
        # transformed and paste example
        f = self.impaths[idx]
        data = self.load_pasted_example(idx)

        # create semantic seg, heatmaps, and centers
        # for pasted examples
        image = data['image']
        masks = data['masks']

        # albumentations doesn't convert masks
        # to tensors, do that here
        c, h, w = image.size()
        if isinstance(image, torch.Tensor):
            masks = [torch.from_numpy(mask) for mask in masks]
            ins_mask = torch.zeros((h, w), dtype=torch.long)
        else:
            assert isinstance(image, np.ndarray)
            ins_mask = np.zeros((h, w), dtype='int')

        output = {'image': image}
        for i,m in enumerate(masks, 1):
            ins_mask += i * m

        heatmap, offsets = heatmap_and_offsets(ins_mask, self.heatmap_sigma)
        output['ctr_hmp'] = heatmap
        output['offsets'] = offsets

        # THIS DOESN'T GIVE PASTED IMAGE FNAME
        fdirs = f.split('/')
        subdir = fdirs[fdirs.index('images') - 1] # subdir is 1 before images dir
        output['fname'] = f

        # confidences are 1-5, subtract 1 to have 0-4 (for cross entropy loss)
        if self.has_confidence:
            output['conf'] = self.confidences_dict[subdir][os.path.basename(f)] - 1

        # the last step is to binarize the mask for semantic segmentation
        output['sem'] = (ins_mask > 0).float()

        return output
