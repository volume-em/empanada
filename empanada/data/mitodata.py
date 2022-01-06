import os
import cv2
import json
import torch
import numpy as np
import albumentations as A
from glob import glob
from skimage import io
from skimage import measure
from torch.utils.data import Dataset
from copy import deepcopy
from empanada.data.copy_paste import copy_paste_class

# ignore bad warning from albumentations.ReplayCompose
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'MitoData', 'MitoDataQueue'
]
    
def heatmap_and_offsets(sl2d, heatmap_sigma=6):
    # make sure, the input is numpy
    convert = False
    if type(sl2d) == torch.Tensor:
        sl2d = sl2d.numpy()
        convert = True
        
    h, w = sl2d.shape
    centers = np.zeros((2, h, w), dtype=np.float32)
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # loop over the instance labels and store
    # relevant centers for each
    rp = measure.regionprops(sl2d)
    for r in rp:
        sub_label = r.label
        y, x = r.centroid
        heatmap[int(y), int(x)] = 1
        centers[0, sl2d == sub_label] = y
        centers[1, sl2d == sub_label] = x
            
    # apply a gaussian filter to spread the centers
    heatmap = cv2.GaussianBlur(heatmap, ksize=(0, 0), 
                               sigmaX=heatmap_sigma, sigmaY=heatmap_sigma, 
                               borderType=cv2.BORDER_CONSTANT)
    
    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax
        
    # convert from centers to offsets
    yindices = np.arange(0, h, dtype=np.float32)
    xindices = np.arange(0, w, dtype=np.float32)

    # add the y indices to the first channel
    # in the output and x indices to the second channel
    offsets = np.zeros_like(centers)
    offsets[0] = centers[0] - yindices[:, None]
    offsets[1] = centers[1] - xindices[None, :]
    offsets[:, sl2d == 0] = 0
    
    # add empty dimension to heatmap
    heatmap = heatmap[None] # (1, H, W)
    
    if convert:
        heatmap = torch.from_numpy(heatmap)
        offsets = torch.from_numpy(offsets)
    
    return heatmap, offsets

class _BaseDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        transforms=None, 
        heatmap_sigma=6,
        weight_gamma=None
    ):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir
        
        self.subdirs = []
        for sd in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, sd)):
                self.subdirs.append(sd)
        
        # images and masks as dicts ordered by subdirectory
        self.impaths_dict = {}
        self.mskpaths_dict = {}
        self.confidences_dict = {}
        for sd in self.subdirs:
            self.impaths_dict[sd] = glob(os.path.join(data_dir, f'{sd}/images/*.tiff'))
            self.mskpaths_dict[sd] = glob(os.path.join(data_dir, f'{sd}/masks/*.tiff'))
            with open(os.path.join(data_dir, f'{sd}/confidences.json'), mode='r') as f:
                self.confidences_dict[sd] = json.load(f)
        
        # calculate weights per example, if weight gamma is not None
        self.weight_gamma = weight_gamma
        if weight_gamma is not None:
            self.weights = self._example_weights(self.impaths_dict, gamma=weight_gamma)
        else:
            self.weights = None
        
        # unpack dicts to lists of images
        self.impaths = []
        for paths in self.impaths_dict.values():
            self.impaths.extend(paths)
            
        self.mskpaths = []
        for paths in self.mskpaths_dict.values():
            self.mskpaths.extend(paths)
        
        print(f'Found {len(self.subdirs)} image subdirectories with {len(self.impaths)} images.')
        
        self.transforms = transforms
        self.heatmap_sigma = heatmap_sigma
        
    def __len__(self):
        return len(self.impaths)
    
    def __add__(self, add_dataset):
        # make a copy of self
        merged_dataset = deepcopy(self)
        
        # add the dicts and append lists/dicts
        for sd in add_dataset.impaths_dict.keys():
            if sd in merged_dataset.impaths_dict:
                # concat lists of paths together
                merged_dataset.impaths_dict[sd] += add_dataset.impaths_dict[sd]
                merged_dataset.mskpaths_dict[sd] += add_dataset.mskpaths_dict[sd]
                #concat dicts of paths and confidences together
                merged_dataset.confidences_dict[sd] = {
                    **merged_dataset.confidences_dict[sd], **add_dataset.confidences_dict[sd]
                }
            else:
                merged_dataset.impaths_dict[sd] = add_dataset.impaths_dict[sd]
                merged_dataset.mskpaths_dict[sd] = add_dataset.mskpaths_dict[sd]
                # concat dicts of paths and confidences together
                merged_dataset.confidences_dict[sd] = add_dataset.confidences_dict[sd]
        
        # recalculate weights
        if merged_dataset.weight_gamma is not None:
            merged_dataset.weights = self._example_weights(
                merged_dataset.impaths_dict, gamma=merged_dataset.weight_gamma
            )
        else:
            merged_dataset.weights = None
        
        # unpack dicts to lists of images
        merged_dataset.impaths = []
        for paths in merged_dataset.impaths_dict.values():
            merged_dataset.impaths.extend(paths)
            
        merged_dataset.mskpaths = []
        for paths in merged_dataset.mskpaths_dict.values():
            merged_dataset.mskpaths.extend(paths)
        
        return merged_dataset
        
    @staticmethod
    def _example_weights(paths_dict, gamma=0.3):
        # counts by source subdirectory
        counts = np.array(
            [len(paths) for paths in paths_dict.values()]
        )
        
        # invert and gamma the distribution
        weights = (1 / counts)
        weights = weights ** (gamma)
        
        # for interpretation, normalize weights 
        # s.t. they sum to 1
        total_weights = weights.sum()
        weights /= total_weights
        
        # repeat weights per n images
        example_weights = []
        for w,c in zip(weights, counts):
            example_weights.extend([w] * c)
            
        return torch.tensor(example_weights)
    
    def __getitem__(self, idx):
        raise NotImplementedError

@copy_paste_class
class MitoData(_BaseDataset):
    def __init__(
        self, 
        data_dir, 
        transforms=None, 
        heatmap_sigma=6,
        weight_gamma=0.3
    ):
        super(MitoData, self).__init__(
            data_dir, transforms, heatmap_sigma, weight_gamma
        )
        
        # used for copy-paste
        self._split_transforms()
        
    def load_example(self, idx):
        # get image and mask
        f = self.impaths[idx]
        image = cv2.imread(f, 0)
        mask = cv2.imread(self.mskpaths[idx], -1)
        assert mask.dtype == np.uint8, f'Mask {self.mskpaths[idx]} not 8-bit!'

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
        data = self.load_pasted_example(idx) #self.load_example(idx)
        
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

        # THIS DOESN'T WORK FOR COPY-PASTE!
        fdirs = f.split('/')
        subdir = fdirs[fdirs.index('images') - 1] # subdir is 1 before images dir
        output['fname'] = f
        
        # confidences are 1-5, subtract 1 to have 0-4 (for cross entropy loss)
        output['conf'] = self.confidences_dict[subdir][os.path.basename(f)] - 1

        # the last step is to binarize the mask for semantic segmentation
        output['sem'] = (ins_mask > 0).float()

        return output
    
class MitoDataQueue(_BaseDataset):
    def __init__(
        self, 
        data_dir, 
        transforms=None, 
        heatmap_sigma=6,
        weight_gamma=0.3
    ):
        super(MitoDataQueue, self).__init__(
            data_dir, transforms, heatmap_sigma, weight_gamma
        )
        
    def __getitem__(self, idx):
        # get image and mask
        f = self.impaths[idx]
        stack = io.imread(f)
        mask = cv2.imread(self.mskpaths[idx], -1)
        assert mask.dtype == np.uint8

        # apply initial transform to first image and mask
        if self.transforms is not None:
            transformed = self.transforms(image=stack[0], mask=mask)
            
            tf_stack = [transformed['image']]
            for image in stack[1:]:
                tf_stack.append(
                    A.ReplayCompose.replay(transformed['replay'], image=image)['image']
                )
            
            stack = torch.cat(tf_stack, dim=0)
            
        output = {}
        output['image'] = stack[None] # (L, H, W) -> (1, L, H, W)
        output['mask'] = transformed['mask']
        
        # add the filename to output dict
        output['fname'] = f
        
        # if not in inference mode, add the heatmap and centers
        if 'mask' in output:
            mask = output['mask']
            heatmap, offsets = heatmap_and_offsets(mask, self.heatmap_sigma)
            output['ctr_hmp'] = heatmap
            output['offsets'] = offsets
            
            #the last step is to binarize the mask for semantic segmentation
            output['sem'] = (mask > 0).float()
            del output['mask']

        return output
    
"""
class MitoData(_BaseDataset):
    def __init__(
        self, 
        data_dir, 
        tfs=None, 
        heatmap_sigma=6,
        weight_gamma=0.3
    ):
        super(MitoData, self).__init__(
            data_dir, tfs, heatmap_sigma, weight_gamma
        )

    def __getitem__(self, idx):
        # get image and mask
        f = self.impaths[idx]
        image = cv2.imread(f, 0)
        
        assert image.ndim == 2 or image.shape[2] == 1, \
        f'Dataset expects single channel grayscale got {image.shape}'
        
        mask = cv2.imread(self.mskpaths[idx], -1)
        assert mask.dtype == np.uint8

        data = {}
        # add channel dimension if needed
        if image.ndim == 2:
            image = image[..., None]
            
        data['image'] = image
        data['mask'] = mask

        if self.tfs is not None:
            output = self.tfs(**data)
            
        # add the filename to output dict
        output['fname'] = f
        
        # if not in inference mode, add the heatmap and centers
        if 'mask' in output:
            mask = output['mask']
            heatmap, offsets = heatmap_and_offsets(mask, self.heatmap_sigma)
            output['ctr_hmp'] = heatmap
            output['offsets'] = offsets
            
            fdirs = f.split('/')
            subdir = fdirs[fdirs.index('images') - 1] # subdir is 1 before images dir
            output['conf'] = self.confidences_dict[subdir][os.path.basename(f)]
            
            # the last step is to binarize the mask for semantic segmentation
            output['sem'] = (mask > 0).float()
            del output['mask']

        return output
"""
