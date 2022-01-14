import os
import json
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
from copy import deepcopy

__all__ = [
    'BaseDataset'
]

class _BaseDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        transforms=None, 
        heatmap_sigma=6,
        weight_gamma=None,
        has_confidence=False
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
        if has_confidence:
            self.confidences_dict = {}
        else:
            self.confidences_dict = None
            
        for sd in self.subdirs:
            self.impaths_dict[sd] = glob(os.path.join(data_dir, f'{sd}/images/*'))
            self.mskpaths_dict[sd] = glob(os.path.join(data_dir, f'{sd}/masks/*'))
            
            # load confidences json if needed
            if self.confidences_dict is not None:
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
        self.has_confidence = has_confidence
        
    def __len__(self):
        return len(self.impaths)
    
    def __add__(self, add_dataset):
        # make a copy of self
        merged_dataset = deepcopy(self)
        
        if self.has_confidence:
            assert add_data.confidences_dict is not None, \
            "Cannot merge a dataset without confidence scores to one with confidence scores"
        
        # add the dicts and append lists/dicts
        for sd in add_dataset.impaths_dict.keys():
            if sd in merged_dataset.impaths_dict:
                # concat lists of paths together
                merged_dataset.impaths_dict[sd] += add_dataset.impaths_dict[sd]
                merged_dataset.mskpaths_dict[sd] += add_dataset.mskpaths_dict[sd]
                #concat dicts of paths and confidences together
                if self.has_confidence:
                    merged_dataset.confidences_dict[sd] = {
                        **merged_dataset.confidences_dict[sd], **add_dataset.confidences_dict[sd]
                    }
            else:
                merged_dataset.impaths_dict[sd] = add_dataset.impaths_dict[sd]
                merged_dataset.mskpaths_dict[sd] = add_dataset.mskpaths_dict[sd]
                # concat dicts of paths and confidences together
                if self.has_confidence:
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
