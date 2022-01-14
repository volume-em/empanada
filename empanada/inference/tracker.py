import os
import json
import math
import numpy as np
from skimage import measure
from itertools import combinations
from tqdm import tqdm
from empanada.array_utils import *

__all__ = [
    'InstanceTracker'
]

def to_box3d(index2d, box, axis):
    assert axis in ['xy', 'xz', 'yz']
    
    # extract box ranges
    h1, w1, h2, w2 = box
    if axis == 'xy':
        box3d = (index2d, h1, w1, index2d+1, h2, w2)
    elif axis == 'xz':
        box3d = (h1, index2d, w1, h2, index2d+1, w2)
    else:
        box3d = (h1, w1, index2d, h2, w2, index2d+1)

    return box3d

def to_coords3d(index2d, coords, axis):
    assert axis in ['xy', 'xz', 'yz']
    
    # split coords into arrays
    hcoords, wcoords = coords #tuple(coords.T)
    dcoords = np.repeat([index2d], len(hcoords))
    if axis == 'xy':
        coords3d = (dcoords, hcoords, wcoords)
    elif axis == 'xz':
        coords3d = (hcoords, dcoords, wcoords)
    else:
        coords3d = (hcoords, wcoords, dcoords)

    return coords3d

def to_starts3d(index2d, starts, axis, shape2d):
    assert axis in ['xy', 'xz', 'yz']
    
    hcoords, wcoords = tuple(coords.T)
    dcoords = np.repeat([index2d], len(hcoords))
    if axis == 'xy':
        coords3d = (dcoords, hcoords, wcoords)
    elif axis == 'xz':
        coords3d = (hcoords, dcoords, wcoords)
    else:
        coords3d = (hcoords, wcoords, dcoords)

    return coords3d

"""
class InstanceTracker:
    def __init__(
        self,
        class_id=None,
        label_divisor=None,
        shape3d=None,
        axis='xy'
    ):
        assert axis in ['xy', 'xz', 'yz']
        self.class_id = class_id
        self.label_divisor = label_divisor
        self.shape3d = shape3d
        self.axis = axis
        self.finished = False
        self.reset()
        
    def reset(self):
        self.instances = {}
        
    def update(self, pan_seg, index2d):
        assert self.class_id is not None
        assert self.label_divisor is not None
        assert self.shape3d is not None
        assert not self.finished, "Cannot update tracker after calling finish!"
        
        # determine the valid instance ids for
        # objects of the given class id
        # (see postprocess.py for details)
        pan_seg = np.copy(pan_seg) # copy for safety
        min_id = self.class_id * self.label_divisor
        max_id = min_id + self.label_divisor
        
        # zero all objects/semantic segs outside of instance_id range
        pan_seg[np.logical_or(pan_seg < min_id, pan_seg >= max_id)] = 0
        
        # extract bounding box and object pixels coords
        rps = measure.regionprops(pan_seg)
        for rp in rps:
            box = to_box3d(index2d, rp.bbox, self.axis)
            coords = to_coords3d(index2d, rp.coords, self.axis)
            
            # convert the coords to raveled indices
            coords_flat = np.ravel_multi_index(coords, self.shape3d)
            
            # update instances dict
            if rp.label not in self.instances:
                starts, runs = rle_encode(coords_flat)
                self.instances[rp.label] = {
                    'box': box, 'starts': [starts], 'runs': [runs]
                }
            else:
                # merge boxes and coords
                instance_dict = self.instances[rp.label]
                starts, runs = rle_encode(coords_flat)
                self.instances[rp.label]['box'] = merge_boxes3d(box, instance_dict['box'])
                self.instances[rp.label]['starts'].append(starts)
                self.instances[rp.label]['runs'].append(runs)
                
    def finish(self):
        # concat all starts and runs
        for instance_id in self.instances.keys():
            if isinstance(self.instances[instance_id]['starts'] , list):
                starts = np.concatenate(self.instances[instance_id]['starts'])
                runs = np.concatenate(self.instances[instance_id]['runs'])
                self.instances[instance_id]['starts'] = starts
                self.instances[instance_id]['runs'] = runs
            else:
                # if starts/runs are not lists, then
                # they've already been concatenated
                continue
            
        self.finished = True
                
    def write_to_json(self, savepath):
        if not self.finished:
            self.finish()
        
        save_dict = self.__dict__
        # convert instance coords to string
        for k in save_dict['instances'].keys():
            starts = save_dict['instances'][k]['starts']
            runs = save_dict['instances'][k]['runs']
            
            save_dict['instances'][k]['rle'] = \
            rle_to_string(starts, runs)
            
            del save_dict['instances'][k]['starts']
            del save_dict['instances'][k]['runs']
        
        with open(savepath, mode='w') as handle:
            json.dump(save_dict, handle, indent=6)
        
    def load_from_json(self, fpath):
        with open(fpath, mode='r') as handle:
            load_dict = json.load(handle)
            
        # convert instance string coords to arrays
        for k in load_dict['instances'].keys():
            rle_string = load_dict['instances'][k]['rle']
            starts, runs = string_to_rle(rle_string)
            load_dict['instances'][k]['starts'] = starts
            load_dict['instances'][k]['runs'] = runs
            
        self.__dict__ = load_dict
"""

class InstanceTracker:
    def __init__(
        self,
        class_id=None,
        label_divisor=None,
        shape3d=None,
        axis='xy'
    ):
        assert axis in ['xy', 'xz', 'yz']
        self.class_id = class_id
        self.label_divisor = label_divisor
        self.shape3d = shape3d
        self.axis = axis
        self.finished = False
        self.reset()
        
        self.axis_nums = {'xy': 0, 'xz': 1, 'yz': 2}
        
    def reset(self):
        self.instances = {}
        
    def update(self, instance_rles, index2d):
        assert self.class_id is not None
        assert self.label_divisor is not None
        assert self.shape3d is not None
        assert not self.finished, "Cannot update tracker after calling finish!"
        
        # extract bounding box and object pixels coords
        for label,attrs in instance_rles.items():
            box = to_box3d(index2d, attrs['box'], self.axis)
            ignore_idx = self.axis_nums[self.axis]
                
            shape2d = tuple([s for i,s in enumerate(self.shape3d) if i != ignore_idx])

            # convert to 3d starts and runs
            if self.axis == 'xy':
                starts = attrs['starts'] + index2d * math.prod(shape2d)
                runs = attrs['runs']
            elif self.axis == 'xz':
                coords2d = np.unravel_index(attrs['starts'], shape2d)
                coords3d = to_coords3d(index2d, coords2d, 'xz')
                starts = np.ravel_multi_index(coords3d, self.shape3d)
                runs = attrs['runs']
            else:
                coords2d_flat = rle_decode(attrs['starts'], attrs['runs'])
                coords2d = np.unravel_index(coords2d_flat, shape2d)
                coords3d = to_coords3d(index2d, coords2d, 'yz')
                starts = np.ravel_multi_index(coords3d, self.shape3d)
                runs = np.ones_like(starts)
            
            # update instances dict
            if label not in self.instances:
                self.instances[label] = {
                    'box': box, 'starts': [starts], 'runs': [runs]
                }
            else:
                # merge boxes and coords
                instance_dict = self.instances[label]
                self.instances[label]['box'] = merge_boxes3d(box, instance_dict['box'])
                self.instances[label]['starts'].append(starts)
                self.instances[label]['runs'].append(runs)
                
    def finish(self):
        # concat all starts and runs
        for instance_id in self.instances.keys():
            if isinstance(self.instances[instance_id]['starts'] , list):
                starts = np.concatenate(self.instances[instance_id]['starts'])
                runs = np.concatenate(self.instances[instance_id]['runs'])
                self.instances[instance_id]['starts'] = starts
                self.instances[instance_id]['runs'] = runs
            else:
                # if starts/runs are not lists, then
                # they've already been concatenated
                continue
            
        self.finished = True
                
    def write_to_json(self, savepath):
        if not self.finished:
            self.finish()
        
        save_dict = self.__dict__
        # convert instance coords to string
        for k in save_dict['instances'].keys():
            starts = save_dict['instances'][k]['starts']
            runs = save_dict['instances'][k]['runs']
            
            save_dict['instances'][k]['rle'] = \
            rle_to_string(starts, runs)
            
            del save_dict['instances'][k]['starts']
            del save_dict['instances'][k]['runs']
            
        for k,v in list(save_dict['instances'].items()):
            save_dict['instances'][str(k)] = v
            del save_dict['instances'][k]
        
        with open(savepath, mode='w') as handle:
            json.dump(save_dict, handle, indent=6)
        
    def load_from_json(self, fpath):
        with open(fpath, mode='r') as handle:
            load_dict = json.load(handle)
            
        # convert instance string coords to arrays
        for k in load_dict['instances'].keys():
            rle_string = load_dict['instances'][k]['rle']
            starts, runs = string_to_rle(rle_string)
            load_dict['instances'][k]['starts'] = starts
            load_dict['instances'][k]['runs'] = runs
            
        self.__dict__ = load_dict
