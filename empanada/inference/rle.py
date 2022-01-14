import numpy as np
from empanada.array_utils import *
from skimage import measure
    
def pan_seg_to_rle_seg(
    pan_seg, 
    labels, 
    label_divisor,
    force_connected=True
):
    # convert from dense panoptic seg to sparse rle segment class
    rle_seg = {}
    for label in labels:
        # convert from pan_seg to instance_seg
        min_id = label * label_divisor
        max_id = min_id + label_divisor
        
        # zero all objects/semantic segs outside of instance_id range
        instance_seg = pan_seg.copy()
        outside_mask = np.logical_or(pan_seg < min_id, pan_seg >= max_id)
        instance_seg[outside_mask] = 0
        
        # relabel connected components
        if force_connected:
            instance_seg = measure.label(instance_seg).astype(instance_seg.dtype)
            instance_seg[instance_seg > 0] += min_id
        
        # measure the regionprops
        instance_attrs = {}
        rps = measure.regionprops(instance_seg)
        for rp in rps:
            # convert from label xy coords to rles
            coords_flat = np.ravel_multi_index(tuple(rp.coords.T), instance_seg.shape)
            starts, runs = rle_encode(coords_flat)
            
            instance_attrs[rp.label] = {'box': rp.bbox, 'starts': starts, 'runs': runs}
            
        # add to the rle_seg
        rle_seg[label] = instance_attrs
        
    return rle_seg

def rle_seg_to_pan_seg(
    rle_seg,
    shape
):
    # convert from dense panoptic seg to sparse rle segment class
    pan_seg = np.zeros(shape, dtype='int').ravel()
    
    for instance_attrs in rle_seg.values():
        for object_id, attrs in instance_attrs.items():
            starts = attrs['starts']
            runs = attrs['runs']
            
            for s,r in zip(starts, runs):
                pan_seg[s:s+r] = object_id
                
    return pan_seg.reshape(shape)