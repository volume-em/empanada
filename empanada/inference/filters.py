import numpy as np
from empanada.array_utils import *

__all__ = [
    'remove_small_objects',
    'remove_pancakes'
]

def remove_small_objects(object_tracker, min_size=64):
    r"""Deletes small objects from an object tracker in-place

    Args:
        object_tracker: empanda.inference.trackers.InstanceTracker
        min_size: Integer, minimum size of object in voxels.
    """
    instance_ids = list(object_tracker.instances.keys())
    for instance_id in instance_ids:
        # sum all the runs in instance
        instance_attrs = object_tracker.instances[instance_id]
        size = instance_attrs['runs'].sum()

        if size < min_size:
            del object_tracker.instances[instance_id]

def remove_pancakes(object_tracker, min_span=4):
    r"""Deletes pancake-shaped objects from an object tracker in-place

    Args:
        object_tracker: empanda.inference.trackers.InstanceTracker
        min_span: Integer, the minimum extent of the objects bounding box.
    """
    instance_ids = list(object_tracker.instances.keys())
    for instance_id in instance_ids:
        # load the box of the instance
        instance_attrs = object_tracker.instances[instance_id]
        box = instance_attrs['box']

        zspan = box[3] - box[0]
        yspan = box[4] - box[1]
        xspan = box[5] - box[2]

        if any(span < min_span for span in [zspan, yspan, xspan]):
            del object_tracker.instances[instance_id]
