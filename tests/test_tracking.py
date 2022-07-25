import os
import pytest
import random
import zarr
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from empanada import array_utils, zarr_utils
from empanada.inference import rle, tracker

def test_tracking_and_filling(tmp_path):
    tmp_path = str(tmp_path)

    # small fake instance segmentation
    vol = np.random.randint(0, 6, size=(100, 100, 100), dtype=np.uint32)

    # add the label_divisor
    vol[vol > 0] += 1000

    # apply tracking in all axes
    xy_tracker = tracker.InstanceTracker(1, 1000, vol.shape, axis='xy')
    xz_tracker = tracker.InstanceTracker(1, 1000, vol.shape, axis='xz')
    yz_tracker = tracker.InstanceTracker(1, 1000, vol.shape, axis='yz')
    for i in range(100):
        xy_rle = rle.pan_seg_to_rle_seg(vol[i], [1], 1000, [1], False)
        xy_tracker.update(xy_rle[1], i)

        xz_rle = rle.pan_seg_to_rle_seg(vol[:, i], [1], 1000, [1], False)
        xz_tracker.update(xz_rle[1], i)

        yz_rle = rle.pan_seg_to_rle_seg(vol[..., i], [1], 1000, [1], False)
        yz_tracker.update(yz_rle[1], i)

        # check that reverse rle op works
        assert_equal(rle.rle_seg_to_pan_seg(xy_rle, vol[i].shape), vol[i])

    # finish tracking
    xy_tracker.finish()
    xz_tracker.finish()
    yz_tracker.finish()

    # write tracker to json and read it back in
    xy_tracker.write_to_json(os.path.join(tmp_path, 'xy_tracker.json'))
    xy_tracker.load_from_json(os.path.join(tmp_path, 'xy_tracker.json'))

    # fill volumes from tracker
    xy_vol = np.zeros_like(vol)
    xz_vol = np.zeros_like(vol)
    yz_vol = np.zeros_like(vol)

    xy_vol = array_utils.numpy_fill_instances(xy_vol, xy_tracker.instances)
    xz_vol = array_utils.numpy_fill_instances(xy_vol, xy_tracker.instances)
    yz_vol = array_utils.numpy_fill_instances(xy_vol, xy_tracker.instances)

    assert_equal(vol, xy_vol)
    assert_equal(xy_vol, xz_vol)
    assert_equal(xz_vol, yz_vol)

    # write to zarr and validate results are the same
    zarray = zarr.open(os.path.join(tmp_path, 'test.zarr'), mode='w')
    # verify that arbitary chunk shapes work
    for _ in range(20):
        chunks = tuple(random.randint(10, 100) for _ in range(3))
        zarray.create_dataset(
            'seg', shape=(100, 100, 100), dtype=np.uint32,
            overwrite=True, chunks=chunks
        )

        zvol = zarray['seg']
        zarr_utils.zarr_fill_instances(zvol, xy_tracker.instances, 4)
        zvol = np.array(zvol)
        assert_equal(zvol, xy_vol)
