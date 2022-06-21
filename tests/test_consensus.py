import os
import pytest
import zarr
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from empanada import array_utils, consensus
from empanada.inference import rle, tracker
from skimage.morphology import ball

def make_spheres():
    s1 = ball(10).astype(np.uint32)
    s2 = ball(20).astype(np.uint32)
    s3 = ball(15).astype(np.uint32)
    s4 = s2.copy()
    s4[:, 20:, 20:] = 0
    
    return s1, s2, s3, s4

@pytest.fixture
def object_trackers():
    # create some volumes and trackers
    shape = (100, 100, 100)
    xy_vol = np.zeros(shape, dtype=np.uint32)
    xz_vol = np.zeros(shape, dtype=np.uint32)
    yz_vol = np.zeros(shape, dtype=np.uint32)

    s1, s2, s3, s4 = make_spheres()
    
    # fill in some instances
    xy_vol[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1001
    xy_vol[15:56, 15:56, 15:56][s2 > 0] = s2[s2 > 0] * 1002

    xz_vol[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1005
    xz_vol[15:56, 15:56, 15:56][s4 > 0] = s4[s4 > 0] * 1004
    xz_vol[:41, 59:100, 59:100][s2 > 0] = s2[s2 > 0] * 1006

    yz_vol[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1003
    yz_vol[15:56, 15:56, 15:56][s4 > 0] = s4[s4 > 0] * 1003
    
    # make trackers from the volumes
    xy_tracker = tracker.InstanceTracker(1, 1000, shape, axis='xy')
    xz_tracker = tracker.InstanceTracker(1, 1000, shape, axis='xy')
    yz_tracker = tracker.InstanceTracker(1, 1000, shape, axis='xy')

    for i, (xy,xz,yz) in enumerate(zip(xy_vol, xz_vol, yz_vol)):
        # convert to an rle_seg and update
        rle_seg_xy = rle.pan_seg_to_rle_seg(xy, [1], 1000, [1], force_connected=False)
        rle_seg_xz = rle.pan_seg_to_rle_seg(xz, [1], 1000, [1], force_connected=False)
        rle_seg_yz = rle.pan_seg_to_rle_seg(yz, [1], 1000, [1], force_connected=False)

        xy_tracker.update(rle_seg_xy[1], i)
        xz_tracker.update(rle_seg_xz[1], i)
        yz_tracker.update(rle_seg_yz[1], i)

    xy_tracker.finish()
    xz_tracker.finish()
    yz_tracker.finish()
    
    return xy_tracker, xz_tracker, yz_tracker

@pytest.fixture
def default_consensus():
    shape = (100, 100, 100)
    s1, s2, s3, s4 = make_spheres()
    
    cons_out = np.zeros(shape, dtype=np.uint32)
    cons_out[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1
    cons_out[15:56, 15:56, 15:56][s4 > 0] = s4[s4 > 0] * 2
    
    return cons_out

@pytest.fixture
def lower_cluster_thr_consensus():
    shape = (100, 100, 100)
    s1, s2, s3, s4 = make_spheres()
    
    cons_out = np.zeros(shape, dtype=np.uint32)
    cons_out[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1
    cons_out[15:56, 15:56, 15:56][s4 > 0] = s4[s4 > 0] * 1
    
    return cons_out

@pytest.fixture
def lower_pixel_thr_consensus():
    shape = (100, 100, 100)
    s1, s2, s3, s4 = make_spheres()
    
    cons_out = np.zeros(shape, dtype=np.uint32)
    cons_out[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1
    cons_out[15:56, 15:56, 15:56][s2 > 0] = s2[s2 > 0] * 1
    
    return cons_out

@pytest.fixture
def bypass_consensus():
    shape = (100, 100, 100)
    s1, s2, s3, s4 = make_spheres()
    
    cons_out = np.zeros(shape, dtype=np.uint32)
    cons_out[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1
    cons_out[15:56, 15:56, 15:56][s2 > 0] = s2[s2 > 0] * 1
    cons_out[:41, 59:100, 59:100][s2 > 0] = s2[s2 > 0] * 2
    
    return cons_out

@pytest.fixture
def semantic_consensus_default():
    shape = (100, 100, 100)
    s1, s2, s3, s4 = make_spheres()
    
    cons_out = np.zeros(shape, dtype=np.uint32)
    cons_out[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1
    cons_out[15:56, 15:56, 15:56][s4 > 0] = s4[s4 > 0] * 1
    
    return cons_out

@pytest.fixture
def semantic_consensus_lower_thr():
    shape = (100, 100, 100)
    s1, s2, s3, s4 = make_spheres()
    
    cons_out = np.zeros(shape, dtype=np.uint32)
    cons_out[:41, :41, :41][s2 > 0] = s2[s2 > 0] * 1
    cons_out[15:56, 15:56, 15:56][s2 > 0] = s2[s2 > 0] * 1
    cons_out[:41, 59:100, 59:100][s2 > 0] = s2[s2 > 0] * 1
    
    return cons_out

def test_default_consensus(object_trackers, default_consensus):
    xy_tracker, xz_tracker, yz_tracker = object_trackers
    
    cons = consensus.merge_objects_from_trackers([xy_tracker, xz_tracker, yz_tracker], pixel_vote_thr=2, cluster_iou_thr=0.75, bypass=False)
    cons_v = array_utils.numpy_fill_instances(np.zeros(xy_tracker.shape3d, dtype=np.uint32), cons)
    
    assert_equal(cons_v, default_consensus)
    
def test_cluster_thr_consensus(object_trackers, lower_cluster_thr_consensus):
    xy_tracker, xz_tracker, yz_tracker = object_trackers
    
    cons = consensus.merge_objects_from_trackers([xy_tracker, xz_tracker, yz_tracker], pixel_vote_thr=2, cluster_iou_thr=0.5, bypass=False)
    cons_v = array_utils.numpy_fill_instances(np.zeros(xy_tracker.shape3d, dtype=np.uint32), cons)

    assert_equal(cons_v, lower_cluster_thr_consensus)
    
def test_pixel_thr_consensus(object_trackers, lower_pixel_thr_consensus):
    xy_tracker, xz_tracker, yz_tracker = object_trackers
    
    cons = consensus.merge_objects_from_trackers([xy_tracker, xz_tracker, yz_tracker], pixel_vote_thr=1, cluster_iou_thr=0.75, bypass=False)
    cons_v = array_utils.numpy_fill_instances(np.zeros(xy_tracker.shape3d, dtype=np.uint32), cons)
    
    assert_equal(cons_v, lower_pixel_thr_consensus)
    
def test_bypass_consensus(object_trackers, bypass_consensus):
    xy_tracker, xz_tracker, yz_tracker = object_trackers
    
    cons = consensus.merge_objects_from_trackers([xy_tracker, xz_tracker, yz_tracker], pixel_vote_thr=1, cluster_iou_thr=0.75, bypass=True)
    cons_v = array_utils.numpy_fill_instances(np.zeros(xy_tracker.shape3d, dtype=np.uint32), cons)
    
    assert_equal(cons_v, bypass_consensus)
    
def test_semantic_consensus_default(object_trackers, semantic_consensus_default):
    xy_tracker, xz_tracker, yz_tracker = object_trackers
    
    # convert from instance to semantic trackers
    instance_attr = consensus.merge_instances(xy_tracker.instances)
    xy_tracker.instances = {1001: instance_attr}
    
    instance_attr = consensus.merge_instances(xz_tracker.instances)
    xz_tracker.instances = {1001: instance_attr}
    
    instance_attr = consensus.merge_instances(yz_tracker.instances)
    yz_tracker.instances = {1001: instance_attr}
    
    cons = consensus.merge_semantic_from_trackers([xy_tracker, xz_tracker, yz_tracker], pixel_vote_thr=2)
    cons_v = array_utils.numpy_fill_instances(np.zeros(xy_tracker.shape3d, dtype=np.uint32), cons)
    
    assert_equal(cons_v, semantic_consensus_default)
    
def test_semantic_consensus_lower_thr(object_trackers, semantic_consensus_lower_thr):
    xy_tracker, xz_tracker, yz_tracker = object_trackers
    
    # convert from instance to semantic trackers
    instance_attr = consensus.merge_instances(xy_tracker.instances)
    xy_tracker.instances = {1001: instance_attr}
    
    instance_attr = consensus.merge_instances(xz_tracker.instances)
    xz_tracker.instances = {1001: instance_attr}
    
    instance_attr = consensus.merge_instances(yz_tracker.instances)
    yz_tracker.instances = {1001: instance_attr}
    
    cons = consensus.merge_semantic_from_trackers([xy_tracker, xz_tracker, yz_tracker], pixel_vote_thr=1)
    cons_v = array_utils.numpy_fill_instances(np.zeros(xy_tracker.shape3d, dtype=np.uint32), cons)
    
    assert_equal(cons_v, semantic_consensus_lower_thr)