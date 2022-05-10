import os
import pytest
import numpy as np
import torch
from skimage import io
from numpy.testing import assert_equal, assert_almost_equal
from empanada import data
from empanada import metrics
from empanada.inference import postprocess

TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data'))

def test_datasets_and_postprocessing():
    # create panoptic dataset
    pan_data = data.PanopticDataset(
        os.path.join(TEST_DIR, 'panoptic'), [1, 2, 3], [2],
        1000, transforms=None
    )

    # load the preprocessed data
    pan_ex = pan_data[0]
    sem = torch.from_numpy(pan_ex['sem'])[None, None]
    ctr_hmp = torch.from_numpy(pan_ex['ctr_hmp']).unsqueeze(0)
    offsets = torch.from_numpy(pan_ex['offsets']).unsqueeze(0)

    # load the ground truth
    pan_seg = torch.from_numpy(
        io.imread(os.path.join(TEST_DIR, 'panoptic/dataset1/masks/pan_seg.tiff'))
    )

    # postprocess the targets
    pan_seg_post = postprocess.get_panoptic_segmentation(
        sem, ctr_hmp, offsets, [2], 1000, 0, 0, 0.1, 7
    )[0]

    # label values change but PQ scores should be
    # ~1. for all classes
    pq_cls = metrics.PQ(metrics.AverageMeter, [1, 2, 3], 1000, 'pan_seg', 'pan_seg')
    pq_dict = pq_cls.calculate({'pan_seg': pan_seg_post}, {'pan_seg': pan_seg})
    for i, (l,v) in enumerate(pq_dict.items()):
        assert_almost_equal(float(v), 1., decimal=3)

    # create panoptic dataset
    ins_data = data.SingleClassInstanceDataset(
        os.path.join(TEST_DIR, 'instance'), transforms=None
    )

    # load the preprocessed data
    ins_ex = ins_data[0]
    sem = torch.from_numpy(ins_ex['sem'])[None, None]
    ctr_hmp = torch.from_numpy(ins_ex['ctr_hmp']).unsqueeze(0)
    offsets = torch.from_numpy(ins_ex['offsets']).unsqueeze(0)

    # load the ground truth
    ins_seg = torch.from_numpy(
        io.imread(os.path.join(TEST_DIR, 'instance/dataset1/masks/ins_seg.tiff'))
    )

    # postprocess the targets
    ins_seg_post = postprocess.get_panoptic_segmentation(
        sem, ctr_hmp, offsets, [1], 1000, 0, 0, 0.1, 7
    )[0]

    # label values change but F1 scores should be
    # ~1. for all classes
    f1_cls = metrics.F1(metrics.AverageMeter, [1], 1000, 0.5, 'pan_seg', 'pan_seg')
    f1_dict = f1_cls.calculate({'pan_seg': ins_seg_post}, {'pan_seg': ins_seg})
    for i, (l,v) in enumerate(f1_dict.items()):
        assert_almost_equal(float(v), 1., decimal=3)
