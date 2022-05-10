import pytest
import numpy as np
import torch
from numpy.testing import assert_equal, assert_almost_equal
from empanada import metrics

@pytest.fixture
def sem_gt_pred_null():
    gt = torch.zeros((1, 1, 128, 128)).float()
    pred = torch.zeros((1, 1, 128, 128)).float()

    return {'sem': gt}, {'sem_logits': pred}

@pytest.fixture
def sem_gt_pred_binary():
    # create a gt with 2 squares (top left
    # and bottom right)
    gt = torch.zeros((1, 1, 128, 128)).float()
    gt[..., :32, :32] = 1.
    gt[..., -32:, -32:] = 1.

    # create pred with 1 square
    # of pos and 1 square of neg
    # logits
    pred = torch.zeros((1, 1, 128, 128)).float()
    s1 = torch.randn((1, 1, 32, 32))
    s2 = torch.randn((1, 1, 32, 32))

    s1_min = s1.min() - 1e-7
    s2_max = s2.max() + 1e-7

    s1 -= s1_min # guarantees all > 0
    s2 -= s2_max # guarantees all < 0

    pred[..., :32, :32] = s1
    pred[..., -32:, -32:] = s2

    return {'sem': gt}, {'sem_logits': pred}

@pytest.fixture
def sem_gt_pred_multiclass():
    # class labels in different squares
    gt = torch.zeros((1, 1, 128, 128)).long()
    gt[..., :32, :32] = 1
    gt[..., -32:, -32:] = 2

    # fill logits to match gt
    pred = -1e9 * torch.ones((1, 3, 128, 128)).float()
    s0 = torch.randn((1, 1, 128, 128))
    s1 = torch.randn((1, 1, 32, 32))
    s2 = torch.randn((1, 1, 32, 32))

    s0_min = s0.min() - 1e-7
    s1_min = s1.min() - 1e-7
    s2_min = s2.min() - 1e-7

    # guarantees all > 0
    s0 -= s0_min
    s1 -= s1_min
    s2 -= s2_min

    s0[..., 0, :32, :32] = -1e9
    s0[..., 0, -32:, -32:] = -1e9

    # fill class channels
    pred[..., 0, :, :] = s0
    pred[..., 1, :32, :32] = s1
    pred[..., 2, -32:, -32:] = s2

    return {'sem': gt}, {'sem_logits': pred}

@pytest.fixture
def panoptic_gt_pred():
    # 1 semantic and 1 instance class
    gt = torch.zeros((1, 1, 128, 128)).long()
    gt[..., :32, :32] = 1001
    gt[..., :32, -32:] = 2001
    gt[..., -32:, -32:] = 2002

    # fill logits to match gt
    pred = torch.zeros((1, 1, 128, 128)).long()
    pred[..., :32, :32] = 1001
    pred[..., :15, -32:] = 2002 # just under 0.5 iou
    pred[..., -32:, -32:] = 2001

    return {'pan_seg': gt}, {'pan_seg': pred}

@pytest.mark.parametrize(
    'gt_pred, labels, expected',
    [
        ('sem_gt_pred_null', [1], [1.]),
        ('sem_gt_pred_binary', [1], [0.5]),
        ('sem_gt_pred_multiclass', [1, 2], [1., 1.])
    ]
)
def test_iou(gt_pred, labels, expected, request):
    gt, pred = request.getfixturevalue(gt_pred)
    iou_cls = metrics.IoU(metrics.AverageMeter, labels, 'sem_logits', 'sem')
    iou_dict = iou_cls.calculate(pred, gt)
    for i, (l,v) in enumerate(iou_dict.items()):
        assert_almost_equal(float(v), expected[i], decimal=3)

@pytest.mark.parametrize(
    'gt_pred, labels, iou_thr, expected_pq, expected_f1',
    [
        ('panoptic_gt_pred', [1, 2], 0.4, [1., 0.5], 1.),
    ]
)
def test_f1_pq(gt_pred, labels, iou_thr, expected_pq, expected_f1, request):
    gt, pred = request.getfixturevalue(gt_pred)

    pq_cls = metrics.PQ(metrics.AverageMeter, labels, 1000, 'pan_seg', 'pan_seg')
    pq_dict = pq_cls.calculate(pred, gt)
    for i, (l,v) in enumerate(pq_dict.items()):
        assert_almost_equal(float(v), expected_pq[i], decimal=3)

    f1_cls = metrics.F1(metrics.AverageMeter, [2], 1000, iou_thr, 'pan_seg', 'pan_seg')
    f1_dict = f1_cls.calculate(pred, gt)
    assert_almost_equal(float(f1_dict[2]), expected_f1, decimal=3)
