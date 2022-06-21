import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from empanada import array_utils

@pytest.fixture
def boxes1():
    z1 = np.array([0, 20, 30])
    z2 = np.array([50, 40, 50])

    y1 = np.array([0, 10, 20])
    y2 = np.array([20, 50, 80])
    x1 = np.array([0, 30, 10])
    x2 = np.array([40, 80, 50])

    boxes2d = np.stack([y1, x1, y2, x2], axis=1)
    boxes3d = np.stack([z1, y1, x1, z2, y2, x2], axis=1)

    return boxes2d, boxes3d

@pytest.fixture
def boxes2():
    z1 = np.array([5, 25])
    z2 = np.array([50, 40])

    y1 = np.array([5, 15])
    y2 = np.array([20, 50])
    x1 = np.array([5, 35])
    x2 = np.array([40, 80])

    boxes2d = np.stack([y1, x1, y2, x2], axis=1)
    boxes3d = np.stack([z1, y1, x1, z2, y2, x2], axis=1)

    return boxes2d, boxes3d

def test_box_functions(boxes1, boxes2):
    # 2D area case
    area2d = np.array([20 * 40, 40 * 50, 60 * 40])
    assert_almost_equal(array_utils.box_area(boxes1[0]), area2d)

    # 3D volume case
    area3d = np.array([50 * 20 * 40, 20 * 40 * 50, 20 * 60 * 40])
    assert_almost_equal(array_utils.box_area(boxes1[1]), area3d)

    # 2D pairwise intersection case
    pint2d = np.array([
        [20 * 40, 10 * 10, 0 * 30],
        [10 * 10, 40 * 50, 30 * 20],
        [0 * 30, 30 * 20, 60 * 40],
    ])
    assert_almost_equal(
        array_utils.box_intersection(boxes1[0]), pint2d
    )

    # 3D pairwise intersection case
    pint3d = np.array([
        [50 * 20 * 40, 20 * 10 * 10, 20 * 0 * 30],
        [20 * 10 * 10, 20 * 40 * 50, 10 * 30 * 20],
        [20 * 0 * 30, 10 * 30 * 20, 20 * 60 * 40],
    ])
    assert_almost_equal(
        array_utils.box_intersection(boxes1[1]), pint3d
    )

    # 2D combination intersection case
    cint2d =  np.array([
        [15 * 35, 5 * 5],
        [10 * 10, 35 * 45],
        [0 * 30, 30 * 15]
    ])
    assert_almost_equal(
        array_utils.box_intersection(boxes1[0], boxes2[0]), cint2d
    )

    # 3D combination intersection case
    cint3d = np.array([
        [45 * 15 * 35, 15 * 5 * 5],
        [20 * 10 * 10, 15 * 35 * 45],
        [20 * 0 * 30, 10 * 30 * 15]
    ])
    assert_almost_equal(
        array_utils.box_intersection(boxes1[1], boxes2[1]), cint3d
    )

    assert_almost_equal(
        array_utils.merge_boxes(boxes1[0][0], boxes2[0][1]),
        np.array([0, 0, 50, 80])
    )

    assert_almost_equal(
        array_utils.merge_boxes(boxes1[1][0], boxes2[1][1]),
        np.array([0, 0, 0, 50, 50, 80])
    )

    # 2D pairwise iou
    union2d = area2d[:, None] + area2d[None, :] - pint2d
    assert_almost_equal(
        array_utils.box_iou(boxes1[0]).todense(),
        pint2d / union2d
    )

    # 3D pairwise iou
    union3d = area3d[:, None] + area3d[None, :] - pint3d
    assert_almost_equal(
        array_utils.box_iou(boxes1[1]).todense(),
        pint3d / union3d
    )

def test_rle_functions():
    # pick and sort a list of indices
    indices1 = np.sort(np.random.choice(range(200), replace=False, size=(120,)))
    indices2 = np.sort(np.random.choice(range(200), replace=False, size=(100,)))

    inter = len(np.intersect1d(indices1, indices2))
    union = len(np.union1d(indices1, indices2))
    iou = inter / union
    ioa = inter / len(indices2)

    starts1, runs1 = array_utils.rle_encode(indices1)
    starts2, runs2 = array_utils.rle_encode(indices2)

    # verify encode and decode are correct
    assert_equal(array_utils.rle_decode(starts1, runs1), indices1)
    assert_equal(array_utils.rle_decode(starts2, runs2), indices2)

    # from string
    rle_str = array_utils.rle_to_string(starts1, runs1)
    str_s, str_r = array_utils.string_to_rle(rle_str)
    assert_equal(str_s, starts1)
    assert_equal(str_r, runs1)

    assert array_utils.rle_intersection(starts1, runs1, starts2, runs2) == inter
    assert array_utils.rle_iou(starts1, runs1, starts2, runs2) == iou
    assert array_utils.rle_ioa(starts1, runs1, starts2, runs2) == ioa
    assert_equal(
        array_utils.rle_decode(*array_utils.merge_rles(starts1, runs1, starts2, runs2)),
        np.union1d(indices1, indices2)
    )

    # index and rle voting should match
    indices3 = np.sort(np.random.choice(range(200), replace=False, size=(150,)))
    starts3, runs3 = array_utils.rle_encode(indices3)

    all_indices = np.sort(np.concatenate([indices1, indices2, indices3]))
    unq, votes = np.unique(all_indices, return_counts=True)

    all_ranges = [
        np.stack([s, s + r], axis=1)
        for s,r in zip([starts1, starts2, starts3], [runs1, runs2, runs3])
    ]
    voted_ranges = array_utils.vote_by_ranges(all_ranges, 2)

    voted_starts, voted_runs = np.split(array_utils.ranges_to_rle(voted_ranges), 2, axis=1)
    assert_equal(array_utils.rle_decode(voted_starts, voted_runs), unq[votes >= 2])