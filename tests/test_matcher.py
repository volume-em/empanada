import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from empanada.inference import rle, matcher

@pytest.fixture
def target_seg():
    seg = np.zeros((200, 200), dtype=np.uint32)

    # create multiple instances
    seg[:16, :16] = 1001
    seg[30:50, 30:50] = 1002
    seg[:10, -10:] = 1003
    seg[-50:, :50] = 1004
    seg[-30:, -30:] = 1005
    seg[100:130, 90:110] = 1006

    return seg

@pytest.fixture
def match_seg():
    seg = np.zeros((200, 200), dtype=np.uint32)

    # create multiple instances
    seg[:16, :16] = 1009
    seg[30:50, 30:50] = 1008
    seg[:10, -10:] = 1007
    seg[-50:, :50] = 1006
    seg[-30:, 45:80] = 1005
    seg[-20:, -20:] = 1004
    seg[100:115, 90:110] = 1003
    seg[115:130, 90:110] = 1002
    seg[50:75, 125:160] = 1001

    return seg

@pytest.fixture
def out_seg():
    seg = np.zeros((200, 200), dtype=np.uint32)

    # create multiple instances
    seg[:16, :16] = 1001
    seg[30:50, 30:50] = 1002
    seg[:10, -10:] = 1003
    seg[-50:, :50] = 1004
    seg[-30:, 45:80] = 1008
    seg[-20:, -20:] = 1005
    seg[100:115, 90:110] = 1006
    seg[115:130, 90:110] = 1006
    seg[50:75, 125:160] = 1007

    return seg

def test_matcher(target_seg, match_seg, out_seg):
    rle_matcher = matcher.RLEMatcher(1, 1000, 0.25, 0.25, True)
    shape = target_seg.shape

    # convert target and match to rles
    target_rle = rle.pan_seg_to_rle_seg(target_seg, [1], 1000, [1], False)
    match_rle = rle.pan_seg_to_rle_seg(match_seg, [1], 1000, [1], False)

    # set the target and match
    rle_matcher.initialize_target(target_rle[1])
    match_rle[1] = rle_matcher(match_rle[1], update_target=False)

    assert_equal(rle.rle_seg_to_pan_seg(match_rle, shape), out_seg)
