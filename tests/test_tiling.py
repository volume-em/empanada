import pytest
import torch
import numpy as np
from numpy.testing import assert_almost_equal
from empanada import consensus, metrics
from empanada.inference import rle, tile
from skimage.morphology import disk
from skimage import measure

@pytest.fixture
def instance_seg():
    shape = (400, 400)
    circle = disk(20).astype(np.uint32)

    seg_out = np.zeros(shape, dtype=np.uint32)
    c = 1001
    for xs in range(0, 351, 50):
        for ys in range(0, 351, 50):    
            ye = ys + 41
            xe = xs + 41
            seg_out[ys:ye, xs:xe][circle > 0] = circle[circle > 0] * c
            c += 1
    
    return seg_out

def test_tiling(instance_seg):
    # create the tiler
    tiler = tile.Tiler(instance_seg.shape, (100, 110), 20)
    
    # tile the instance segmentation
    # and relabel objects
    tile_segs = []
    for idx in range(len(tiler)):
        tile_seg = measure.label(tiler(instance_seg, idx)).astype(np.uint32)
        tile_seg[tile_seg > 0] += 1000
        tile_segs.append(tile_seg)
        
    # convert the segmentations to run length encodings
    rle_segs = []
    for i,ts in enumerate(tile_segs):
        # create and translate the rle seg
        rle_seg = rle.pan_seg_to_rle_seg(ts, [1], 1000, [1], False)
        rle_seg = tiler.translate_rle_seg(rle_seg, i)
        rle_segs.append(rle_seg)
        
    # merge the tiles without overlap mask
    tiled_rle_seg = consensus.merge_objects_from_tiles([rs[1] for rs in rle_segs])
    tiled_instance_seg = rle.rle_seg_to_pan_seg({1: tiled_rle_seg}, instance_seg.shape)
    
    # without matching instance ids, we can check the tiled
    # segmentation is equivalent by measuring the F1 scores
    instance_seg = torch.from_numpy(instance_seg.astype('int'))[None, None]
    tiled_instance_seg = torch.from_numpy(tiled_instance_seg.astype('int'))[None, None]
    
    f1_cls = metrics.F1(metrics.AverageMeter, [1], 1000, 0.5, 'pan_seg', 'pan_seg')
    f1_dict = f1_cls.calculate({'pan_seg': instance_seg}, {'pan_seg': tiled_instance_seg})
    for i, (l,v) in enumerate(f1_dict.items()):
        assert_almost_equal(float(v), 1., decimal=3)