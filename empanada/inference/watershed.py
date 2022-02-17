import numpy as np
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import dilation
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

__all__ = ['bc_watershed']

def bc_watershed(
    volume, 
    thres1=0.9, 
    thres2=0.8, 
    thres3=0.85, 
    seed_thres=32,
    min_size=128,
    label_divisor=1000
):
    r"""
    Copied from: https://github.com/zudi-lin/pytorch_connectomics/blob/b6457ea4bc7d9b01ef3a00781dff252ab5d4a1d3/connectomics/utils/process.py
    
    Convert binary foreground probability maps and instance contours to 
    instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_ 
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args: 
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        seed_thr (int): minimum size of seed in voxels. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
    """
    
    assert volume.shape[0] == 2
    semantic = volume[0]
    boundary = volume[1]
    seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2))
    foreground = (semantic > int(255*thres3))
    seed = label(seed_map)
    seed = remove_small_objects(seed, seed_thres)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_objects(segm, min_size)
    segm[segm > 0] += label_divisor
        
    return cast2dtype(segm)

def cast2dtype(segm):
    """Cast the segmentation mask to the best dtype to save storage.
    """
    mid = np.max(segm)

    m_type = np.uint64
    if mid < 2**8:
        m_type = np.uint8
    elif mid < 2**16:
        m_type = np.uint16
    elif mid < 2**32:
        m_type = np.uint32
    
    return segm.astype(m_type)