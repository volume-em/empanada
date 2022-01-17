import cv2
import numpy as np
import torch
from skimage import measure

__all__ = [
    'heatmap_and_offsets'
]

def heatmap_and_offsets(sl2d, heatmap_sigma=6):
    # make sure, the input is numpy
    convert = False
    if type(sl2d) == torch.Tensor:
        sl2d = sl2d.numpy()
        convert = True

    h, w = sl2d.shape
    centers = np.zeros((2, h, w), dtype=np.float32)
    heatmap = np.zeros((h, w), dtype=np.float32)

    # loop over the instance labels and store
    # relevant centers for each
    rp = measure.regionprops(sl2d)
    for r in rp:
        sub_label = r.label
        y, x = r.centroid
        heatmap[int(y), int(x)] = 1
        centers[0, sl2d == sub_label] = y
        centers[1, sl2d == sub_label] = x

    # apply a gaussian filter to spread the centers
    heatmap = cv2.GaussianBlur(heatmap, ksize=(0, 0),
                               sigmaX=heatmap_sigma, sigmaY=heatmap_sigma,
                               borderType=cv2.BORDER_CONSTANT)

    hmax = heatmap.max()
    if hmax > 0:
        heatmap = heatmap / hmax

    # convert from centers to offsets
    yindices = np.arange(0, h, dtype=np.float32)
    xindices = np.arange(0, w, dtype=np.float32)

    # add the y indices to the first channel
    # in the output and x indices to the second channel
    offsets = np.zeros_like(centers)
    offsets[0] = centers[0] - yindices[:, None]
    offsets[1] = centers[1] - xindices[None, :]
    offsets[:, sl2d == 0] = 0

    # add empty dimension to heatmap
    heatmap = heatmap[None] # (1, H, W)

    if convert:
        heatmap = torch.from_numpy(heatmap)
        offsets = torch.from_numpy(offsets)

    return heatmap, offsets
