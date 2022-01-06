"""
Description:
------------

This script accepts a directory with image volume files and slices cross sections 
from the given axes (xy, xz, yz). The resultant cross sections are saved in 
the given save directory.

Importantly, the saved image files are given a slightly different filename:
We add '-LOC-{axis}_{slice_index}' to the end of the filename, where axis denotes the
cross-sectioning plane (0->xy, 1->xz, 2->yz) and the slice index is the position of
the cross-section on that axis. Once images from 2d and 3d datasets
start getting mixed together, it can be difficult to keep track of the
provenance of each patch. Everything that appears before '-LOC-' is the
name of the original dataset, the axis and slice index allow us to lookup the
exact location of the cross-section in the volume.

Example usage:
--------------

python cross_section3d.py {imdir} {savedir} --axes 0 1 2 --spacing 1 --processes 4

For help with arguments:
------------------------

python cross_section3d.py --help

"""

import os
import math
import pickle
import argparse
import numpy as np
from glob import glob
from skimage import io
from skimage import measure
from multiprocessing import Pool

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("int16"): 32767,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

def patch_crop(image, mask, crop_size=224, relabel=True):
    if image.ndim == 3:
        if image.shape[2] not in [1, 3]:
            print('Accidentally 3d?', image.shape)
        image = image[..., 0]
        
    # at least 1 image patch
    ysize, xsize = image.shape
    ny = max(1, int(round(ysize / crop_size)))
    nx = max(1, int(round(xsize / crop_size)))
    
    patches = []
    patch_masks = []
    locs = []
    for y in range(ny):
        # start and end indices for y
        ys = y * crop_size
        ye = min(ys + crop_size, ysize)
        for x in range(nx):
            # start and end indices for x
            xs = x * crop_size
            xe = min(xs + crop_size, xsize)
            
            # crop the patch
            patch = image[ys:ye, xs:xe]
            patch_mask = measure.label(mask[ys:ye, xs:xe]).astype(np.uint8)

            patches.append(patch)
            patch_masks.append(patch_mask)
            locs.append(f'{ys}-{ye}_{xs}-{xe}')

    return patches, patch_masks, locs

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('fdir', type=str, metavar='imdir', help='Directory containing 2d image files')
    parser.add_argument('savedir', type=str, metavar='savedir', help='Path to save the patch files')
    parser.add_argument('-cs', '--crop_size', dest='crop_size', type=int, metavar='crop_size', default=512,
                        help='Size of square image patches. Default 512.')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=4,
                        help='Number of processes to run, more processes will run faster but consume more memory')
    

    args = parser.parse_args()

    # read in the parser arguments
    fdir = args.fdir
    savedir = args.savedir
    crop_size = args.crop_size
    processes = args.processes
    
    # check if the savedir exists, if not create it
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    # get the list of all images (png, jpg, tif)
    im_fpaths = sorted(glob(os.path.join(fdir, 'images/*')))
    mk_fpaths = sorted(glob(os.path.join(fdir, 'masks/*')))
    print(f'Found {len(im_fpaths)} images to process')

    def create_patches(*args):
        im_fp, mk_fp = args[0]
        # extract the experiment name from the filepath
        # add a special case for .nii.gz files
        if im_fp[-5:] == 'nii.gz':
            fext = 'nii.gz'
        else:
            fext = im_fp.split('.')[-1]
            
        exp_name = os.path.basename(im_fp).split(f'.{fext}')[0]
        
        # check if results have already been generated,
        # skip this image if so. useful for resuming
        out_path = os.path.join(savedir, exp_name + '.pkl')
        if os.path.isfile(out_path):
            print(f'Already processed {im_fp}, skipping!')
            return
        
        # try to load the image, if it's not possible
        # then pass but print        
        try:
            im = io.imread(im_fp)
            assert (im.min() >= 0), 'Negative images not allowed!'
            msk = io.imread(mk_fp)
        except:
            print('Failed to open: ', im_fp)
            return
        
        if im.dtype != np.uint8:
            dtype = im.dtype
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            im = im.astype(np.float32) / max_value
            im = (im * 255).astype(np.uint8)
        
        patch_dict = {'names': [], 'patches': [], 'patch_masks': []}
                    
        # crop the image into patches
        patches, patch_masks, locs = patch_crop(im, msk, crop_size, relabel=True)

        # appropriate filenames with location
        names = []
        for loc_str in locs:
            # add the -LOC- to indicate the point of separation between
            # the dataset name and the slice location information
            patch_loc_str = f'-LOC-2d-{loc_str}'
            names.append(exp_name + patch_loc_str)
            
        # store results in patch_dict
        patch_dict['names'].extend(names)
        patch_dict['patches'].extend(patches)
        patch_dict['patch_masks'].extend(patch_masks)
                
        out_path = os.path.join(savedir, exp_name + '.pkl')
        with open(out_path, 'wb') as handle:
            pickle.dump(patch_dict, handle)
    
    # running the function with multiple processes
    # results in a much faster runtime
    
    arg_iter = zip(
        im_fpaths,
        mk_fpaths
    )
    
    with Pool(processes) as pool:
        pool.map(create_patches, arg_iter)