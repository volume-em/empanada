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
import imagehash
import numpy as np
import SimpleITK as sitk
from skimage import measure
from glob import glob
from PIL import Image
from skimage import io
from multiprocessing import Pool

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("int16"): 32767,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

def calculate_hash(image, crop_size, hash_size=8):
    # calculate the hash on the resized image
    imsize = (crop_size, crop_size)
    pil_image = Image.fromarray(image).resize(imsize, resample=2)
    
    return imagehash.dhash(pil_image, hash_size=hash_size).hash

def patch_and_hash(image, mask, crop_size=512, hash_size=8, relabel=True):
    
    if image.ndim == 3:
        print('Accidentally 3d?', image.shape)
        image = image[..., 0]
        
    # at least 1 image patch
    ysize, xsize = image.shape
    ny = max(1, int(round(ysize / crop_size)))
    nx = max(1, int(round(xsize / crop_size)))
    
    patches = []
    patch_masks = []
    hashes = []
    locs = []
    for y in range(ny):
        # start and end indices for y
        ys = y * crop_size
        ye = min(ys + crop_size, ysize)
        for x in range(nx):
            # start and end indices for x
            xs = x * crop_size
            xe = min(xs + crop_size, xsize)
            
            # crop the patch and calculate its hash
            patch = image[ys:ye, xs:xe]
            patch_hash = calculate_hash(patch, crop_size, hash_size)
            
            patch_mask = mask[ys:ye, xs:xe]
            if relabel:
                patch_mask = measure.label(patch_mask).astype(np.uint8)
                assert patch_mask.max() != 255 # num objects should stay within 8-bit

            patches.append(patch)
            patch_masks.append(patch_mask)
            hashes.append(patch_hash)
            locs.append(f'{ys}-{ye}_{xs}-{xe}')

    return patches, patch_masks, hashes, locs

def deduplicate(patch_dict, min_distance):
    # all hashes are the same size
    hashes = np.array(patch_dict['hashes'])
    hashes = hashes.reshape(len(hashes), -1)

    # randomly permute the hashes such that we'll have random ordering
    random_indices = np.random.permutation(np.arange(0,  len(hashes)))
    hashes = hashes[random_indices]

    # loop through the hashes and assign images to sets of near duplicates
    # until all of the hashes are exhausted
    exemplars = []
    while len(hashes) > 0:
        ref_hash = hashes[0]

        # a match has Hamming distance less than min_distance
        matches = np.where(
            np.logical_xor(ref_hash, hashes).sum(1) <= min_distance
        )[0]

        # ref_hash is the exemplar (i.e. first in matches)
        exemplars.append(random_indices[matches[0]])

        # remove all the matched images from both hashes and indices
        hashes = np.delete(hashes, matches, axis=0)
        random_indices = np.delete(random_indices, matches, axis=0)
        
    # keep only the exemplars
    names = []
    patches = []
    patch_masks = []
    for index in exemplars:
        names.append(patch_dict['names'][index])
        patches.append(patch_dict['patches'][index])
        patch_masks.append(patch_dict['patch_masks'][index])
        
    return {'names': names, 'patches': patches, 'patch_masks': patch_masks}
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Create dataset for nn experimentation')
    parser.add_argument('fdir', type=str, metavar='fdir', help='Directory containing image and mask volume directories')
    parser.add_argument('savedir', type=str, metavar='savedir', help='Path to save deduplicated datasets')
    parser.add_argument('-a', '--axes', dest='axes', type=int, metavar='axes', nargs='+', default=[0, 1, 2],
                        help='Volume axes along which to slice (0-xy, 1-xz, 2-yz)')
    parser.add_argument('-s', '--spacing', dest='spacing', type=int, metavar='spacing', default=1,
                        help='Spacing between image slices')
    parser.add_argument('-cs', '--crop_size', dest='crop_size', type=int, metavar='crop_size', default=512,
                        help='Size of square image patches. Default 512.')
    parser.add_argument('-hs', '--hash_size', dest='hash_size', type=int, metavar='hash_size', default=8,
                        help='Size of the image hash. Default 8 (assumes crop size of 224).')
    parser.add_argument('-d', '--min_distance', dest='min_distance', type=int, metavar='min_distance', default=12,
                        help='Minimum Hamming distance between hashes to be considered unique. Default 12 (assumes hash size of 8)')
    parser.add_argument('-p', '--processes', dest='processes', type=int, metavar='processes', default=4,
                        help='Number of processes to run, more processes will run faster but consume more memory')
    

    args = parser.parse_args()

    # read in the parser arguments
    fdir = args.fdir
    savedir = args.savedir
    axes = args.axes
    spacing = args.spacing
    crop_size = args.crop_size
    hash_size = args.hash_size
    min_distance = args.min_distance
    processes = args.processes
    
    # check if the savedir exists, if not create it
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    # get the list of all volumes (mrc, tif, nrrd, nii.gz, etc.)
    im_fpaths = sorted(glob(os.path.join(fdir, 'images/*')))
    msk_fpaths = sorted(glob(os.path.join(fdir, 'masks/*')))
    print(f'Found {len(im_fpaths)} image volumes to process')
    
    for imf, mkf in zip(im_fpaths, msk_fpaths):
        assert os.path.basename(imf) == os.path.basename(mkf)

    def create_slices(*args):
        im_fp, mk_fp = args[0]
        # extract the experiment name from the filepath
        # add a special case for .nii.gz files
        if im_fp[-5:] == 'nii.gz':
            fext = 'nii.gz'
        else:
            fext = im_fp.split('.')[-1]
            
        exp_name = os.path.basename(im_fp).split(f'.{fext}')[0]
        
        # check if results have already been generated
        # skip this volume, if so. useful for resuming
        out_path = os.path.join(savedir, exp_name + '.pkl')
        if os.path.isfile(out_path):
            print(f'Already processed {im_fp}, skipping!')
            return
        
        # try to load the volume, if it's not possible
        # then pass but print        
        try:
            im = sitk.ReadImage(im_fp)
            msk = sitk.ReadImage(mk_fp)
            
            if len(im.GetSize()) > 3:
                raise Exception
            
            print(im.GetSize(), im_fp)
        except:
            print('Failed to open: ', im_fp)
            pass
        
        # extract the pixel size from the volume
        # if the z-pixel size is more than 25% different
        # from the x-pixel size, don't slice over orthogonal
        # directions
        pixel_sizes = im.GetSpacing()
        anisotropy = np.abs(pixel_sizes[0] - pixel_sizes[2]) / pixel_sizes[0]
        
        im = sitk.GetArrayFromImage(im)
        msk = sitk.GetArrayFromImage(msk)
        assert (im.min() >= 0), 'Negative images not allowed!'
        
        if im.dtype != np.uint8:
            dtype = im.dtype
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            im = im.astype(np.float32) / max_value
            im = (im * 255).astype(np.uint8)
        
        patch_dict = {'names': [], 'patches': [], 'patch_masks': [], 'hashes': []}
        for axis in axes:
            # only process xy slices if the volume is anisotropic
            if (anisotropy > 0.25 or 'video' in exp_name) and (axis != 0):
                continue
                
            # evenly spaced slices
            nmax = im.shape[axis] - 1
            slice_indices = np.arange(0, nmax, spacing, dtype='int')
            zpad = math.ceil(math.log(nmax, 10))
            
            for idx in slice_indices:
                # slice the volume on the proper axis
                if axis == 0:
                    im_slice = im[idx]
                    msk_slice = msk[idx]
                elif axis == 1:
                    im_slice = im[:, idx]
                    msk_slice = msk[:, idx]
                else:
                    im_slice = im[:, :, idx]
                    msk_slice = msk[:, :, idx]
                    
                # crop the image into patches
                patches, patch_masks, hashes, locs = patch_and_hash(im_slice, msk_slice, crop_size, hash_size, relabel=True)

                # appropriate filenames with location
                names = []
                for loc_str in locs:
                    # add the -LOC- to indicate the point of separation between
                    # the dataset name and the slice location information
                    index_str = str(idx).zfill(zpad)
                    patch_loc_str = f'-LOC-{axis}_{index_str}_{loc_str}'
                    names.append(exp_name + patch_loc_str)

                # store results in patch_dict
                patch_dict['names'].extend(names)
                patch_dict['patches'].extend(patches)
                patch_dict['patch_masks'].extend(patch_masks)
                patch_dict['hashes'].extend(hashes)
                
        # remove duplicate patches
        patch_dict = deduplicate(patch_dict, min_distance)
                
        out_path = os.path.join(savedir, exp_name + '.pkl')
        with open(out_path, 'wb') as handle:
            pickle.dump(patch_dict, handle)
    
    # running the function with multiple processes
    # results in a much faster runtime
    arg_iter = zip(
        im_fpaths,
        msk_fpaths
    )
    
    with Pool(processes) as pool:
        pool.map(create_slices, arg_iter)