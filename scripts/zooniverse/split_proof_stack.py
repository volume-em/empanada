import os
import argparse
import numpy as np
import pandas as pd
import pickle
from skimage import io

from skimage import measure

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Path to image stack .tif file')
    parser.add_argument('mask', type=str, help='Path to mask stack .tif file')
    parser.add_argument('attributes', type=str, help='Path to attributes csv file')
    parser.add_argument('save_dir', type=str, help='Directory in which to save results')
    parser.add_argument('--ignore', type=int, nargs='+', help='Image indices to ignore')
    args = parser.parse_args()
    
    # read in the volumes and attributes
    image = io.imread(args.image)
    mask = io.imread(args.mask)
    ignore = args.ignore if args.ignore is not None else []
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'confs'), exist_ok=True)
    
    attrs_df = pd.read_csv(args.attributes)
    
    span = 1 + attrs_df['end'][0] - attrs_df['start'][0]
        
    assert len(attrs_df) == len(image) // span
    assert len(image) == len(mask)
    
    # for each labeled image in flipbook of
    # 5 save it's image and mask with
    # average confidence
    for attr_idx, im_attrs in attrs_df.iterrows():
        fname = '.'.join(im_attrs['image_name'].split('.')[:-1])
        
        # skip images in ignore
        if attr_idx in ignore:
            continue
        
        #assert fname.endswith('_2.jpg')
        #fname = fname.replace('_2.jpg', '.tiff')
        
        start = im_attrs['start']
        midpt = (im_attrs['end'] - im_attrs['start']) // 2
        stack_idx = start + midpt
        
        # get the real dimensions before padding
        if 'height' in im_attrs:
            h = int(im_attrs['height'])
            w = int(im_attrs['width'])
        else:
            # calculate padding from image if missing in csv
            im = image[stack_idx]
            h = np.any(im, axis=1).nonzero()[0][-1] + 1
            w = np.any(im, axis=0).nonzero()[0][-1] + 1
        
        im = image[stack_idx, :h, :w]
        msk = mask[stack_idx, :h, :w]
        
        # only applies to Batch 3a
        #msk = measure.label(msk)
        
        assert msk.max() < 256
        msk = msk.astype(np.uint8)
        
        io.imsave(os.path.join(args.save_dir, f'images/{fname}.tiff'), im, check_contrast=False)
        io.imsave(os.path.join(args.save_dir, f'masks/{fname}.tiff'), msk, check_contrast=False)
        
        conf = int(im_attrs['median_confidence'])
        with open(os.path.join(args.save_dir, f'confs/{fname}.pkl'), mode='wb') as handle:
            pickle.dump(conf, handle)
