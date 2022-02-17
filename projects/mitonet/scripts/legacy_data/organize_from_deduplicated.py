import os, sys, cv2, argparse
import pickle
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dedupe_dir', type=str)
    parser.add_argument('savedir', type=str)
    args = parser.parse_args()
    
    # parse the arguments
    dedupe_dir = args.dedupe_dir
    savedir = args.savedir
    
    # make sure the savedir exists
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # list all pkl deduplicated files
    fpaths = glob(os.path.join(dedupe_dir, 'jrc*.pkl'))
    
    for fp in tqdm(fpaths):
        dataset_name = os.path.basename(fp)
        if '-ROI-' in dataset_name:
            dataset_name = dataset_name.split('-ROI-')[0]
        elif '-LOC-2d-' in dataset_name:
            dataset_name = dataset_name.split('-LOC-2d-')[0]
        elif '-LOC-' in dataset_name:
            dataset_name = dataset_name.split('-LOC-')[0]
        else:
            dataset_name = dataset_name[:-len('.pkl')]
        
        dataset_savedir = os.path.join(savedir, dataset_name)
        if not os.path.exists(dataset_savedir):
            os.mkdir(dataset_savedir)
            os.mkdir(os.path.join(dataset_savedir, 'images'))
            os.mkdir(os.path.join(dataset_savedir, 'masks'))
            
        # load the patches_dict
        with open(fp, mode='rb') as handle:
            patches_dict = pickle.load(handle)
            
        conf_dict = None
        if 'confidences.json' in os.listdir(dataset_savedir):
            with open(os.path.join(dataset_savedir, 'confidences.json'), mode='r') as handle:
                conf_dict = json.load(handle)
            
        for fn, img, msk in zip(patches_dict['names'], patches_dict['patches'], patches_dict['patch_masks']):
            if os.path.exists(os.path.join(dataset_savedir, f'images/{fn}.tiff')):
                continue
            else:
                io.imsave(os.path.join(dataset_savedir, f'images/{fn}.tiff'), img, check_contrast=False)
                io.imsave(os.path.join(dataset_savedir, f'masks/{fn}.tiff'), msk, check_contrast=False)

                if conf_dict is not None:
                    conf_dict[fn] = 1
            
        if conf_dict is not None:
            with open(os.path.join(dataset_savedir, 'confidences.json'), mode='w') as handle:
                json.dump(conf_dict, handle, indent=6) 
