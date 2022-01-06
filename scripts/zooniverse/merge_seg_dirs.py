import os
import json
import argparse
from shutil import copy
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=str, help='Path containing image, mask, and conf directories from a proofread batch')
    parser.add_argument('dst_dir', type=str, help='Directory for save the panoptic segmentations')
    args = parser.parse_args()
    
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    assert os.path.isdir(dst_dir)
    assert os.path.isdir(src_dir)
    
    src_subdirs = set([sd for sd in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, sd))])
    dst_subdirs = set([sd for sd in os.listdir(dst_dir) if os.path.isdir(os.path.join(dst_dir, sd))])
    
    intersect_subdirs = src_subdirs.intersection(dst_subdirs)
    setdiff_subdirs = src_subdirs - intersect_subdirs
    
    # for the setdiff_subdirs, just perform a rename
    for sd in setdiff_subdirs:
        #print('setdiff rename:', os.path.join(src_dir, sd), os.path.join(dst_dir, sd))
        os.rename(os.path.join(src_dir, sd), os.path.join(dst_dir, sd))
      
    # for the intersect_subdirs, perform a merge of the confidence dicts
    for sd in intersect_subdirs:
        with open(os.path.join(src_dir, sd), mode='r') as handle:
            src_conf_dict = json.load(handle)
            
        with open(os.path.join(dst_dir, sd), mode='r') as handle:
            dst_conf_dict = json.load(handle)
            
        with open(os.path.join(dst_dir, sd), mode='w') as handle:
            json.dump({**dst_conf_dict, **src_conf_dict}, handle, indent=6)

        src_imnames = list(src_conf_dict.keys())
        for src_im in src_imnames:
            os.rename(os.path.join(src_dir, f'{sd}/images/{src_im}'), os.path.join(dst_dir, f'{sd}/images/{src_im}'))
            os.rename(os.path.join(src_dir, f'{sd}/masks/{src_im}'), os.path.join(dst_dir, f'{sd}/masks/{src_im}'))