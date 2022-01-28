import os
import json
import pickle
import argparse
from shutil import copy
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=str, help='Path containing image, mask, and conf directories from a proofread batch')
    parser.add_argument('dst_dir', type=str, help='Directory for save the panoptic segmentations')
    args = parser.parse_args()
    
    src_dir = args.src_dir
    imdir = os.path.join(src_dir, 'images')
    mkdir = os.path.join(src_dir, 'student_masks')
    cfdir = os.path.join(src_dir, 'confs')
    
    assert os.path.isdir(imdir)
    assert os.path.isdir(mkdir)
    assert os.path.isdir(cfdir)
    
    dst_dir = args.dst_dir
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        dst_subdirs = []
    else:
        dst_subdirs = [sd for sd in os.listdir(dst_dir) if os.path.isdir(os.path.join(dst_dir, sd))]
    
    # glob all image fpaths, masks and confidence
    # files have the same names (except for the file type suffix)
    impaths = sorted(glob(os.path.join(imdir, '*.tiff')))
    mkpaths = sorted(glob(os.path.join(mkdir, '*.tiff')))
    cfpaths = sorted(glob(os.path.join(cfdir, '*.pkl')))
    
    assert len(impaths) == len(mkpaths)
    assert len(mkpaths) == len(cfpaths)
    
    for i in range(len(impaths)):
        imp = impaths[i]
        mkp = mkpaths[i]
        cfp = cfpaths[i]
        
        with open(cfp, mode='rb') as handle:
            conf_score = pickle.load(handle)
        
        assert os.path.basename(imp) == os.path.basename(mkp)
        assert os.path.basename(imp).replace('.tiff', '.pkl') == os.path.basename(cfp)
        
        dataset_name = os.path.basename(imp)
        if '-ROI-' in dataset_name:
            dataset_name = dataset_name.split('-ROI-')[0]
        elif '-LOC-2d-' in dataset_name:
            dataset_name = dataset_name.split('-LOC-2d-')[0]
        elif '-LOC-' in dataset_name:
            dataset_name = dataset_name.split('-LOC-')[0]
        else:
            dataset_name = dataset_name[:len('.tiff')] # remove .tiff
            
            
        dataset_dir = os.path.join(dst_dir, f'{dataset_name}')
        dst_imdir = os.path.join(dataset_dir, 'images')
        dst_mkdir = os.path.join(dataset_dir, 'masks')
        
        if dataset_name in dst_subdirs:
            # merge with existing subdirectory for the dataset
            with open(os.path.join(dataset_dir, 'confidences.json'), mode='r') as handle:
                conf_json = json.load(handle)
        else:
            # create new subdirectory for the dataset
            os.makedirs(dst_imdir)
            os.makedirs(dst_mkdir)
                
            dst_subdirs.append(dataset_name)
            conf_json = {}
            
        conf_json[os.path.basename(imp)] = conf_score
            
        # move the images and masks
        #os.rename(imp, os.path.join(dst_imdir, os.path.basename(imp)))
        #os.rename(mkp, os.path.join(dst_mkdir, os.path.basename(mkp)))
        copy(imp, os.path.join(dst_imdir, os.path.basename(imp)))
        copy(mkp, os.path.join(dst_mkdir, os.path.basename(mkp)))
            
        # save the json
        with open(os.path.join(dataset_dir, 'confidences.json'), mode='w') as handle:
            json.dump(conf_json, handle, indent=6)