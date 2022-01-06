import os
import json
import argparse
import pandas as pd
from shutil import copy, rmtree
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=str, help='Directory containing annotations')
    parser.add_argument('conversion_csv', type=str, help='Subdirectory conversion csv')
    args = parser.parse_args()
    
    src_dir = args.src_dir
    conversion_csv = args.conversion_csv
    assert os.path.isdir(src_dir)
    assert os.path.isfile(conversion_csv)
    
    src_subdirs = set([sd for sd in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, sd))])
    
    convert_df = pd.read_csv(conversion_csv)
    converter = {}
    for i,row in convert_df.iterrows():
        prefix = row['prefix']
        if prefix.startswith('STEM-SD-'):
            prefix = prefix[len('STEM-SD-'):]
        
        dst_sd = prefix.split('-SD-')[0]
        converter[row['random_prefix']] = dst_sd
      
    # for the intersect_subdirs, perform a merge of the confidence dicts
    for sd in src_subdirs:
        with open(os.path.join(src_dir, f'{sd}/confidences.json'), mode='r') as handle:
            src_conf_dict = json.load(handle)
            
        dst_sd = converter[sd]
        os.makedirs(os.path.join(src_dir, dst_sd), exist_ok=True)
        os.makedirs(os.path.join(src_dir, f'{dst_sd}/images'), exist_ok=True)
        os.makedirs(os.path.join(src_dir, f'{dst_sd}/masks'), exist_ok=True)
            
        if os.path.isfile(os.path.join(src_dir, f'{dst_sd}/confidences.json')):
            with open(os.path.join(src_dir, f'{dst_sd}/confidences.json'), mode='r') as handle:
                dst_conf_dict = json.load(handle)
        else:
            dst_conf_dict = {}
            
        with open(os.path.join(src_dir, f'{dst_sd}/confidences.json'), mode='w') as handle:
            json.dump({**dst_conf_dict, **src_conf_dict}, handle, indent=6)

        src_imnames = list(src_conf_dict.keys())
        for src_im in src_imnames:
            os.rename(os.path.join(src_dir, f'{sd}/images/{src_im}'), os.path.join(src_dir, f'{dst_sd}/images/{src_im}'))
            os.rename(os.path.join(src_dir, f'{sd}/masks/{src_im}'), os.path.join(src_dir, f'{dst_sd}/masks/{src_im}'))
            
        #print('Removing:', os.path.join(src_dir, sd))
        rmtree(os.path.join(src_dir, sd))