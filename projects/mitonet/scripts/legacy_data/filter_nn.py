DEFAULT_WEIGHTS = "https://www.dropbox.com/s/2libiwgx0qdgxqv/patch_quality_classifier_nn.pth?raw=1"

import os, sys, argparse, cv2
import json
import numpy as np
from skimage import io
from glob import glob

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.models import resnet34
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Classifies a set of images by fitting a random forest to an array of descriptive features'
    )
    parser.add_argument('segdir', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--imsize', type=int, metavar='imsize', default=224,
                        help='Image size for predictions (assumes square image)')
    parser.add_argument('--confidence_thr', type=float, metavar='confidence_thr', default=0.5,
                        help='Confidence threshold for binary classification')
    parser.add_argument('--keep_thr', type=float, metavar='keep_thr', default=0.1,
                        help='Percent of uninformative patches to keep in segdir')
    args = parser.parse_args()
    
    segdir = args.segdir
    outdir = args.outdir
    imsize = args.imsize
    confidence_thr = args.confidence_thr
    keep_thr = args.keep_thr
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # set up evaluation transforms (assumes imagenet 
    # pretrained as default in train_nn.py)
    normalize = Normalize() # default is imagenet normalization
    eval_tfs = Compose([
        Resize(imsize, imsize),
        normalize,
        ToTensorV2()
    ])

    # create the resnet34 model
    model = resnet34()

    # modify the output layer to predict 1 class only
    model.fc = nn.Linear(in_features=512, out_features=1)

    # load the weights from online
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_WEIGHTS, map_location='cpu')
    msg = model.load_state_dict(state_dict)
    model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    cudnn.benchmark = True

    # make a basic dataset class for loading and 
    # augmenting images WITHOUT any labels
    class SimpleDataset(Dataset):
        def __init__(self, impaths, tfs=None):
            super(SimpleDataset, self).__init__()
            self.impaths = impaths
            self.tfs = tfs

        def __len__(self):
            return len(self.impaths)

        def __getitem__(self, idx):
            # load the image
            fp = self.impaths[idx]
            image = cv2.imread(fp, 0)[..., None]
            image = np.repeat(image, 3, axis=-1)

            # apply transforms
            if self.tfs is not None:
                image = self.tfs(image=image)['image']

            return {'fname': fp, 'image': image}
        
    # pattern created by organize_from_deduplicated.py
    impaths = glob(os.path.join(segdir, '**/images/*.tiff'))

    # create dataset and loader
    tst_data = SimpleDataset(impaths, eval_tfs)
    test = DataLoader(tst_data, batch_size=128, shuffle=False, 
                      pin_memory=True, num_workers=8)

    # run inference on the entire set of unlabeled images
    tst_predictions = []
    for data in tqdm(test, total=len(test)):
        with torch.no_grad():
            # load data onto gpu then forward pass
            images = data['image'].to('cuda:0' if torch.cuda.is_available() else 'cpu', non_blocking=True)
            output = model(images)
            predictions = nn.Sigmoid()(output)

        predictions = predictions.detach().cpu().numpy()
        tst_predictions.append(predictions)

    tst_predictions = np.concatenate(tst_predictions, axis=0)
    tst_predictions = (tst_predictions[:, 0] >= confidence_thr).astype(np.uint8)
    
    # filter out images that are classified as uninformative
    filter_out = np.array(impaths)[np.where(tst_predictions == 0)[0]]

    for imp in tqdm(filter_out):
        # keep image if randomly under threshold
        if np.random.random() < keep_thr:
            continue

        # first get the subdir and fname
        sd = imp.split('/')[-3]
        fname = os.path.basename(imp)
        filtered_subdirs = os.listdir(outdir)

        # load the confidence scores for the given subdir
        #with open(os.path.join(segdir, f'{sd}/confidences.json'), mode='r') as handle:
        #    orig_conf_dict = json.load(handle)

        #conf_value = orig_conf_dict[fname]

        # create subdir for storing the uninformative patches
        if sd not in filtered_subdirs:
            os.makedirs(os.path.join(outdir, sd))
            os.makedirs(os.path.join(outdir, f'{sd}/images'))
            os.makedirs(os.path.join(outdir, f'{sd}/masks'))
            #conf_dict = {fname: conf_value}
        else:
            # load the confidence dict in uninformative dir for appending
            #with open(os.path.join(outdir, f'{sd}/confidences.json'), mode='r') as handle:
            #    conf_dict = json.load(handle)

            #conf_dict[fname] = conf_value
            pass

        # move the uninformative image and its mask
        # then save the confidences with this newly added patch
        os.rename(imp, os.path.join(outdir, f'{sd}/images/{fname}'))
        os.rename(imp.replace('/images/', '/masks/'), os.path.join(outdir, f'{sd}/masks/{fname}'))
        #with open(os.path.join(outdir, f'{sd}/confidences.json'), mode='w') as handle:
        #    json.dump(conf_dict, handle)
