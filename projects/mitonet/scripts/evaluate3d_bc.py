import os
import argparse
import zarr
import mlflow
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from empanada.data import VolumeDataset
from empanada import models
from empanada.inference import engines
from empanada.inference.rle import pan_seg_to_rle_seg
from empanada.inference.tracker import InstanceTracker
from empanada.inference.watershed import bc_watershed
from empanada.inference.postprocess import factor_pad
from empanada.zarr_utils import *
from empanada.array_utils import *

from empanada.evaluation import *
from empanada.config_loaders import load_config

archs = sorted(name for name in models.__dict__
    if callable(models.__dict__[name])
)

def parse_args():
    parser = argparse.ArgumentParser(description='Runs panoptic deeplab training')
    parser.add_argument('model_config', type=str, metavar='model_config', help='Path to a model config yaml file')
    parser.add_argument('infer_config', type=str, metavar='infer_config', help='Path to a inference config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_name = args.model_config.split('/')[-1].split('.yaml')[0]
    class_name = 'mito'

    # read the config files
    config = load_config(args.model_config)
    infer_config = load_config(args.infer_config)

    # merge the config files
    config['INFERENCE'] = infer_config

    volume_path = args.volume_path
    weight_path = os.path.join(config['TRAIN']['model_dir'], f'{config_name}_checkpoint.pth.tar')

    # validate parameters
    model_arch = config['MODEL']['arch']
    engine_name = config['INFERENCE']['engine']

    assert model_arch in archs, f"Unrecognized model architecture {model_arch}."
    filter_names = []
    filter_kwargs = []
    if 'filters' in config['INFERENCE']:
        for f in config['INFERENCE']['filters']:
            assert f['name'] in filters.__dict__
            filter_names.append(f['name'])
            del f['name']
            filter_kwargs.append(f)

    # setup model and engine class
    model = models.__dict__[model_arch](**config['MODEL'])
    engine_cls = engines.__dict__[engine_name]

    # load model state
    state = torch.load(weight_path, map_location='cpu')
    state_dict = state['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    msg = model.load_state_dict(state['state_dict'], strict=True)
    model.to('cuda' if torch.cuda.device_count() > 0 else 'cpu') # move model to GPU 0

    # set the evaluation transforms
    norms = state['norms']
    gray_channels = 1
    eval_tfs = A.Compose([
        A.Normalize(**norms),
        ToTensorV2()
    ])

    # load the zarr volume
    data = zarr.open(volume_path, mode='r+')
    volume = data['em']
    shape = volume.shape

    axes = {'xy': 0, 'xz': 1, 'yz': 2}
    axes = {plane: axes[plane] for plane in config['INFERENCE']['axes']}
    coeff = 255 // len(axes)

    axis_stacks = {}
    for axis_name, axis in axes.items():
        print(f'Predicting {axis_name} stack')

        # create the inference engine
        inference_engine = engine_cls(model, **config['INFERENCE']['engine_params'])

        # make axis-specific dataset
        dataset = VolumeDataset(volume, axis, eval_tfs, scale=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                pin_memory=True, drop_last=False, num_workers=8)

        stack = np.zeros((2, *shape), dtype=np.uint8)

        fill_index = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            h, w = image.size()[-2:]

            image = factor_pad(image, 128)
            bc_seg = inference_engine(image) # (1, 2, H, W)

            if bc_seg is None:
                continue
            else:
                # remove padding and unit dimensions
                bc_seg = bc_seg[0, :, :h, :w].cpu().numpy() # (2, H, W)
                bc_seg = (coeff * bc_seg).astype(np.uint8)

                # store the seg and contours
                put(stack, fill_index, bc_seg, axis+1)
                fill_index += 1

        final_segs = inference_engine.end()
        if final_segs:
            for i, bc_seg in enumerate(final_segs):
                # remove padding and unit dimensions
                bc_seg = bc_seg[0, :, :h, :w].cpu().numpy() # (2, H, W)
                bc_seg = (coeff * bc_seg).astype(np.uint8)

                # store the seg and contours
                put(stack, fill_index, bc_seg, axis+1)
                fill_index += 1

        # store the segmentation stack
        axis_stacks[axis_name] = stack

    # add the axis stacks together
    print('Summing stacks...')
    orthostack = sum(list(axis_stacks.values()))

    # store the segmentation
    data.create_dataset(
        f'{config_name}_{class_name}_bc_pred', data=orthostack,
        overwrite=True, chunks=(1, None, None)
    )

    # apply bc watershed
    print('Running watershed...')
    instance_seg = bc_watershed(orthostack, **config['INFERENCE']['watershed_params'])

    data.create_dataset(
        f'{config_name}_{class_name}_pred', data=instance_seg,
        overwrite=True, chunks=(1, None, None)
    )

    # only supports a single class
    print('Running tracking...')
    label_divisor = config['INFERENCE']['watershed_params']['label_divisor']
    pred_tracker = InstanceTracker(1, label_divisor, shape, 'xy')
    for index2d,seg2d in tqdm(enumerate(instance_seg), total=len(instance_seg)):
        rle_seg = pan_seg_to_rle_seg(seg2d, [1], label_divisor, [1], force_connected=False)
        pred_tracker.update(rle_seg[1], index2d)

    pred_tracker.finish()
    pred_tracker.write_to_json(os.path.join(volume_path, f'{config_name}_{class_name}_pred.json'))

    # run evaluation
    semantic_metrics = {'IoU': iou}
    instance_metrics = {'F1_50': f1_50, 'F1_75': f1_75, 'Precision_50': precision_50,
                        'Precision_75': precision_75, 'Recall_50': recall_50, 'Recall_75': recall_75}
    panoptic_metrics = {'PQ': panoptic_quality}
    evaluator = Evaluator(semantic_metrics, instance_metrics, panoptic_metrics)

    gt_json = os.path.join(volume_path, f'{class_name}_gt.json')
    pred_json = os.path.join(volume_path, f'{config_name}_{class_name}_pred.json')
    results = evaluator(gt_json, pred_json)
    results = {f'{class_name}_{k}': v for k,v in results.items()}

    for k, v in results.items():
        print(k, v)
