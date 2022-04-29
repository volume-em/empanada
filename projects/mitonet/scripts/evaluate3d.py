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
from empanada.inference import filters
from empanada.inference.postprocess import factor_pad
from empanada.config_loaders import load_config
from empanada.inference.patterns import *
from empanada.evaluation import *

archs = sorted(name for name in models.__dict__
    if callable(models.__dict__[name])
)

def parse_args():
    parser = argparse.ArgumentParser(description='Runs panoptic deeplab training')
    parser.add_argument('model_config', type=str, metavar='model_config', help='Path to a model config yaml file')
    parser.add_argument('infer_config', type=str, metavar='infer_config', help='Path to a inference config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    parser.add_argument('--save-panoptic', action='store_true', help='Whether to save raw panoptic segmentation for each stack.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_name = os.path.basename(args.model_config).split('.yaml')[0]

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

    filters_dict = config['INFERENCE'].get('filters')
    if filters_dict is not None:
        for f in config['INFERENCE']['filters']:
            assert f['name'] in filters.__dict__

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
    model.to('cuda' if torch.cuda.is_available() else 'cpu') # move model to GPU 0

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
    class_labels = config['INFERENCE']['labels']
    thing_list = config['INFERENCE']['engine_params']['thing_list']
    label_divisor = config['INFERENCE']['engine_params']['label_divisor']

    # create a separate tracker for
    # each prediction axis and each segmentation class
    trackers = create_axis_trackers(axes, class_labels, label_divisor, shape)

    for axis_name, axis in axes.items():
        print(f'Predicting {axis_name} stack')

        # create placeholder volume for stack results, if desired
        if args.save_panoptic:
            # chunk in axis direction only
            chunks = [None, None, None]
            chunks[axis] = 1
            chunks = tuple(chunks)
            stack = data.create_dataset(
                f'{config_name}_panoptic_{axis_name}', shape=shape,
                dtype=np.uint32, chunks=chunks, overwrite=True
            )
        else:
            stack = None

        # create the inference engine
        inference_engine = engine_cls(model, **config['INFERENCE']['engine_params'])

        # create a separate matcher for each thing class
        matchers = create_matchers(thing_list, label_divisor, **config['INFERENCE']['matcher_params'])

        # setup matcher for multiprocessing
        queue = mp.Queue()
        rle_stack = []
        matcher_out, matcher_in = mp.Pipe()
        matcher_args = (
            matchers, queue, rle_stack, matcher_in,
            class_labels, label_divisor, thing_list
        )
        matcher_proc = mp.Process(target=forward_matching, args=matcher_args)
        matcher_proc.start()

        # make axis-specific dataset
        dataset = VolumeDataset(volume, axis, eval_tfs, scale=1)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            pin_memory=True, drop_last=False, num_workers=8
        )

        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            h, w = image.size()[-2:]

            image = factor_pad(image, 128)
            pan_seg = inference_engine(image)

            if pan_seg is None:
                # building the queue
                queue.put(None)
                continue
            else:
                pan_seg = pan_seg.squeeze()[:h, :w].cpu().numpy()
                queue.put(pan_seg)

        final_segs = inference_engine.end()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                pan_seg = pan_seg.squeeze()[:h, :w].cpu().numpy()
                queue.put(pan_seg)

        # finish and close forward matching process
        queue.put('finish')
        rle_stack = matcher_out.recv()[0]
        matcher_proc.join()

        print(f'Propagating labels backward through the stack...')
        for index,rle_seg in tqdm(backward_matching(rle_stack, matchers, shape[axis]), total=shape[axis]):
            update_trackers(rle_seg, index, trackers[axis_name], axis, stack)

        finish_tracking(trackers[axis_name])
        for tracker in trackers[axis_name]:
            apply_filters(tracker, filters_dict)

    # create the final instance segmentations
    for class_id in config['INFERENCE']['labels']:
        class_name = config['DATASET']['class_names'][class_id]

        print(f'Creating consensus segmentation for class {class_name}...')
        class_trackers = get_axis_trackers_by_class(trackers, class_id)

        # merge instances from orthoplane inference if applicable
        if len(class_trackers) > 1:
            if class_id in thing_list:
                consensus_tracker = create_instance_consensus(class_trackers, **config['INFERENCE']['consensus_params'])
                apply_filters(consensus_tracker, filters_dict)
            else:
                consensus_tracker = create_semantic_consensus(class_trackers, config['INFERENCE']['consensus_params']['pixel_vote_thr'])
        else:
            consensus_tracker = class_trackers[0]

        dtype = np.uint32 if class_id in thing_list else np.uint8

        # decode and fill the instances
        consensus_vol = data.create_dataset(
            f'{config_name}_{class_name}_pred', shape=shape, dtype=dtype,
            overwrite=True, chunks=(1, None, None)
        )
        fill_volume(consensus_vol, consensus_tracker.instances, processes=4)
        consensus_tracker.write_to_json(os.path.join(volume_path, f'{config_name}_{class_name}_pred.json'))

    # run evaluation
    semantic_metrics = {'IoU': iou}
    instance_metrics = {'F1_50': f1_50, 'F1_75': f1_75, 'Precision_50': precision_50,
                        'Precision_75': precision_75, 'Recall_50': recall_50, 'Recall_75': recall_75}
    panoptic_metrics = {'PQ': panoptic_quality}
    evaluator = Evaluator(semantic_metrics, instance_metrics, panoptic_metrics)

    for class_id, class_name in config['DATASET']['class_names'].items():
        gt_json = os.path.join(volume_path, f'{class_name}_gt.json')
        pred_json = os.path.join(volume_path, f'{config_name}_{class_name}_pred.json')
        results = evaluator(gt_json, pred_json)
        results = {f'{class_name}_{k}': v for k,v in results.items()}

        for k, v in results.items():
            print(k, v)

        try:
            run_id = state.get('run_id')
            if run_id is not None:
                volname = os.path.basename(volume_path).split('.zarr')[0][:20]
                with mlflow.start_run(run_id=run_id) as run:
                    for k, v in results.items():
                        mlflow.log_metric(f'{volname}_{k}', v, step=0)
        except:
            print('Results not stored in MLFlow, script was run in the wrong directory.')
