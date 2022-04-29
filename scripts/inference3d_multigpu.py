"""
TODO:
Add confidence prediction
"""
import os
import sys
import argparse
import yaml
import zarr
import mlflow
import numpy as np
import torch
import torch.nn as nn
from skimage import io
from skimage import measure
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from empanada import models
from empanada.inference import engines
from empanada.inference.matcher import SequentialMatcher
from empanada.inference.tracker import InstanceTracker
from empanada.inference import filters
from empanada.inference.array_utils import *
from empanada.zarr_utils import *
from empanada.aggregation.consensus import merge_objects3d
from empanada.evaluation import *

from config_utils import load_train_config, load_inference_config
from sampler import DistributedEvalSampler

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pickle
import functools

from collections import deque

from time import time

from empanada.inference.postprocess import merge_semantic_and_instance

archs = sorted(name for name in models.__dict__
    if callable(models.__dict__[name])
)

def factor_pad_tensor(tensor, factor=128):
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    else:
        return nn.ZeroPad2d((0, pad_right, 0, pad_bottom))(tensor)

def parse_args():
    parser = argparse.ArgumentParser(description='Runs panoptic deeplab training')
    parser.add_argument('model_config', type=str, metavar='model_config', help='Path to a model config yaml file')
    parser.add_argument('infer_config', type=str, metavar='infer_config', help='Path to a inference config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    return parser.parse_args()

def snakemake_args():
    params = vars(snakemake.params)
    params['model_config'] = snakemake.input[0]
    params['infer_config'] = snakemake.input[1]
    params['volume_path'] = snakemake.input[2]
    del params['_names']

    return argparse.Namespace(**params)

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    # all tensors are same size
    world_size = dist.get_world_size()
    max_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in range(world_size)
    ]

    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for tensor in tensor_list:
        buffer = tensor.cpu().numpy().tobytes()
        data_list.append(pickle.loads(buffer))

    return data_list

def get_panoptic_seg(sem, instance_cells, config):
    label_divisor = config['INFERENCE']['engine_params']['label_divisor']
    thing_list = config['INFERENCE']['engine_params']['thing_list']
    stuff_area = config['INFERENCE']['engine_params']['stuff_area']
    void_label = config['INFERENCE']['engine_params']['void_label']

    # keep only label for instance classes
    instance_seg = torch.zeros_like(sem)
    for thing_class in thing_list:
        instance_seg[sem == thing_class] = 1

    # map object ids
    instance_seg = (instance_seg * instance_cells[None]).long()

    pan_seg = merge_semantic_and_instance(
        sem, instance_seg, label_divisor, thing_list,
        stuff_area, void_label
    )

    return pan_seg

def run_forward_matchers(stack, axis, matchers, queue, config):
    # create the deques for sem and instance cells\
    confidence_thr = config['INFERENCE']['engine_params']['confidence_thr']
    median_kernel_size = config['INFERENCE']['engine_params']['median_kernel_size']
    mid_idx = (median_kernel_size - 1) // 2

    fill_index = 0
    sem_queue = deque(maxlen=median_kernel_size)
    cell_queue = deque(maxlen=median_kernel_size)

    while True:
        sem, cells = queue.get()

        if sem is None:
            break

        sem_queue.append(sem)
        cell_queue.append(cells)

        nq = len(sem_queue)
        if nq <= mid_idx:
            # take last item in the queue
            median_sem = sem_queue[-1]
            cells = cell_queue[-1]
        elif nq > mid_idx and nq < median_kernel_size:
            # continue while the queue builds
            median_sem = None
        else:
            # nq == median_kernel_size
            # use the middle item in the queue
            # with the median segmentation probs
            median_sem = torch.median(
                torch.cat(list(sem_queue), dim=0), dim=0, keepdim=True
            ).values
            cells = cell_queue[mid_idx]

        # harden the segmentation to (N, 1, H, W)
        if median_sem is not None:
            if median_sem.size(1) > 1: # multiclass segmentation
                median_sem = torch.argmax(median_sem, dim=1, keepdim=True)
            else:
                median_sem = (median_sem >= confidence_thr).long() # need integers

            pan_seg = get_panoptic_seg(median_sem, cells, config)
        else:
            pan_seg = None

        if pan_seg is None:
            continue
        else:
            pan_seg = pan_seg.squeeze().numpy()
            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)

            zarr_put3d(stack, fill_index, pan_seg, axis)
            fill_index += 1

    # fill out the final segmentations
    for sem,cells in zip(list(sem_queue)[mid_idx + 1:], list(cell_queue)[mid_idx + 1:]):
        if sem.size(1) > 1: # multiclass segmentation
            sem = torch.argmax(sem, dim=1, keepdim=True)
        else:
            sem = (sem >= confidence_thr).long() # need integers

        pan_seg = get_panoptic_seg(sem, cells, config)
        pan_seg = pan_seg.squeeze().numpy()

        for matcher in matchers:
            if matcher.target_seg is None:
                pan_seg = matcher.initialize_target(pan_seg)
            else:
                pan_seg = matcher(pan_seg)

        zarr_put3d(stack, fill_index, pan_seg, axis)
        fill_index += 1

    return None


def main_worker(gpu, config):
    config['gpu'] = gpu
    rank = gpu

    dist.init_process_group(backend=config['TRAIN']['dist_backend'], init_method=config['TRAIN']['dist_url'],
                            world_size=config['world_size'], rank=rank)

    model_arch = config['MODEL']['arch']
    config_name = config['config_name']
    weight_path = os.path.join(config['TRAIN']['model_dir'], f'{config_name}_checkpoint.pth.tar')

    # setup model and engine class
    model = models.__dict__[model_arch](**config['MODEL'])
    engine_cls = engines.MultiGPUInferenceEngine

    # load model state
    state = torch.load(weight_path, map_location='cpu')
    state_dict = state['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    msg = model.load_state_dict(state['state_dict'], strict=True)

    torch.cuda.set_device(config['gpu'])
    model.cuda(config['gpu'])

    config['TRAIN']['batch_size'] = 1
    config['TRAIN']['workers'] = 0
    model = DDP(model, device_ids=[config['gpu']])

    eval_tfs = A.Compose([
        A.Normalize(**state['norms']),
        ToTensorV2()
    ])

    axis, axis_name = 0, 'xy'
    data = zarr.open(config['volume_path'], mode='r+')
    volume = data['em']
    shape = volume.shape
    dataset = ZarrData(volume, 0, eval_tfs)
    sampler = DistributedEvalSampler(dataset)

    dataloader = DataLoader(
        dataset, batch_size=config['TRAIN']['batch_size'], shuffle=False,
        pin_memory=True, drop_last=False, num_workers=config['TRAIN']['workers'],
        sampler=sampler
    )

    # if in main process, create zarr to store results
    # chunk in axis direction only
    if rank == 0:
        chunks = [None, None, None]
        chunks[axis] = 1
        chunks = tuple(chunks)
        stack = data.create_dataset(f'{config_name}_panoptic_{axis_name}', shape=shape,
                                    dtype=np.uint64, chunks=chunks,
                                    overwrite=True)

        class_labels = config['INFERENCE']['labels']
        thing_list = config['INFERENCE']['engine_params']['thing_list']
        label_divisor = config['INFERENCE']['engine_params']['label_divisor']
        matchers = [
            SequentialMatcher(thing_class, label_divisor, **config['INFERENCE']['matcher_params'])
            for thing_class in thing_list
        ]

        queue = mp.Queue()
        matcher_proc = mp.Process(target=run_forward_matchers, args=(stack, axis, matchers, queue, config))
        matcher_proc.start()

    # create the inference engine
    inference_engine = engine_cls(model, **config['INFERENCE']['engine_params'])

    iterator = dataloader if rank != 0 else tqdm(dataloader, total=len(dataloader))
    for batch in iterator:
        index = batch['index']
        image = batch['image']
        h, w = image.size()[2:]
        image = factor_pad_tensor(image, 128)

        output = inference_engine.infer(image)
        sem = output['sem']
        instance_cells = inference_engine.get_instance_cells(output['ctr_hmp'], output['offsets'])

        # get median semantic seg
        sems = all_gather(sem)
        instance_cells = all_gather(instance_cells)

        # move both sem and instance_cells to cpu
        sems = [sem.cpu() for sem in sems]
        instance_cells = [cells.cpu() for cells in instance_cells]

        if rank == 0:
            for sem, cells in zip(sems, instance_cells):
                queue.put(
                    (sem[..., :h, :w], cells[..., :h, :w])
                )

    # pass None to queue to mark the end of inference
    queue.put((None, None))
    matcher_proc.join()

if __name__ == "__main__":
    if 'snakemake' in globals():
        args = snakemake_args()
    else:
        args = parse_args()

    config_name = args.model_config.split('/')[-1].split('.yaml')[0]

    # read the config files
    config = load_train_config(args.model_config)
    infer_config = load_inference_config(args.infer_config)

    # merge the config files
    config['INFERENCE'] = infer_config
    config['config_name'] = config_name

    volume_path = args.volume_path
    config['volume_path'] = volume_path

    # validate parameters
    model_arch = config['MODEL']['arch']
    assert model_arch in archs, f"Unrecognized model architecture {model_arch}."
    filter_names = []
    filter_kwargs = []
    if 'filters' in config['INFERENCE']:
        for f in config['INFERENCE']['filters']:
            assert f['name'] in filters.__dict__
            filter_names.append(f['name'])
            del f['name']
            filter_kwargs.append(f)

    # load the zarr volume
    #data = zarr.open(volume_path, mode='r+')
    #volume = data['em']
    #shape = volume.shape

    # TODO: ANISOTROPY OPTIONS
    #axes = {'xy': 0, 'xz': 1, 'yz': 2}
    #axes = {plane: axes[plane] for plane in config['INFERENCE']['axes']}

    # create a separate tracker for
    # each prediction axis and each segmentation class
    #trackers = {}
    #class_labels = config['INFERENCE']['labels']
    #thing_list = config['INFERENCE']['engine_params']['thing_list']
    #label_divisor = config['INFERENCE']['engine_params']['label_divisor']
    #for axis_name, axis in axes.items():
    #    trackers[axis_name] = [
    #        InstanceTracker(class_id, label_divisor, volume.shape, axis_name)
    #        for class_id in class_labels
    #    ]

    config['world_size'] = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=config['world_size'], args=(config,))

    """

    for axis_name, axis in axes.items():
        print(f'Predicting {axis_name} stack')

        # chunk in axis direction only
        chunks = [None, None, None]
        chunks[axis] = 1
        chunks = tuple(chunks)
        stack = data.create_dataset(f'{config_name}_panoptic_{axis_name}', shape=shape,
                                    dtype=np.uint64, chunks=chunks,
                                    overwrite=True)

        # prime the model for the given image dimension

        # create the inference engine
        inference_engine = engine_cls(model, **config['INFERENCE']['engine_params'])

        # create a separate matcher for each thing class
        matchers = [
            SequentialMatcher(thing_class, label_divisor, **config['INFERENCE']['matcher_params'])
            for thing_class in thing_list
        ]

        queue = mp.Queue()
        matcher_proc = mp.Process(target=run_forward_matchers, args=(stack, axis, matchers, queue))
        matcher_proc.start()

        # make axis-specific dataset


        fill_index = 0
        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            h, w = image.size()[2:]
            image = factor_pad_tensor(image, 128)

            pan_seg = inference_engine(image)
            if pan_seg is None:
                # building the queue
                continue

            #pan_seg = pan_seg[0]
            if pan_seg is None:
                # building the queue
                queue.put((fill_index, pan_seg))
                continue
            else:
                pan_seg = pan_seg.squeeze()[:h, :w] # remove padding and unit dimensions
                queue.put((fill_index, pan_seg.cpu().numpy()))
                fill_index += 1

        final_segs = inference_engine.end()
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                #pan_seg = pan_seg[0]
                pan_seg = pan_seg.squeeze()[:h, :w] # remove padding
                queue.put((fill_index, pan_seg.cpu().numpy()))
                fill_index += 1

        queue.put((fill_index, 'DONE'))
        matcher_proc.join()

        print(f'Propagating labels backward through the stack...')
        # set the matchers to not assign new labels
        # and not split disconnected components

        for matcher in matchers:
            matcher.assign_new = False
            matcher.force_connected = False

        # TODO: multiprocessing the loading with a Queue
        # skip the bottom slice
        rev_indices = np.arange(0, stack.shape[axis])[::-1]
        for rev_idx in tqdm(rev_indices):
            rev_idx = rev_idx.item()
            pan_seg = zarr_take3d(stack, rev_idx, axis)

            for matcher in matchers:
                if matcher.target_seg is None:
                    pan_seg = matcher.initialize_target(pan_seg)
                else:
                    pan_seg = matcher(pan_seg)

            # leave the last slice in the stack alone
            if rev_idx < (stack.shape[axis] - 1):
                zarr_put3d(stack, rev_idx, pan_seg, axis)

            # track each instance for each class
            for tracker in trackers[axis_name]:
                tracker.update(pan_seg, rev_idx)

        # finish tracking
        for tracker in trackers[axis_name]:
            tracker.finish()

        # apply any filters
        if filter_names:
            for filt,kwargs in zip(filter_names, filter_kwargs):
                for tracker in trackers[axis_name]:
                    filters.__dict__[filt](tracker, **kwargs)

    # create the final instance segmentations
    for class_id, class_name in zip(config['INFERENCE']['labels'], config['DATASET']['class_names']):
        # get the relevant trackers for the class_label
        print(f'Creating consensus segmentation for class {class_name}...')

        class_trackers = []
        for axis_name, axis_trackers in trackers.items():
            for tracker in axis_trackers:
                if tracker.class_id == class_id:
                    class_trackers.append(tracker)

        # merge instances from orthoplane inference if applicable
        if len(class_trackers) > 1:
            consensus_tracker = InstanceTracker(class_id, label_divisor, volume.shape, 'xy')

            consensus_tracker.instances = merge_objects3d(class_trackers)

            # apply filters to final merged segmentation
            if filter_names:
                for filt,kwargs in zip(filter_names, filter_kwargs):
                    filters.__dict__[filt](consensus_tracker, **kwargs)
        else:
            consensus_tracker = class_trackers[0]


        # decode and fill the instances
        consensus_vol = data.create_dataset(
            f'{config_name}_{class_name}_pred', shape=shape, dtype=np.uint64,
            overwrite=True, chunks=(1, None, None)
        )
        zarr_fill_instances(consensus_vol, consensus_tracker.instances)
        consensus_tracker.write_to_json(os.path.join(volume_path, f'{config_name}_{class_name}_pred.json'))

    # run evaluation
    semantic_metrics = {'IoU': iou}
    instance_metrics = {'F1_50': f1_50, 'F1_75': f1_75, 'Precision_50': precision_50,
                        'Precision_75': precision_75, 'Recall_50': recall_50, 'Recall_75': recall_75}
    panoptic_metrics = {'PQ': panoptic_quality}
    evaluator = Evaluator(semantic_metrics, instance_metrics, panoptic_metrics)

    for class_name in config['DATASET']['class_names']:
        gt_json = os.path.join(volume_path, f'{class_name}_gt.json')
        pred_json = os.path.join(volume_path, f'{config_name}_{class_name}_pred.json')
        results = evaluator(gt_json, pred_json)
        results = {f'{class_name}_{k}': v for k,v in results.items()}

        for k, v in results.items():
            print(k, v)

        run_id = state.get('run_id')
        if run_id is not None:
            volname = os.path.basename(volume_path).split('.zarr')[0][:20]
            with mlflow.start_run(run_id=run_id) as run:
                for k, v in results.items():
                    mlflow.log_metric(f'{volname}_{k}', v, step=0)
    """
