import argparse
from multiprocessing import Pool

import numpy as np
import torch
import zarr
from tqdm import tqdm

from empanada import models
from empanada.array_utils import *
from empanada.config_loaders import load_config
from empanada.evaluation import *
from empanada.inference.postprocess import factor_pad
from empanada.inference.rle import encode_boundary3d, ins_seg_to_rle_seg
from empanada.inference.stitch import *
from empanada.inference.tile import Cuber
from empanada.inference.watershed import bc_watershed
from empanada.zarr_utils import *

archs = sorted(name for name in models.__dict__
    if callable(models.__dict__[name])
)

def factor_pad(image, factor=16):
    d, h, w = image.shape
    pad_back = factor - d % factor if d % factor != 0 else 0
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    
    padding = ((0, pad_back), (0, pad_bottom), (0, pad_right))

    padded_image = np.pad(image, padding)
    return padded_image

def process_chunks(*args):
    bc, pred, chunk_index, slices = args[0]
    
    roi = bc[(slice(None),) + slices]
    ws_roi = bc_watershed(roi, 0.7, 0.3, 0.5, 32, None, None, use_mask_wts=True)
    
    # fill the prediction
    pred[slices] = ws_roi
    
    rle_roi = ins_seg_to_rle_seg(ws_roi)
    boundaries = encode_boundary3d(ws_roi)# * roi[1] <= (255 * 0.3))
            
    return {chunk_index: {'rle': rle_roi, 'boundaries': boundaries}}

def fill_zarr_mp(*args):
    r"""Helper function for multiprocessing the filling of zarr slices"""
    # fills zarr array with multiprocessing
    array, slices, rle = args[0]
    array[slices] = numpy_fill_instances(np.zeros_like(array[slices]), rle)

def parse_args():
    parser = argparse.ArgumentParser(description='Runs panoptic deeplab training')
    parser.add_argument('model_config', type=str, metavar='model_config', help='Path to a model config yaml file')
    parser.add_argument('infer_config', type=str, metavar='infer_config', help='Path to a inference config yaml file')
    parser.add_argument('volume_path', type=str, metavar='volume_path', help='Path to a Zarr volume')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config_name = args.model_config.split('/')[-1].split('.yaml')[0]
    
    size_thr = 2000

    # read the config files
    config = load_config(args.model_config)
    infer_config = load_config(args.infer_config)

    # merge the config files
    config['INFERENCE'] = infer_config

    volume_path = args.volume_path
    model = torch.jit.load(config['model'])
    
    device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    model.to(device) # move model to GPU 0
    model.eval()
    
    print(config['model'])
    print('Loaded model. Num parameters:', sum(p[1].numel() for p in model.named_parameters()))

    # set the evaluation transforms
    norms = config['norms']
    
    # load the zarr volume
    data = zarr.open(volume_path, mode='r+')
    volume = data['em']
    shape = volume.shape
    d, h, w = shape
    print('Loaded volume of shape', shape)
    
    seg = data.create_dataset(
        f'{config_name}_bc_pred3d', shape=(2, *shape),
        dtype=np.uint8, chunks=(2, *volume.chunks), overwrite=True
    )
    
    # set chunk size based on the chunks
    # of the segmentation array
    max_cs = (384, 384, 384)
    
    cs = tuple([s * (m // s) for s,m in zip(volume.chunks, max_cs)])
    cuber = Cuber(volume.shape, cs, halo=0.1)
    n_cubes = len(cuber.cubes)
    
    for chunk_index, slices in tqdm(cuber.cubes.items(), total=n_cubes):
        # crop the cube of data
        cube = volume[slices['infer']]
        bd, bh, bw = cube.shape

        # preprocess the tensor
        cube = factor_pad(cube, factor=16)
        cube = cube.astype(np.float32) / 255
        cube = (cube - norms['mean']) / norms['std']
        cube = torch.from_numpy(cube)[None, None].to(device)

        # predict
        with torch.no_grad():
            prediction = model(cube)
            sem = torch.sigmoid(prediction['sem_logits'])
            cnt = torch.sigmoid(prediction['cnt_logits'])

        # quantize as numpy
        sem = 255 * sem.squeeze().cpu().numpy()
        cnt = 255 * cnt.squeeze().cpu().numpy()
        
        # remove the padding
        sem = sem[:bd, :bh, :bw][slices['cut']]
        cnt = cnt[:bd, :bh, :bw][slices['cut']]
        
        # store the result
        seg_slices = (slice(None),) + slices['fill']
        seg[seg_slices] = np.stack([sem, cnt], axis=0)
    
    # create the prediction volume
    pred = data.create_dataset(
        'bc3d_mito_pred', shape=volume.shape, dtype=np.uint32, 
        chunks=volume.chunks, overwrite=True
    )
        
    print(f'Running watershed postprocessing...')
    n = len(cuber.cubes)
    args_iter = zip(
        [seg] * n,
        [pred] * n,
        list(cuber.cubes.keys()),
        [sl['fill'] for sl in cuber.cubes.values()]
    )
    
    with Pool(8) as pool:
        chunk_attrs = list(tqdm(pool.imap(process_chunks, args_iter), total=n_cubes))
        
    # unpack the list of chunk attrs
    chunks = {}
    while chunk_attrs:
        chunk_attr = chunk_attrs.pop()

        assert len(chunk_attr) == 1
        for chunk_index, attr in chunk_attr.items():
            chunks[chunk_index] = {}
            for k,v in attr.items():
                chunks[chunk_index][k] = v
                
    print(f'Stitching instance labels...')
    graph, node_lookup = global_instance_graph(chunks, cuber, initial_label=1)
    add_instance_edges(graph, chunks, node_lookup, cuber, 0.1, 100)
    merge_graph_instances(graph)
    remove_small_objects(graph, 1000)
    forward_map = create_forward_map(graph)
    relabel_chunk_rles(chunks, forward_map)
    
    print(f'Storing instance metadata...')
    metadata = {}
    for node in graph.nodes:
        metadata[node] = {
            'box': graph.nodes[node]['box'],
            'area': graph.nodes[node]['area'],
            'chunks': list(graph.nodes[node]['chunk_lookup'].keys())
        }
        
    print(f'Filling the global segmentation...')
    args_iter = zip(
        [pred] * n_cubes,
        [sl['fill'] for sl in cuber.cubes.values()],
        [chunks[ci]['rle'] for ci in cuber.cubes.keys()]
    )
    
    with Pool(8) as pool:
        output = list(tqdm(pool.imap(fill_zarr_mp, args_iter), total=n))