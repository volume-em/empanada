import os
import cv2
import sys
import yaml
import zarr
import argparse
import numpy as np
import torch
import time
from copy import deepcopy
from glob import glob
from skimage.io.collection import alphanumeric_key
from tqdm import tqdm

# imports managed in each submodule
import metadata_parsers
import preprocess
from preprocess import create_image_pyramid
from alignment import xcorr_transform, apply_affine
from mitonet.inference.engines import MultiScaleInferenceEngine
from mitonet.inference.matcher import SequentialMatcher
from mitonet.inference.tracker import InstanceTracker
from mitonet.zarr_utils import zarr_put3d, zarr_take3d

md_parsers = sorted(name for name in metadata_parsers.__dict__
    if callable(metadata_parsers.__dict__[name]) and not name.startswith('__')
)

preops = sorted(name for name in preprocess.__dict__
    if callable(preprocess.__dict__[name]) and not name.startswith('__')
)

"""
TODO:
Bin at the beginning
Crop chevrons
Make apply_affine work with multiprocessing
"""

def preprocess_image(image, ops):
    ops = deepcopy(ops)
    for op in ops:
        op_name = op['op']
        del op['op']
        image = preprocess.__dict__[op_name](image, **op)
    
    return image

def factor_pad_tensor(tensor, factor=16):
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    else:
        return torch.nn.ZeroPad2d((0, pad_right, 0, pad_bottom))(tensor)
    
def rescale_intensity(image):
    if image.dtype.kind == 'f':
        min_val = image.min()
        max_val = image.max()
    else:
        type_info = np.iinfo(image.dtype)
        min_val = type_info.min
        max_val = type_info.max
        
    image = image.astype(np.float32)
    
    return (image - min_val) / (max_val - min_val)

def normalize_tensor(tensor, mean, std):
    return (tensor - mean) / std

def id2rgb(id_map):
    id_map_copy = id_map.copy()
    rgb_shape = tuple(list(id_map.shape) + [3])
    rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
    for i in range(3):
        rgb_map[..., i] = id_map_copy % 256
        id_map_copy //= 256
    return rgb_map

def load_images(impaths):
    return [cv2.imread(imp, 0) for imp in impaths]

def xform_images(images, args):
    # transform images iteratively
    n = len(images)
    xforms = []
    for t, m in zip(range(0, n - 1), range(1, n)):
        target_image = images[t]
        moving_image = images[m]
        
        # calculate the transform and apply it
        xform = xcorr_transform(target_image, moving_image, args.alignment_scale)
        moving_image = apply_affine(moving_image, xform)
        
        # update the images and transforms
        images[m] = moving_image
        xforms.append(xform)
        
    return images, xforms

def create_multiscale_zarr(args, n_images, image_shapes):
    # create the main container
    data = zarr.open(args.savedir)
    
    # create the datasets
    em_group = data.create_group('em', overwrite=True)
    label_group = data.create_group('labels', overwrite=True)
    
    # create the multiple scales
    for i,shape in enumerate(image_shapes):
        em_group.create_dataset(f's{i}', shape=(n_images, *shape), chunks=tuple(args.chunks), dtype=np.uint8, overwrite=True)
        label_group.create_dataset(f's{i}', shape=(n_images, *shape), chunks=tuple(args.chunks), dtype='int', overwrite=True)
        
    return data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imdir', type=str)
    parser.add_argument('savedir', type=str)
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()

    # read the config file
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['imdir'] = args.imdir
    config['savedir'] = args.savedir
    
    # convert all to a namespace
    args = argparse.Namespace(**config)

    # we'll remove this in the future
    assert args.inference_scale in args.pyramid_scales
    
    if args.metadata_format is not None:
        assert args.metadata_format in md_parsers
        metadata_loader = metadata_parsers.__dict__[args.metadata_format]
    else:
        metadata_loader = lambda x: x
    
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
        
    # setup the segmentation model
    base_model = torch.jit.load(args.panoptic_model)
    render_model = torch.jit.load(args.pointrend_model)
    
    # matchers and trackers
    thing_list = args.engine_params['thing_list']
    label_divisor = args.engine_params['label_divisor']

    xy_trackers = {}
    matchers = {}
    for scale in args.pyramid_scales:
        #xy_trackers[scale] = [
        #    InstanceTracker(thing_class, label_divisor, volume.shape, 'xy') 
        #    for thing_class in args.labels
        #]
        matchers[scale] = [
            SequentialMatcher(thing_class, label_divisor, **args.matcher_params)
            for thing_class in thing_list
        ]

    current_files = set()
    processed_files = set()
    last_image = None
    end_of_acquisition = False
    
    fill_idx = 0
    
    while not end_of_acquisition:
        files_to_process = set()
        # get all files in the directory
        current_files = set([f for f in glob(os.path.join(args.imdir, '*')) if not f.startswith('.')])
        if current_files:
            # get the last file from the current files based on the file names
            # and remove it, it not might have finished saving
            last_file = sorted(current_files, key=alphanumeric_key)[-1]
            current_files.remove(last_file)
            files_to_process = current_files - processed_files
            
        print(current_files, files_to_process)

        if files_to_process and not processed_files:
            # creation phase
            metadata = metadata_loader(list(files_to_process)[0])
            images = load_images(files_to_process)
            images = [preprocess_image(image, args.preprocess) for image in images]
            
            # calculate and transform each image
            initial_xform = np.eye(3)[None]
            if len(images) > 1:
                images, xforms = xform_images(images, args)
                xforms = [initial_xform] + xforms
            else:
                xforms = initial_xform
                
            # create image pyramid for each image
            image_pyramids = []
            for im_x, image in enumerate(images):
                pyramid = create_image_pyramid(image, args.pyramid_scales)
                image_pyramids.append(pyramid)
                
            image_shapes = [img.shape for img in image_pyramids[0]]
            data_store = create_multiscale_zarr(args, len(image_pyramids), image_shapes)
            
            # store the images
            for im_x, pyramid in enumerate(image_pyramids):
                for i, pyr_img in enumerate(pyramid):
                        zarr_put3d(data_store['em'][f's{i}'], im_x, pyr_img, axis=0)

            pyr_idx = args.pyramid_scales.index(args.inference_scale)
            #warmup = factor_pad_tensor(torch.randn((1, 1, *image_pyramids[0][pyr_idx].shape)), 128)

            #print('Warming up JIT model...')
            #for _ in range(3):
            #    with torch.no_grad():
            #        wout = base_model(warmup)

            # warmup the JIT model and create the inference engine
            inference_engine = MultiScaleInferenceEngine(
                base_model, render_model, scales=args.pyramid_scales, input_scale=args.inference_scale,
                **args.engine_params
            )
                
            # run model inference on the given scale
            # of the image pyramid
            for image_pyr in image_pyramids:
                image = image_pyr[pyr_idx]
                image = rescale_intensity(image)
                image = torch.from_numpy(image)[None, None]
                image = normalize_tensor(image, mean=args.model_norms['mean'], std=args.model_norms['mean'])
                image = factor_pad_tensor(image, 128)
                
                start = time.time()
                pan_pyr = inference_engine(image)
                print('Inference time:', time.time() - start)
                if pan_pyr is None:
                    # building the queue
                    continue

                for i, (pan_seg, scale) in enumerate(zip(pan_pyr, args.pyramid_scales)):
                    h, w = image_pyr[i].shape
                    pan_seg = pan_seg[0, :h, :w] # remove padding
                    pan_seg = pan_seg.cpu().numpy()

                    # update the panoptic segmentations for each
                    # thing class by passing it through matchers
                    for matcher in matchers[scale]:
                        if matcher.target_seg is None:
                            pan_seg = matcher.initialize_target(pan_seg)
                        else:
                            pan_seg = matcher(pan_seg)

                    # fill the zarr array
                    zarr_put3d(data_store['labels'][f's{i}'], fill_idx, pan_seg, axis=0)
                    
                fill_idx += 1

            # set last image to the last
            # TRANSFORMED image
            last_image = images[-1]

        elif files_to_process:
            # appending phase
            metadata = metadata_loader(list(files_to_process)[0])
            images = load_images(files_to_process)
            images = [preprocess_image(image, args.preprocess) for image in images]
            
            # calculate and transform each image
            images, xforms = xform_images([last_image] + images, args)
            images = images[1:] # skip last image
                
            # create image pyramid for each image
            image_pyramids = []
            for im_x, image in enumerate(images):
                pyramid = create_image_pyramid(image, args.pyramid_scales)
                for i, pyr_img in enumerate(pyramid):
                    pyr_img = pyr_img[None] # add 1 z for concat
                    data_store['em'][f's{i}'].append(pyr_img, axis=0)
                    data_store['labels'][f's{i}'].append(np.zeros_like(pyr_img).astype('int'), axis=0)
                    
                image_pyramids.append(pyramid)
            
            # run model inference on the given scale
            # of the image pyramid
            pyr_idx = args.pyramid_scales.index(args.inference_scale)
            for image_pyr in image_pyramids:
                image = image_pyr[pyr_idx]
                image = rescale_intensity(image)
                image = torch.from_numpy(image)[None, None]
                image = normalize_tensor(image, mean=args.model_norms['mean'], std=args.model_norms['mean'])
                image = factor_pad_tensor(image, 128)

                pan_pyr = inference_engine(image)
                if pan_pyr is None:
                    # building the queue
                    continue

                for i, (pan_seg, scale) in enumerate(zip(pan_pyr, args.pyramid_scales)):
                    h, w = image_pyr[i].shape
                    pan_seg = pan_seg[0, :h, :w] # remove padding
                    pan_seg = pan_seg.cpu().numpy()

                    # update the panoptic segmentations for each
                    # thing class by passing it through matchers
                    for matcher in matchers[scale]:
                        if matcher.target_seg is None:
                            pan_seg = matcher.initialize_target(pan_seg)
                        else:
                            pan_seg = matcher(pan_seg)

                    # fill the zarr array
                    zarr_put3d(data_store['labels'][f's{i}'], fill_idx, pan_seg, axis=0)
                    
                fill_idx += 1

            # set last image to the last
            # TRANSFORMED image
            last_image = images[-1]

        # add the files which we have yield to the processed list.
        processed_files.update(files_to_process)
        time.sleep(1)

        # wait for user command to end program
        #if 'end' in  lower(user_input):
        #    end_of_acquisition = True

    # run the end of acquisition
    #print('Entering end phase of acquisition!')
    #xy_trackers = engine.apply_reverse_matching(axis='xy')

    # rechunk the em dataset
    """
    from rechunker import rechunk
    target_chunks = (100, 10, 1)
    max_mem = "2GB"
    plan = rechunk(source_array, target_chunks, max_mem,
                   "target_store.zarr",
                   "temp_store.zarr")
    plan.execute()

    print('Running xz inference...')
    engine.reset()
    zarr_data = ZarrData(args.savedir, 'xz')
    xz_loader = DataLoader(zarr_data, num_workers=8)
    for index, data in enumerate(xz_loader):
        pan_segs = engine.run_model_inference(images, **vars(args))
        zarr_put_3d(pan_segs, axis='xz')

    xz_trackers = engine.apply_reverse_matching(axis='xz')

    print('Running yz inference...')
    engine.reset()
    zarr_data = ZarrData(args.savedir, 'yz')
    yz_loader = DataLoader(zarr_data, num_workers=8)
    for index, data in enumerate(yz_loader):
        pan_segs = engine.run_model_inference(images, **vars(args))
        zarr_put_3d(pan_segs, axis='yz')

    yz_trackers = engine.apply_reverse_matching(axis='yz')
    """


    #if args.metadata_format == 'atlas5':
    #    assert (args.slice_thickness is not None), \
    #    "Must provide z slice thickness in nanometers for AT stacks!"

    #main_create_mrc(args)
