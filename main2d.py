import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import time
from postprocess import get_panoptic_segmentation

end_file = 'final.log'

def factor_pad_tensor(tensor, factor=16):
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    else:
        return nn.ZeroPad2d((0, pad_right, 0, pad_bottom))(tensor)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('imdir', type=str)
    parser.add_argument('savedir', type=str)
    args = parser.parse_args()

    # load the model
    model = torch.jit.load(args.model_file)

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    current_files = set()
    processed_files = set(os.listdir(args.savedir))
    end_of_acquisition = False
    while not end_of_acquisition:
        #user_input = input()
        current_files = set(os.listdir(args.imdir))
        files_to_process = current_files - processed_files

        if end_file in current_files:
            end_of_acquisition = True

        for imf in files_to_process:
            try:
                image = cv2.imread(os.path.join(args.imdir, imf), 0)
                h, w = image.shape
            except Exception as err:
                # this happens when image isn't fully saved
                continue

            image = (image / 255).astype(np.float32)
            image = torch.from_numpy(image)[None, None]
            image = normalize_tensor(image, mean=0.6, std=0.1)
            image = factor_pad_tensor(image)

            start = time.time()
            output = model(image)
            print(f'Image size {(h, w)}, inference time {time.time() - start}')

            start = time.time()
            pan_seg, centers = get_panoptic_segmentation(
                (torch.sigmoid(output['sem_logits']) > 0.5).long(),
                output['ctr_hmp'], output['offsets'],
                [1], 1000, 64, 0, 0.1, 7
            )
            pan_seg = pan_seg[..., :h, :w].squeeze().numpy()

            print(f'Centers size {centers.size()}, postprocessing time {time.time() - start}')
            pan_seg = id2rgb(pan_seg)

            cv2.imwrite(os.path.join(args.savedir, imf), pan_seg)
            processed_files.update([imf])

            time.sleep(2)

    print('Finished!')
