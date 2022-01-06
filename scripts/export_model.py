import os
import yaml
import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, WeightedRandomSampler
from empanada.data import MitoData, CopyPaste
from empanada.inference.postprocess import get_panoptic_segmentation
from empanada.models.panoptic_deeplab import *
from empanada.models.quantization.panoptic_deeplab import *

augmentations = sorted(name for name in A.__dict__
    if callable(A.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

def parse_args():
    parser = argparse.ArgumentParser(description='Exports an optionally quantized panoptic deeplab model')
    parser.add_argument('config', type=str, metavar='config', help='Path to a config yaml file')
    parser.add_argument('model_state', type=str, metavar='model_state', help='Path to a model state dict')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to a save the quantized model')
    parser.add_argument('-nc', type=int, default=32, metavar='nc', help='Number of calibration batches for quantization')
    parser.add_argument('--half', action='store_true', help='Whether to store the model in half-precision')
    parser.add_argument('--quantize', action='store_true', help='Whether to quantize the model for CPU')
    return parser.parse_args()

def create_dataloader(config, norms):
    # create the data loader
    # set the training image augmentations
    config['aug_string'] = []
    dataset_augs = []
    for aug_params in config['TRAIN']['augmentations']:
        aug_name = aug_params['aug']
        
        assert aug_name in augmentations or aug_name == 'CopyPaste', \
        f'{aug_name} is not a valid augmentation!'
        
        config['aug_string'].append(aug_params['aug'])
        del aug_params['aug']
        if aug_name == 'CopyPaste':
            dataset_augs.append(CopyPaste(**aug_params))
        else:
            dataset_augs.append(A.__dict__[aug_name](**aug_params))
        
    config['aug_string'] = ','.join(config['aug_string'])
        
    tfs = A.Compose([
        *dataset_augs,
        A.Normalize(**norms),
        ToTensorV2()
    ])
    
    # create training dataset and loader
    train_dataset = MitoData(config['TRAIN']['train_dir'], tfs, weight_gamma=config['TRAIN']['weight_gamma'])
    if config['TRAIN']['additional_train_dirs'] is not None:
        for train_dir in config['TRAIN']['additional_train_dirs']:
            add_dataset = MitoData(train_dir, tfs, weight_gamma=config['TRAIN']['weight_gamma'])
            train_dataset = train_dataset + add_dataset
    
    if config['TRAIN']['weight_gamma'] is not None:
        train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=config['TRAIN']['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['TRAIN']['workers'], pin_memory=True, sampler=train_sampler
    )
    
    return train_loader

def main():
    args = parse_args()

    # read the config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    model_fpath = args.model_state
    save_path = args.save_path
    num_calibration_batches = args.nc
    half = args.half
    quantize = args.quantize
    
    # create model directory if None
    if not os.path.isfile(model_fpath):
        raise Exception(f'Model {model_fpath} does not exist!')
        
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # validate parameters
    model_arch = config['MODEL']['arch']
    assert model_arch in ['PanopticDeepLab', 'PanopticDeepLabPR'], \
    "Only Panoptic-DeepLab currently supports quantization!"
    
    # load the state
    state = torch.load(model_fpath, map_location='cpu')
    norms = state['norms']
    state_dict = state['state_dict']
    
    # remove module. prefix from state_dict keys
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    
    # create the model and prepare for quantization
    if args.quantize:
        if model_arch == 'PanopticDeepLab':
            model = QuantizablePanopticDeepLab(**config['MODEL'])
        else:
            model = QuantizablePanopticDeepLabPR(**config['MODEL'])
    else:
        if model_arch == 'PanopticDeepLab':
            model = PanopticDeepLab(**config['MODEL'])
        else:
            model = PanopticDeepLabPR(**config['MODEL'])
        
    model.load_state_dict(state_dict)
    if args.quantize:
        print('Quantizing model...')
        
        # create the data loader
        train_loader = create_dataloader(config, norms)
        
        model.eval()
        model.fuse_model()
    
        # specify quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)

        # calibrate with the training set
        for i, batch in enumerate(train_loader):
            print(f'Calibration batch {i + 1} of {num_calibration_batches}')
            with torch.no_grad():
                images = batch['image']
                output = model(images)

            if i == num_calibration_batches - 1:
                break

        torch.quantization.convert(model, inplace=True)
        print('Model quantized successfully!')
    elif args.half:
        model.half()
        model.eval()
        print('Model converted to half precision!')
    else:
        model.eval()
    
    torch.jit.save(torch.jit.script(model), save_path)
    print(f'Model exported to {save_path}')
    
if __name__ == "__main__":
    main()
