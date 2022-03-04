import os
import yaml
import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from empanada import data
from torch.utils.data import DataLoader, WeightedRandomSampler

from empanada.models import quantization as quant_models
from empanada.models.quantization.point_rend import QuantizablePointRendSemSegHead

from empanada.config_loaders import load_config

augmentations = sorted(name for name in A.__dict__
    if callable(A.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

datasets = sorted(name for name in data.__dict__
    if callable(data.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

def parse_args():
    parser = argparse.ArgumentParser(description='Exports an optionally quantized panoptic deeplab model')
    parser.add_argument('config', type=str, metavar='config', help='Path to a config yaml file')
    parser.add_argument('save_path', type=str, metavar='save_path', help='Path to a save the quantized model')
    parser.add_argument('-nc', type=int, default=32, metavar='nc', help='Number of calibration batches for quantization')
    parser.add_argument('-pf', type=int, default=128, metavar='pf',
                        help='Factor by which image dimensions must be divisible for model inference')
    return parser.parse_args()

def create_dataloader(config, norms):
    # create the data loader
    # set the training image augmentations
    config['aug_string'] = []
    dataset_augs = []
    for aug_params in config['TRAIN']['augmentations']:
        aug_name = aug_params['aug']

        assert aug_name in augmentations \
        f'{aug_name} is not a valid augmentation!'

        config['aug_string'].append(aug_params['aug'])
        del aug_params['aug']
        dataset_augs.append(A.__dict__[aug_name](**aug_params))

    config['aug_string'] = ','.join(config['aug_string'])

    tfs = A.Compose([
        *dataset_augs,
        A.Normalize(**norms),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.05))

    # create training dataset and loader
    dataset_class_name = config['TRAIN']['dataset_class']
    data_cls = data.__dict__[dataset_class_name]
    train_dataset = data_cls(config['TRAIN']['train_dir'], tfs, weight_gamma=config['TRAIN']['weight_gamma'])
    if config['TRAIN']['additional_train_dirs'] is not None:
        for train_dir in config['TRAIN']['additional_train_dirs']:
            add_dataset = data_cls(train_dir, tfs, weight_gamma=config['TRAIN']['weight_gamma'])
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
    config = load_config(args.config)

    # get model path from train.py saving convention
    config_name = os.path.basename(args.config).split('.yaml')[0]
    model_fpath = os.path.join(config['TRAIN']['model_dir'], f"{config_name}_checkpoint.pth.tar")

    save_path = args.save_path
    num_calibration_batches = args.nc

    # create model directory if None
    if not os.path.isfile(model_fpath):
        raise Exception(f'Model {model_fpath} does not exist!')

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # validate parameters
    model_arch = config['MODEL']['arch']
    base_arch = model_arch[:-len('PR')]
    base_quant_arch = 'Quantizable' + base_arch

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

    # prep state dict for base and render models
    base_state_dict = {}
    pr_state_dict = {}

    for key in list(state_dict.keys()):
        if key.startswith('semantic_pr.'):
            pr_state_dict[key[len('semantic_pr.'):]] = state_dict[key]
        else:
            base_state_dict[key] = state_dict[key]

    # create the GPU and CPU versions of the models
    pr_nin = config['MODEL']['fpn_dim'] if 'fpn_dim' in config['MODEL'] else config['MODEL']['decoder_channels']
    gpu_base_model = quant_models.__dict__[base_quant_arch](**config['MODEL'], quantize=False)
    gpu_render_model = QuantizablePointRendSemSegHead(nin=pr_nin ,**config['MODEL'])

    # load the state dicts
    gpu_base_model.load_state_dict(base_state_dict, strict=True)
    gpu_render_model.load_state_dict(pr_state_dict, strict=True)

    # export the gpu models
    gpu_base_model.eval()
    gpu_base_model.fuse_model()
    gpu_base_model.cuda()
    gpu_base_model = torch.jit.script(gpu_base_model)

    gpu_render_model.eval()
    gpu_render_model.cuda()
    gpu_render_model = torch.jit.script(gpu_render_model)

    torch.jit.save(gpu_base_model, os.path.join(save_path, f'{model_arch}_{config_name}_base_gpu.pth'))
    torch.jit.save(gpu_render_model, os.path.join(save_path, f'{model_arch}_{config_name}_render_gpu.pth'))
    print('Exported GPU models successfully!')

    cpu_base_model = quant_models.__dict__[base_quant_arch](**config['MODEL'], quantize=True)
    cpu_render_model = QuantizablePointRendSemSegHead(nin=pr_nin, **config['MODEL'])
    cpu_base_model.load_state_dict(base_state_dict, strict=True)
    cpu_render_model.load_state_dict(pr_state_dict, strict=True)

    print('Quantizing model...')

    # create the data loader
    train_loader = create_dataloader(config, norms)

    cpu_base_model.eval()
    cpu_base_model.fuse_model()
    cpu_render_model.eval()

    # specify quantization configuration
    cpu_base_model.set_qconfig('fbgemm')
    torch.quantization.prepare(cpu_base_model, inplace=True)

    # calibrate with the training set
    for i in range(num_calibration_batches):
        batch = iter(train_loader).next()
        print(f'Calibration batch {i + 1} of {num_calibration_batches}')
        with torch.no_grad():
            images = batch['image']
            output = cpu_base_model(images)

    torch.quantization.convert(cpu_base_model, inplace=True)
    print('Model quantized successfully!')

    torch.jit.save(torch.jit.script(cpu_base_model), os.path.join(save_path, f'{model_arch}_{config_name}_base_cpu.pth'))
    torch.jit.save(torch.jit.script(cpu_render_model), os.path.join(save_path, f'{model_arch}_{config_name}_render_cpu.pth'))
    print('Exported CPU models successfully!')

    # export a yaml file describing the models
    desc = {
        'base_model_cpu': os.path.join(os.path.abspath(save_path), f'{model_arch}_{config_name}_base_cpu.pth'),
        'render_model_cpu': os.path.join(os.path.abspath(save_path), f'{model_arch}_{config_name}_render_cpu.pth'),
        'base_model_gpu': os.path.join(os.path.abspath(save_path), f'{model_arch}_{config_name}_base_gpu.pth'),
        'render_model_gpu': os.path.join(os.path.abspath(save_path), f'{model_arch}_{config_name}_render_gpu.pth'),
        'norms': {'mean': norms['mean'], 'std': norms['std']},
        'padding_factor': args.pf,
        'labels': config['DATASET']['labels'],
        'thing_list': config['DATASET']['thing_list'],
        'class_names': config['DATASET']['class_names']
    }

    with open(os.path.join(save_path, f'{model_arch}_{config_name}.yaml'), mode='w') as f:
        yaml.dump(desc, f)

    print('Saved .yaml metadata file!')

if __name__ == "__main__":
    main()
