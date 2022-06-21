import os
import yaml
import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from empanada import data
from torch.utils.data import DataLoader, WeightedRandomSampler
from empanada.models import quantization as quant_models
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
    parser.add_argument('-pf', type=int, default=128, metavar='pf',
                        help='Factor by which image dimensions must be divisible for model inference')
    parser.add_argument('-nc', type=int, default=32, metavar='nc', help='Number of calibration batches for quantization')
    parser.add_argument('--quantize', action='store_true', help='If given, model will be quantized to optimize CPU performance.')
    return parser.parse_args()

def create_dataloader(config, norms):
    # create the data loader
    # set the training image augmentations
    config['aug_string'] = []
    dataset_augs = []
    for aug_params in config['TRAIN']['augmentations']:
        aug_name = aug_params['aug']

        assert aug_name in augmentations, \
        f'{aug_name} is not a valid augmentation!'

        config['aug_string'].append(aug_params['aug'])
        del aug_params['aug']
        dataset_augs.append(A.__dict__[aug_name](**aug_params))

    config['aug_string'] = ','.join(config['aug_string'])

    tfs = A.Compose([
        *dataset_augs,
        A.Normalize(**norms),
        ToTensorV2()
    ])

    # create training dataset and loader
    dataset_class_name = config['TRAIN']['dataset_class']
    data_cls = data.__dict__[dataset_class_name]
    
    train_dataset = data_cls(config['TRAIN']['train_dir'], tfs, **config['TRAIN']['dataset_params'])
    if config['TRAIN']['additional_train_dirs'] is not None:
        for train_dir in config['TRAIN']['additional_train_dirs']:
            add_dataset = data_cls(train_dir, tfs, **config['TRAIN']['dataset_params'])
            train_dataset = train_dataset + add_dataset

    if config['TRAIN']['dataset_params']['weight_gamma'] is not None:
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
    quant_arch = 'Quantizable' + model_arch

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

    model = quant_models.__dict__[quant_arch](**config['MODEL'], quantize=False)

    # prep the model
    model.load_state_dict(state_dict)
    model.eval()
    model.fuse_model()
    model.cuda()
    model = torch.jit.script(model)
    
    print('Model scripted successfully.')
    
    model_out = os.path.join(save_path, f'{model_arch}_{config_name}.pth')
    torch.jit.save(model, model_out)
    print('Exported model successfully.')
    
    # NOTE: Do this after saving or model performance is degraded
    with torch.no_grad():
        x = torch.randn((1, 1, 256, 256)).cuda()
        output = model(x)
            
    print('Validated forward pass.')

    # make the CPU model with quantization
    if args.quantize:
        print('Quantizing model...')
        cpu_model = quant_models.__dict__[quant_arch](**config['MODEL'], quantize=True)
        cpu_model.load_state_dict(state_dict)
        cpu_model.eval()
        cpu_model.fuse_model()

        # specify quantization configuration
        cpu_model.fix_qconfig('fbgemm')
        cpu_model.prepare_quantization()

        # calibrate with the training set
        train_loader = create_dataloader(config, norms)
        for i in range(num_calibration_batches):
            batch = iter(train_loader).next()
            print(f'Calibration batch {i + 1} of {num_calibration_batches}')
            with torch.no_grad():
                images = batch['image']
                output = cpu_model(images)

        torch.quantization.convert(cpu_model, inplace=True)
        print('Model quantized successfully.')

        cpu_model = torch.jit.script(cpu_model)
        
        cpu_model_out = os.path.join(save_path, f'{model_arch}_{config_name}_quantized.pth')
        torch.jit.save(cpu_model, cpu_model_out)
        print('Exported quantized model successfully.')
        
        with torch.no_grad():
            x = torch.randn((1, 1, 256, 256))
            output = cpu_model(x)
            
        print('Validated forward pass.')
    else:
        cpu_model_out = None

    # export a yaml file describing the models
    finetune_params = {
        'dataset_class': config['TRAIN']['dataset_class'],
        'dataset_params': config['TRAIN']['dataset_params'],
        'criterion': config['TRAIN']['criterion'],
        'criterion_params': config['TRAIN']['criterion_params'],
        'engine': config['EVAL']['engine'],
        'engine_params': config['EVAL']['engine_params'],
    }
    desc = {
        'model': model_out,
        'model_quantized': cpu_model_out,
        'norms': {'mean': norms['mean'], 'std': norms['std']},
        'padding_factor': args.pf,
        'thing_list': config['DATASET']['thing_list'],
        'labels': config['DATASET']['labels'],
        'class_names': config['DATASET']['class_names'],
        'FINETUNE': finetune_params
    }

    with open(os.path.join(save_path, f'{model_arch}_{config_name}.yaml'), mode='w') as f:
        yaml.dump(desc, f)

    print('Saved .yaml metadata file!')

if __name__ == "__main__":
    main()
