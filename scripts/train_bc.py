import os
import time
import yaml
import argparse
import mlflow
import random

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
from skimage import measure
from matplotlib import pyplot as plt

from empanada import losses
from empanada import data
from empanada import metrics
from empanada import models
from empanada.config_loaders import load_config
from empanada.data.utils.sampler import DistributedWeightedSampler
from empanada.data.utils.transforms import FactorPad

archs = sorted(name for name in models.__dict__
    if callable(models.__dict__[name])
)

schedules = sorted(name for name in lr_scheduler.__dict__
    if callable(lr_scheduler.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

optimizers = sorted(name for name in optim.__dict__
    if callable(optim.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

augmentations = sorted(name for name in A.__dict__
    if callable(A.__dict__[name]) and not name.startswith('__')
    and name[0].isupper()
)

datasets = sorted(name for name in data.__dict__
    if callable(data.__dict__[name])
)

def parse_args():
    parser = argparse.ArgumentParser(description='Runs panoptic deeplab training')
    parser.add_argument('config', type=str, metavar='config', help='Path to a config yaml file')
    return parser.parse_args()

def main():
    args = parse_args()

    # read the config file
    config = load_config(args.config)

    config['config_file'] = args.config
    config['config_name'] = os.path.basename(args.config).split('.yaml')[0]
    
    # create model directory if None
    if not os.path.isdir(config['TRAIN']['model_dir']):
        os.mkdir(config['TRAIN']['model_dir'])

    # validate parameters
    assert config['MODEL']['arch'] in archs
    assert config['TRAIN']['lr_schedule'] in schedules
    assert config['TRAIN']['optimizer'] in optimizers
    
    if 'Accuracy' in config['TRAIN']['metrics']:
        assert config['MODEL']['confidence_head']

    if config['TRAIN']['dist_url'] == "env://" and config['TRAIN']['world_size'] == -1:
        config['TRAIN']['world_size'] = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    if config['TRAIN']['multiprocessing_distributed']:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config['TRAIN']['world_size'] = ngpus_per_node * config['TRAIN']['world_size']
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config['TRAIN']['gpu'], ngpus_per_node, config)

def main_worker(gpu, ngpus_per_node, config):
    config['gpu'] = gpu
    
    if config['gpu'] is not None:
        print(f"Use GPU: {gpu} for training")

    if config['TRAIN']['multiprocessing_distributed']:
        if config['TRAIN']['dist_url'] == "env://" and config['TRAIN']['rank'] == -1:
            config['TRAIN']['rank'] = int(os.environ["RANK"])
        if config['TRAIN']['multiprocessing_distributed']:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config['TRAIN']['rank'] = config['TRAIN']['rank'] * ngpus_per_node + config['gpu']

        dist.init_process_group(backend=config['TRAIN']['dist_backend'], init_method=config['TRAIN']['dist_url'],
                                world_size=config['TRAIN']['world_size'], rank=config['TRAIN']['rank'])

    # setup the model and pick dataset class
    model_arch = config['MODEL']['arch']
    model = models.__dict__[model_arch](**config['MODEL'])
    dataset_class_name = config['TRAIN']['dataset_class']
    data_cls = data.__dict__[dataset_class_name]
        
    # load pre-trained weights, if using
    if config['TRAIN']['whole_pretraining'] is not None:
        state = torch.load(config['TRAIN']['whole_pretraining'], map_location='cpu')
        state_dict = state['state_dict']
        
        # remove the prefix 'module' from all of the keys
        for k in list(state_dict.keys()):
            if k.startswith('module'):
                state_dict[k[len('module.'):]] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]
            
        msg = model.load_state_dict(state['state_dict'], strict=True)
        norms = state['norms']
    elif config['TRAIN']['encoder_pretraining'] is not None:
        state = torch.load(config['TRAIN']['encoder_pretraining'], map_location='cpu')
        state_dict = state['state_dict']
        
        # add the prefix 'encoder' to all of the keys
        for k in list(state_dict.keys()):
            if not k.startswith('fc'):
                state_dict['encoder.' + k] = state_dict[k]

            # delete renamed or unused k
            del state_dict[k]
            
        msg = model.load_state_dict(state['state_dict'], strict=False)
        norms = {}
        norms['mean'] = state['norms'][0]
        norms['std'] = state['norms'][1]
    else:
        norms = config['DATASET']['norms']

    finetune_layer = config['TRAIN']['finetune_layer']
    # start by freezing all encoder parameters
    for pname, param in model.named_parameters():
        if 'encoder' in pname:
            param.requires_grad = False
    # freeze encoder layers
    if finetune_layer == 'none':
        # leave all encoder layers frozen
        pass
    elif finetune_layer == 'all':
        # unfreeze all encoder parameters
        for pname, param in model.named_parameters():
            if 'encoder' in pname:
                param.requires_grad = True
    else:
        valid_layers = ['stage1', 'stage2', 'stage3', 'stage4']
        assert finetune_layer in valid_layers
        # unfreeze all layers from finetune_layer onward
        for layer_name in valid_layers[valid_layers.index(finetune_layer):]:
            # freeze all encoder parameters
            for pname, param in model.named_parameters():
                if f'encoder.{layer_name}' in pname:
                    param.requires_grad = True

    num_trainable = sum(p[1].numel() for p in model.named_parameters() if p[1].requires_grad)
    print(f'Model with {num_trainable} trainable parameters.')
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif config['TRAIN']['multiprocessing_distributed']:
        # use Synced batchnorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config['gpu'] is not None:
            torch.cuda.set_device(config['gpu'])
            model.cuda(config['gpu'])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config['TRAIN']['batch_size'] = int(config['TRAIN']['batch_size'] / ngpus_per_node)
            config['TRAIN']['workers'] = int((config['TRAIN']['workers'] + ngpus_per_node - 1) / ngpus_per_node)
            model = DDP(model, device_ids=[config['gpu']])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = DDP(model)
    elif config['gpu'] is not None:
        torch.cuda.set_device(config['gpu'])
        model = model.cuda(config['gpu'])
    else:
        # script the model
        #model = torch.jit.script(model, optimize=True)
        model = torch.nn.DataParallel(model).cuda()
        #raise Exception

    cudnn.benchmark = True
        
    # set the training image augmentations
    config['aug_string'] = []
    dataset_augs = []
    for aug_params in config['TRAIN']['augmentations']:
        aug_name = aug_params['aug']
        
        assert aug_name in augmentations, \
        f'{aug_name} is not a valid albumentations augmentation!'
        
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
    train_dataset = data_cls(config['TRAIN']['train_dir'], tfs, weight_gamma=config['TRAIN']['weight_gamma'])
    if config['TRAIN']['additional_train_dirs'] is not None:
        for train_dir in config['TRAIN']['additional_train_dirs']:
            add_dataset = data_cls(train_dir, tfs, weight_gamma=config['TRAIN']['weight_gamma'])
            train_dataset = train_dataset + add_dataset
            
    if config['TRAIN']['multiprocessing_distributed']:
        if config['TRAIN']['weight_gamma'] is None:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = DistributedWeightedSampler(train_dataset, train_dataset.weights)
    elif config['TRAIN']['weight_gamma'] is not None:
        train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))
    else:
        train_sampler = None
        
    # num workers always less than number of batches in train dataset
    num_workers = min(config['TRAIN']['workers'], len(train_dataset) // config['TRAIN']['batch_size'])

    train_loader = DataLoader(
        train_dataset, batch_size=config['TRAIN']['batch_size'], shuffle=(train_sampler is None),
        num_workers=config['TRAIN']['workers'], pin_memory=torch.cuda.is_available(), sampler=train_sampler,
        drop_last=True
    )

    if config['EVAL']['eval_dir'] is not None:
        eval_tfs = A.Compose([
            FactorPad(128), # pad image to be divisible by 128
            A.Normalize(**norms),
            ToTensorV2()
        ])
        eval_dataset = data_cls(config['EVAL']['eval_dir'], eval_tfs)
        # evaluation runs on a single gpu
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, 
                                 pin_memory=torch.cuda.is_available(), 
                                 num_workers=config['TRAIN']['workers'])
        
        # pick images to track during validation
        if config['EVAL']['eval_track_indices'] is None:
            # randomly pick 8 examples from eval set to track
            config['EVAL']['eval_track_indices'] = [random.randint(0, len(eval_dataset)) for _ in range(8)]
        
    else:
        eval_loader = None
        
    # set criterion
    criterion_name = config['TRAIN']['criterion']
    if config['gpu'] is not None:
        criterion = losses.__dict__[criterion_name](**config['TRAIN']['criterion_params']).cuda(config['gpu'])
    else:
        criterion = losses.__dict__[criterion_name](**config['TRAIN']['criterion_params']).cuda()

    # set optimizer and lr scheduler
    opt_name = config['TRAIN']['optimizer']
    opt_params = config['TRAIN']['optimizer_params']
    optimizer = configure_optimizer(model, opt_name, **opt_params)
    
    schedule_name = config['TRAIN']['lr_schedule']
    schedule_params = config['TRAIN']['schedule_params']
    
    if 'steps_per_epoch' in schedule_params:
        n_steps = schedule_params['steps_per_epoch']
        if n_steps != len(train_loader):
            schedule_params['steps_per_epoch'] = len(train_loader)
            print(f'Steps per epoch adjusted from {n_steps} to {len(train_loader)}')
    
    scheduler = lr_scheduler.__dict__[schedule_name](optimizer, **schedule_params)

    scaler = GradScaler() if config['TRAIN']['amp'] else None

    # optionally resume from a checkpoint
    config['run_id'] = None
    config['start_epoch'] = 0
    if config['TRAIN']['resume'] is not None:
        if os.path.isfile(config['TRAIN']['resume']):
            print("=> loading checkpoint")
            if config['gpu'] is None:
                checkpoint = torch.load(config['TRAIN']['resume'])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config['gpu'])
                checkpoint = torch.load(config['TRAIN']['resume'], map_location=loc)

            config['start_epoch'] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
                
            # use the saved norms
            norms = checkpoint['norms']
            config['run_id'] = checkpoint['run_id']

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config['TRAIN']['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config['TRAIN']['resume']))

    # training and evaluation loop
    if 'epochs' in config['TRAIN']['schedule_params']:
        epochs = config['TRAIN']['schedule_params']['epochs']
    elif 'epochs' in config['TRAIN']:
        epochs = config['TRAIN']['epochs']
    else:
        raise Exception('Number of training epochs not defined!')

    config['TRAIN']['epochs'] = epochs

    # log important parameters and start/resume mlflow run
    prepare_logging(config)

    for epoch in range(config['start_epoch'], epochs):
        if config['TRAIN']['multiprocessing_distributed']:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, 
              scheduler, scaler, epoch, config)
        
        is_distributed = config['TRAIN']['multiprocessing_distributed']
        gpu_rank = config['TRAIN']['rank'] % ngpus_per_node

        # evaluate on validation set, does not support multiGPU
        if eval_loader is not None and (epoch + 1) % config['EVAL']['epochs_per_eval'] == 0:
            if not is_distributed or (is_distributed and gpu_rank == 0):
                validate(eval_loader, model, criterion, epoch, config)

        save_now = (epoch + 1) % config['TRAIN']['save_freq'] == 0
        if save_now and not is_distributed or (is_distributed and gpu_rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config['MODEL']['arch'],
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'run_id': mlflow.active_run().info.run_id,
                'norms': norms
            }, os.path.join(config['TRAIN']['model_dir'], f"{config['config_name']}_checkpoint.pth.tar"))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def prepare_logging(config):
    # log parameters for run, or resume existing run
    if config['run_id'] is None and config['TRAIN']['rank'] == 0:
        # log parameters in mlflow
        mlflow.end_run()
        mlflow.set_experiment(config['DATASET']['dataset_name'])
        
        # log the full config file after inheritance
        artifact_path = 'mlruns/' + mlflow.get_artifact_uri().split('/mlruns/')[-1]
        config_fp = os.path.join(artifact_path, os.path.basename(config['config_file']))
        with open(config_fp, mode='w') as f:
            yaml.dump(config, f)

        #we don't want to add everything in the config
        #to mlflow parameters, we'll just add the most
        #likely to change parameters
        mlflow.log_param('run_name', config['TRAIN']['run_name'])
        mlflow.log_param('architecture', config['MODEL']['arch'])
        mlflow.log_param('encoder_pretraining', config['TRAIN']['encoder_pretraining'])
        mlflow.log_param('whole_pretraining', config['TRAIN']['whole_pretraining'])
        mlflow.log_param('epochs', config['TRAIN']['epochs'])
        mlflow.log_param('batch_size', config['TRAIN']['batch_size'])
        mlflow.log_param('lr_schedule', config['TRAIN']['lr_schedule'])
        mlflow.log_param('optimizer', config['TRAIN']['optimizer'])

        aug_names = config['aug_string']
        mlflow.log_param('augmentations', aug_names)
    else:
        # resume existing run
        mlflow.start_run(run_id=config['run_id'])
        
def log_metrics(progress, meters, epoch, dataset='Train'):
    # log all the losses from progress
    for meter in progress.meters:
        mlflow.log_metric(f'{dataset}_{meter.name}', meter.avg, step=epoch)
        
    for metric_name, values in meters.history.items():
        mlflow.log_metric(f'{dataset}_{metric_name}', values[-1], step=epoch)
    
def configure_optimizer(model, opt_name, **opt_params):
    """
    Takes an optimizer and separates parameters into two groups
    that either use weight decay or are exempt.
    
    Only BatchNorm parameters and biases are excluded.
    """
    
    # easy if there's no weight_decay
    if 'weight_decay' not in opt_params:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)
    elif opt_params['weight_decay'] == 0:
        return optim.__dict__[opt_name](model.parameters(), **opt_params)

    # otherwise separate parameters into two groups
    decay = set()
    no_decay = set()

    blacklist = (torch.nn.BatchNorm2d,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            full_name = '%s.%s' % (mn, pn) if mn else pn

            if full_name.endswith('bias'):
                no_decay.add(full_name)
            elif full_name.endswith('weight') and isinstance(m, blacklist):
                no_decay.add(full_name)
            else:
                decay.add(full_name)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert(len(inter_params) == 0), "Overlapping decay and no decay"
    assert(len(param_dict.keys() - union_params) == 0), "Missing decay parameters"

    decay_params = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]

    #the adapt_lr key tells LARS not to adapt the lr (see 'LARC.py')
    param_groups = [
        {"params": decay_params, **opt_params},
        {"params": no_decay_params, **opt_params}
    ]
    param_groups[1]['weight_decay'] = 0 # overwrite default to 0 for no_decay group

    return optim.__dict__[opt_name](param_groups, **opt_params)

def train(
    train_loader, 
    model, 
    criterion, 
    optimizer, 
    scheduler, 
    scaler, 
    epoch, 
    config
):
    # generic progress
    batch_time = ProgressAverageMeter('Time', ':6.3f')
    data_time = ProgressAverageMeter('Data', ':6.3f')
    losses = ProgressEMAMeter('Total_Loss', ':.4e')
    sem_ce_losses = ProgressEMAMeter('sem_CE_Loss', ':.4e')
    cnt_ce_losses = ProgressEMAMeter('cnt_CE_Loss', ':.4e')
    

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, sem_ce_losses, cnt_ce_losses],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    # end of epoch metrics
    class_names = config['DATASET']['class_names']
    metric_dict = {}
    for metric_params in config['TRAIN']['metrics']:
        reg_name = metric_params['name']
        metric_name = metric_params['metric']
        metric_params = {k: v for k,v in metric_params.items() if k not in ['name', 'metric']}
        metric_dict[reg_name] = metrics.__dict__[metric_name](metrics.EMAMeter, **metric_params) 

    meters = metrics.ComposeMetrics(metric_dict, class_names)

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        images = batch['image']
        target = {k: v for k,v in batch.items() if k not in ['image', 'fname']}

        if config['gpu'] is not None:
            images = images.cuda(config['gpu'], non_blocking=True)
        if torch.cuda.is_available():
            target = {k: tensor.cuda(config['gpu'], non_blocking=True) 
                      for k,tensor in target.items()}
            
        # zero grad before running
        optimizer.zero_grad()

        # compute output
        if scaler is not None:
            with autocast():
                output = model(images)
                loss, aux_loss = criterion(output, target)  # output and target are both dicts
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(images)
            loss, aux_loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        # update the LR    
        scheduler.step()

        # record losses
        losses.update(loss.item())
        sem_ce_losses.update(aux_loss['sem_ce'])
        cnt_ce_losses.update(aux_loss['cnt_ce'])
        
        # calculate human-readable per epoch metrics
        with torch.no_grad():
            meters.evaluate(output, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['TRAIN']['print_freq'] == 0:
            progress.display(i)
            
    # end of epoch print evaluation metrics
    print('\n')
    print(f'Epoch {epoch} training metrics:')
    meters.display()
    log_metrics(progress, meters, epoch, dataset='Train')
    
def validate(
    eval_loader, 
    model, 
    criterion, 
    epoch, 
    config
):
    # metric tracking
    class_names = config['DATASET']['class_names']
    metric_dict = {}
    for metric_params in config['TRAIN']['metrics']:
        reg_name = metric_params['name']
        metric_name = metric_params['metric']
        metric_params = {k: v for k,v in metric_params.items() if k not in ['name', 'metric']}
        metric_dict[reg_name] = metrics.__dict__[metric_name](metrics.AverageMeter, **metric_params) 

    meters = metrics.ComposeMetrics(metric_dict, class_names)
    
    # switch to eval mode
    model.eval()
    
    # validation tracking
    batch_time = ProgressAverageMeter('Time', ':6.3f')
    losses = ProgressAverageMeter('Total_Loss', ':.4e')
    
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, losses],
        prefix='Validation: '
    )
    
    for i, batch in enumerate(eval_loader):
        end = time.time()
        images = batch['image']
        target = {k: v for k,v in batch.items() if k not in ['image', 'fname']}

        if config['gpu'] is not None:
            images = images.cuda(config['gpu'], non_blocking=True)
        if torch.cuda.is_available():
            target = {k: tensor.cuda(config['gpu'], non_blocking=True) 
                      for k,tensor in target.items()}
            
        # compute panoptic segmentations
        # from prediction and ground truth
        output = model(images)
        output['sem'] = torch.sigmoid(output['sem_logits']) > 0.5
        output['cnt'] = torch.sigmoid(output['cnt_logits']) > 0.5
 
        loss, aux_loss = criterion(output, target)
        losses.update(loss.item())
        
        # compute metrics
        with torch.no_grad():
            meters.evaluate(output, target)
            
        batch_time.update(time.time() - end)
            
        if i % config['TRAIN']['print_freq'] == 0:
            progress.display(i)
            
    # end of epoch print evaluation metrics
    print('\n')
    print(f'Validation results:')
    meters.display()
    log_metrics(progress, meters, epoch, dataset='Eval')
    
class ProgressAverageMeter(metrics.AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        super().__init__()

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressEMAMeter(metrics.EMAMeter):
    """Computes and stores the exponential moving average and current value"""
    def __init__(self, name, fmt=':f', momentum=0.98):
        self.name = name
        self.fmt = fmt
        super().__init__(momentum)

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
if __name__ == "__main__":
    main()
