import os
import json
from copy import copy

import argparse
import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.free_form_inpainting_archs as module_arch
from trainer import Trainer
from utils.logger import Logger
from utils.util import make_dirs

from utils.logging_config import logger


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume, output_root_dir=None, pretrained_path=None):
    train_logger = Logger()

    # setup data_loader instances
    inference_only = config.get('inference_only', False)
    if inference_only:
        data_loader = None
    else:
        data_loader = get_instance(module_data, 'data_loader', config)

    if data_loader is not None and data_loader.valid_sampler:
        assert 'valid_data_loader' not in config.keys(), ('valid set can only be eigther split '
                                                          'from train set or specified in config.')
        valid_data_loader = data_loader.split_validation()
    elif 'valid_data_loader' in config.keys():
        valid_data_loader = get_instance(module_data, 'valid_data_loader', config)
    else:
        valid_data_loader = None

    if 'test_data_loader' in config.keys():
        if isinstance(config['test_data_loader'], list):
            test_data_loader = [
                getattr(module_data, entry['type'])(**entry['args'])
                for entry in config['test_data_loader']
            ]
        else:
            test_data_loader = get_instance(module_data, 'test_data_loader', config)
    else:
        test_data_loader = None

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # setup instances of losses
    losses = {
        entry['nickname']: (
            getattr(module_loss, entry['type'])(**entry['args']),
            entry['weight']
        )
        for entry in config['losses']
    }

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    g_params = []
    d_s_params = []
    d_t_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'temporal_discriminator' in name:
            d_t_params.append(param)
        elif 'spatial_discriminator' in name:
            d_s_params.append(param)
        else:
            g_params.append(param)

    optimizer_g = get_instance(torch.optim, 'optimizer', config, g_params)
    if hasattr(model, 'spatial_discriminator'):
        optimizer_d_s = get_instance(torch.optim, 'optimizer', config, d_s_params)
    else:
        optimizer_d_s = None
    if hasattr(model, 'temporal_discriminator'):
        optimizer_d_t = get_instance(torch.optim, 'optimizer', config, d_t_params)
    else:
        optimizer_d_t = None
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer_g)

    trainer = Trainer(
        model, losses, metrics, optimizer_g, optimizer_d_s, optimizer_d_t,
        resume=resume,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        train_logger=train_logger,
        test_data_loader=test_data_loader,
        pretrained_path=pretrained_path
    )

    if output_root_dir is not None:
        make_dirs(output_root_dir)
        trainer.printlog = True
        trainer.evaluate_test_set(
            output_root_dir=output_root_dir, epoch=0)
    else:
        trainer.train()


def override_data_setting(config, dataset_config):
    new_config = copy(config)
    # override all data_loader settings with dataset_config
    for key in dataset_config.keys():
        if key == 'name':
            continue
        if key in new_config.keys():
            logger.info(f'Overriding "{key}" of config by given dataset_config')
            del new_config[key]
        new_config[key] = dataset_config[key]

    # add dataset name on experiment name
    new_config['name'] = new_config['name'] + '_' + dataset_config['name']
    return new_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--dataset_config', default=None, type=str,
                        help='the dataset config json file (will '
                             'override the train/val/test data settings in config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-p', '--pretrained_path', default=None, type=str,
                        help='pretrained checkpoint path (default: None)')
    parser.add_argument('-od', '--output_root_dir', default=None, type=str,
                        help='Output root directory for the test mode (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.dataset_config is not None:
        dataset_config = json.load(open(args.dataset_config))
        config = override_data_setting(config, dataset_config)

    main(config, args.resume, args.output_root_dir, args.pretrained_path)
