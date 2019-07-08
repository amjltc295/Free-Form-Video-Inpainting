import os
import math
import json
import logging
import datetime
import torch
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self, model, losses, metrics, optimizer_g,
        optimizer_d_s, optimizer_d_t,
        resume, config,
        train_logger=None,
        pretrained_path=None,
    ):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)

        self.losses = losses
        self.metrics = metrics
        self.optimizer_g = optimizer_g
        self.optimizer_d_s = optimizer_d_s
        self.optimizer_d_t = optimizer_d_t

        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']

        # Set pretrained_load_strict to False to load model without strict state name matching
        # It's useful when pretrained model without GAN but we want to use GAN for this time
        self.pretrained_load_strict = config['trainer']['pretrained_load_strict']

        self.train_logger = train_logger

        # configuration to monitor model performance and save best
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], config['name'], start_time)
        # setup visualization writer instance
        writer_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, config['visualization']['tensorboardX'])

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)
        elif pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # put model into DataParallel module only after the checkpoint is loaded
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            msg = (f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, "
                   f"but only {n_gpu} are available on this machine.")
            self.logger.warning(msg)
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            monitor_value = None
            if self.monitor_mode != 'off':
                try:
                    if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) or\
                            (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                        self.monitor_best = log[self.monitor]
                        best = True
                    monitor_value = log[self.monitor]

                except KeyError:
                    if epoch == 1:
                        msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor)\
                            + "for performance monitoring. model_best checkpoint won\'t be updated."
                        self.logger.warning(msg)
            if epoch % self.save_freq == 0 or best:
                self._save_checkpoint(epoch, save_best=best, monitor_value=monitor_value)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False, monitor_value=None):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__

        # assure that we save the model state without DataParallel module
        if isinstance(self.model, torch.nn.DataParallel):
            # get the original state out from DataParallel module
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': model_state,
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d_s': self.optimizer_d_s.state_dict() if self.optimizer_d_s is not None else None,
            'optimizer_d_t': self.optimizer_d_t.state_dict() if self.optimizer_d_t is not None else None,
            'monitor_best': self.monitor_best,
            'config': self.config
        }

        best_str = '-best-so-far' if save_best else ''
        monitor_str = f'-{self.monitor}{monitor_value:.4f}' if monitor_value is not None else ''
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}{monitor_str}{best_str}.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        if self.optimizer_d_s is not None:
            self.optimizer_d_s.load_state_dict(checkpoint['optimizer_d_s'])
        if self.optimizer_d_t is not None:
            self.optimizer_d_t.load_state_dict(checkpoint['optimizer_d_t'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def _load_pretrained(self, pretrained_path):
        self.logger.info(f"Loading pretrained checkpoint from {pretrained_path}")
        checkpoint = torch.load(pretrained_path)
        pretrained_state = checkpoint['state_dict']
        if self.pretrained_load_strict:
            self.model.load_state_dict(pretrained_state)
        else:
            current_state = self.model.state_dict()
            lack_modules = set([
                k.split('.')[0]
                for k in current_state.keys()
                if k not in pretrained_state.keys()
            ])
            self.logger.info(f"Allowing lack of submodules for pretrained model.")
            self.logger.info(f"Submodule(s) not in pretrained model but in current model: {lack_modules}")
            redundant_modules = set([
                k.split('.')[0]
                for k in pretrained_state.keys()
                if k not in current_state.keys()
            ])
            self.logger.info(f"Submodule(s) not in current model but in pretraired model: {set(redundant_modules)}")

            # used_pretrained_state = {k: v for k, v in pretrained_state.items() if k in current_state}
            used_pretrained_state = {}
            prefixs = [
                'generator.coarse_net.upsample_module.',
                'generator.coarse_net.downsample_module.',
            ]
            for k, v in pretrained_state.items():
                if k in current_state:
                    used_pretrained_state[k] = v
                else:
                    # Backward compatible
                    for prefix in prefixs:
                        new_key = prefix + k
                        if new_key in current_state:
                            self.logger.warning(f"Load key to new model: {k} -> {new_key}")
                            used_pretrained_state[new_key] = v
            current_state.update(used_pretrained_state)
            self.model.load_state_dict(current_state)
        self.logger.info("Pretrained checkpoint loaded")
