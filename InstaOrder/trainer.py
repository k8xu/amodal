import os
import cv2
import time
import time
import numpy as np

import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import models
import utils
import datasets
import inference as infer
import pdb
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import random


class Trainer(object):
    def __init__(self, args):
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.

        # get rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        if self.rank == 0:
            # logger
            self.tb_logger = None
            self.wb_logger = None
            now = datetime.now().strftime('%m-%d-%H-%M')
            if args.model['algo'] == "InstaDepthNet_od" or args.model['algo'] == "InstaDepthNet_d":
                self.run_name = f'{args.data["dataset"]}_{args.trainer["exp_name"]}_{args.data["patch_or_image"]}' \
                                f'_ovl{args.model["overlap_weight"]}_dist{args.model["distinct_weight"]}' \
                                f'_dorder{args.model["dorder_weight"]}_smooth{args.model["smooth_weight"]}' \
                                f'_occ{args.model["occ_order_weight"]}_{now}'
            elif args.data['trainval_dataset'] == "SupDepthOrderDataset":
                self.run_name = f'{args.data["dataset"]}_{args.trainer["exp_name"]}_{args.data["patch_or_image"]}' \
                                f'_w{args.model["overlap_weight"]}_{now}'

            else:
                self.run_name = f'{args.data["dataset"]}_{args.trainer["exp_name"]}_{args.data["patch_or_image"]}_{now}'

            if args.trainer['wandb']:
                import wandb
                wandb.init(project="InstaOrder", name=f'Train/{self.run_name}', config=args)
                self.wb_logger = wandb

            elif args.trainer['tensorboard']:
                try:
                    from tensorboardX import SummaryWriter
                except:
                    raise Exception("Please switch off \"tensorboard\" "
                                    "in your config file if you do not "
                                    "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter('{}/events'.format(self.folder2save))

            # mkdir path
            BASE_DIR = args.data['base_dir']
            self.folder2save = f"{BASE_DIR}/data/out/InstaOrder/{self.run_name}"
            os.makedirs('{}/logs'.format(self.folder2save), exist_ok=True)
            os.makedirs('{}/checkpoints'.format(self.folder2save), exist_ok=True)

            if args.validate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_offline_val.txt'.format(self.folder2save))
            else:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_train.txt'.format(self.folder2save))

        # create model
        self.model = models.__dict__[args.model['algo']](args.model, load_pretrain=args.load_pretrain, dist_model=True)

        self.start_iter = 0
        if args.load_model is not None:
            self.model.load_state(args.load_model, Iter=None, resume=True)
            self.start_iter = int(args.load_model.split('iter_')[-1].split('.')[0])

        self.curr_step = self.start_iter

        # lr scheduler & datasets
        trainval_class = datasets.__dict__[args.data['trainval_dataset']]

        if not args.validate:  # train
            self.lr_scheduler = utils.StepLRScheduler(
                self.model.optim,
                args.model['lr_steps'],
                args.model['lr_mults'],
                args.model['lr'],
                args.model['warmup_lr'],
                args.model['warmup_steps'],
                last_iter=self.start_iter - 1)

            train_dataset = trainval_class(args.data, 'train', args.model["algo"])
            train_sampler = utils.DistributedGivenIterationSampler(
                train_dataset,
                args.model['total_iter'],
                args.data['batch_size'],
                last_iter=self.start_iter - 1)
            self.train_loader = DataLoader(train_dataset,
                                           batch_size=args.data['batch_size'],
                                           shuffle=False,
                                           num_workers=args.data['workers'],
                                           pin_memory=False,
                                           sampler=train_sampler)

        val_dataset = trainval_class(args.data, 'val', args.model["algo"])
        val_sampler = utils.DistributedSequentialSampler(val_dataset)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.data['batch_size_val'],
            shuffle=False,
            num_workers=args.data['workers'],
            pin_memory=False,
            sampler=val_sampler)

        self.args = args

    def run(self):

        # offline validate
        if self.args.validate:
            self.validate('off_val')
            return
        if self.args.trainer['initial_val']:
            self.validate('on_val')

        # train
        self.train()

    def train(self):
        btime_rec = utils.AverageMeter(10)
        dtime_rec = utils.AverageMeter(10)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('train')
        end = time.time()

        if self.rank == 0:
            self.num_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            self.logger.info(f"#parameters: {self.num_params}")
            self.wb_logger.log({'#parameters': self.num_params}, step=0)

        for i, inputs in enumerate(self.train_loader):

            self.curr_step = self.start_iter + i
            self.lr_scheduler.step(self.curr_step)
            curr_lr = self.lr_scheduler.get_lr()[0]

            # measure data loading time
            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)
            loss_dict = self.model.step()

            loss_dict_to_log = {}
            if len(loss_dict) == 2:
                loss_dict_to_log, loss_dict = loss_dict

            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())

            btime_rec.update(time.time() - end)
            end = time.time()

            self.curr_step += 1

            # logging
            if self.rank == 0 and self.curr_step % self.args.trainer['print_freq'] == 0:
                loss_str = ""
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('lr', curr_lr, self.curr_step)
                if self.wb_logger is not None:
                    self.wb_logger.log({'lr': curr_lr}, step=self.curr_step)
                for k in recorder.keys():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('train_{}'.format(k), recorder[k].avg, self.curr_step)
                    if self.wb_logger is not None:
                        self.wb_logger.log({f"train/{k}": recorder[k].avg}, step=self.curr_step)
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(k, loss=recorder[k])

                for kk in loss_dict_to_log.keys():
                    if self.wb_logger is not None and kk is not None:
                        self.wb_logger.log({f"train/{kk}": loss_dict_to_log[kk]}, step=self.curr_step)

                self.logger.info(
                    'Iter: [{0}/{1}]\t'.format(self.curr_step, len(self.train_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=dtime_rec) + loss_str +
                    'lr {lr:.2g}'.format(lr=curr_lr))

            # save
            if (self.rank == 0 and
                    (self.curr_step % self.args.trainer['save_freq'] == 0 or
                     self.curr_step == self.args.model['total_iter'])):
                self.model.save_state(
                    "{}/checkpoints".format(self.folder2save),
                    self.curr_step)

            # validate
            if (self.curr_step % self.args.trainer['val_freq'] == 0 or self.curr_step == self.args.model['total_iter']):
                self.validate('on_val')

    def validate(self, phase):
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('eval')
        end = time.time()
        for i, inputs in enumerate(self.val_loader):
            if ('val_iter' in self.args.trainer and
                    self.args.trainer['val_iter'] != -1 and
                    i == self.args.trainer['val_iter']):
                break

            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)

            loss_dict_to_log, loss_dict = self.model.forward_only()

            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())

            btime_rec.update(time.time() - end)
            end = time.time()

        # logging
        if self.rank == 0:
            loss_str = ""
            for k in recorder.keys():
                if self.tb_logger is not None and phase == 'on_val':
                    self.tb_logger.add_scalar('val_{}'.format(k), recorder[k].avg, self.curr_step)
                if self.wb_logger is not None and phase == 'on_val':
                    self.wb_logger.log({f"val/{k}": recorder[k].avg}, step=self.curr_step)
                loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(k, loss=recorder[k])

            for kk in loss_dict_to_log.keys():
                if self.wb_logger is not None and phase == 'on_val' and kk is not None:
                    self.wb_logger.log({f"val/{kk}": loss_dict_to_log[kk]}, step=self.curr_step)

            self.logger.info(
                'Validation Iter: [{0}]\t'.format(self.curr_step) +
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=btime_rec) +
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    data_time=dtime_rec) + loss_str)

        self.model.switch_to('train')
