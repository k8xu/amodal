import argparse
import yaml
import os
import numpy as np
import sys

sys.path.append('.')
from datasets import reader
import models
import inference as infer
import utils
from tqdm import tqdm

import matplotlib.pyplot as plt
from midas.midas_net import MidasNet
import torch
from torch.utils.data import DataLoader
import cv2
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--load_model', required=True, type=str)
    parser.add_argument('--test_num', default=-1, type=int)
    parser.add_argument('--pairs', default='all', type=str)
    parser.add_argument('--save_pngs', default=0, type=int)
    parser.add_argument('--zd', default=1, type=int, help="zero_division, 1 is correct")
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config) as f:
        config = yaml.load(f)
    BASE_DIR = config['data']['base_dir']

    for k, v in config.items():
        for kkk, vvv in v.items():
            if type(vvv) == str and '/data/' in vvv:
                v[kkk] = f"{BASE_DIR}{vvv}"
        setattr(args, k, v)

    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    tester = Tester(args)
    tester.run()


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.curr_step = 0
        self.ckpt_name = f'{self.args.data["dataset"]}_{self.args.load_model}'
        assert self.args.pairs in ["all", "nbor"]

        # set logger
        self.wb_logger = None
        if args.trainer['wandb']:
            import wandb
            self.run_name = self.ckpt_name
            wandb.init(project="InstaOrder", name=f'Test/{self.run_name}', config=args)

            self.wb_logger = wandb

        self.prepare_data()

        self.folder2save = f"{self.args.data['base_dir']}/data/out_pngs/{self.args.data['dataset']}/{self.ckpt_name}"
        os.makedirs(f"{self.folder2save}/rgb/", exist_ok=True)
        os.makedirs(f"{self.folder2save}/rgb_AB/", exist_ok=True)
        os.makedirs(f"{self.folder2save}/pred_disp/", exist_ok=True)
        os.makedirs('{}/logs'.format(self.folder2save), exist_ok=True)
        self.logger = utils.create_logger(
            'global_logger',
            '{}/logs/log_eval.txt'.format(self.folder2save))

    def prepare_data(self):
        config = self.args.data
        self.dataset = config['dataset']
        self.data_root = self.args.data['val_image_root']

        if self.dataset == "diw":
            test_data = reader.DIWDataset(self.args)
        else:
            raise NotImplementedError
        self.dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    def prepare_model(self):
        # create model
        if self.args.model['algo'] == "midas_pretrained":
            self.model = MidasNet(self.args.load_model, non_negative=True)
            self.model.cuda()
            self.model.eval()
        else:
            self.model = models.__dict__[self.args.model['algo']](self.args.model, dist_model=False)
            self.model.load_state(self.args.load_model)
            self.model.switch_to('eval')

    def run(self):
        self.prepare_model()
        self.eval_ordinal_via_disp()

    def eval_ordinal_via_disp(self):
        errors = []
        correct, wrong = 0, 0
        error_name = []
        for image_orig, image, Ayx_Byx_ord, thumb_filename in self.dataloader:
            # print(len(errors))
            image_orig = np.array(image_orig[0])
            image = image.cuda()

            # 1. dispairty prediction
            if self.args.model['algo'] == "midas_pretrained":
                pred_disp = self.model(image)
            elif self.args.model['algo'] == "InstaDepthNet_d" or self.args.model['algo'] == "InstaDepthNet_od":
                zero_arr = torch.zeros((1, 1, image.shape[2], image.shape[3])).cuda()
                pred_disp, _, _ = self.model.model(image, zero_arr, zero_arr)

            pred_disp = torch.nn.functional.interpolate(
                pred_disp.unsqueeze(1),
                size=image_orig.shape[:2],
                mode="bilinear",
                align_corners=False,
            )  # 1,1,h,w
            pred_disp = pred_disp.squeeze().cpu().detach().numpy()  # h,w

            img_name_2_save = thumb_filename[0].split('/')[-1]
            Ayx, Byx, gt_depth_ord = Ayx_Byx_ord

            # for visualization
            masked_image = copy.deepcopy(image_orig)
            masked_image[Ayx[0]:Ayx[0] + 5, Ayx[1]:Ayx[1] + 5, :] = (255, 0, 0)  # A:red
            masked_image[Byx[0]:Byx[0] + 5, Byx[1]:Byx[1] + 5, :] = (0, 0, 255)  # B:blue

            dispA, dispB = pred_disp[Ayx[0], Ayx[1]], pred_disp[Byx[0], Byx[1]]

            # 2. compare ordinal
            # note that 'disparity ordinal' is opposite of 'depth ordinal'
            if dispA > dispB:
                pred_depth_ord = "<"
            elif dispA < dispB:
                pred_depth_ord = ">"
            elif dispA == dispB:
                pred_depth_ord = "="

            gt_depth_ord = gt_depth_ord[0]
            if gt_depth_ord == pred_depth_ord:
                correct += 1
                TF = f'T{gt_depth_ord}'
                errors.append(0)
            else:
                wrong += 1
                TF = f'F_gt{gt_depth_ord}_pred{pred_depth_ord}'
                errors.append(1)
                error_name.append(len(errors))

            img_name_2_save = len(errors)
            # print(f"[{img_name_2_save}], {TF}")
            plt.imsave(f"{self.folder2save}/pred_disp/{img_name_2_save}_{TF}.png", pred_disp, cmap='inferno')
            plt.imsave(f"{self.folder2save}/rgb/{img_name_2_save}.png", image_orig)
            plt.imsave(f"{self.folder2save}/rgb_AB/{img_name_2_save}.png", masked_image)

        # 3. calculate WHDR
        print(f"computed error on {len(errors)}")
        print(f"wrong/all = {sum(errors)}/{len(errors)}")
        print(f"WHDR = {sum(errors) / len(errors) * 100}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
