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
from utils.common_utils import disp_to_depth
import cv2
from utils.common_utils import UnNormalize

BASE_DIR = ''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--load_model', required=True, type=str)
    parser.add_argument('--test_num', default=-1, type=int)
    parser.add_argument('--pairs', default='all', type=str)
    parser.add_argument('--disp_select_method', default='', type=str)
    parser.add_argument('--save_pngs', default=0, type=int)
    parser.add_argument('--zd', default=1, type=int, help="zero_division, 1 is correct")
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config) as f:
        config = yaml.load(f)

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
        if self.args.model['algo'] == "midas_pretrained":
            self.model_name = f"{self.args.data['dataset']}_{self.args.model['algo']}_{self.args.disp_select_method}"
            self.curr_step = 0
            self.ckpt_name = f"{self.model_name}"

        else:
            self.model_name = f'{self.args.data["dataset"]}_{self.args.load_model.split("/checkpoints/")[0].split("/")[-1]}'
            self.curr_step = int(self.args.load_model.split("/")[-1].split(".")[0].split("_")[-1])
            self.ckpt_name = f"{self.model_name}_iter{self.curr_step}_{self.args.disp_select_method}"

        # print(f"using gt_ordering of {self.gt_ordering}")
        assert self.args.pairs in ["all", "nbor"]

        # set logger
        self.wb_logger = None
        if args.trainer['wandb']:
            import wandb
            # self.run_name = f'{self.model_name}_pairs({self.args.pairs})_num{args.test_num}'
            self.run_name = self.ckpt_name
            wandb.init(project="InstaOrder", name=f'Test/{self.run_name}', config=args)

            self.wb_logger = wandb

        self.prepare_data()

        self.convert = 'median'
        # self.convert = 'scale-shift'
        self.folder2save = f"{BASE_DIR}/data/out_pngs/{self.args.data['dataset']}/{self.ckpt_name}_{self.convert}"
        os.makedirs(f"{self.folder2save}/rgb/", exist_ok=True)
        os.makedirs(f"{self.folder2save}/pred_disp/", exist_ok=True)
        os.makedirs(f"{self.folder2save}/gt_disp/", exist_ok=True)
        os.makedirs(f"{self.folder2save}/distribution/disp/", exist_ok=True)
        os.makedirs(f"{self.folder2save}/distribution/depth/", exist_ok=True)
        os.makedirs('{}/logs'.format(self.folder2save), exist_ok=True)
        self.logger = utils.create_logger(
            'global_logger',
            '{}/logs/log_eval.txt'.format(self.folder2save))

    def prepare_data(self):
        config = self.args.data
        self.dataset = config['dataset']
        self.data_root = self.args.data['val_image_root']

        if self.dataset == "kitti":
            test_data = reader.KITTIDataset(self.args)
            self.min_depth, self.max_depth = 1e-3, 80
        elif self.dataset == "nyu":
            test_data = reader.NYUDataset(self.args)
            self.min_depth, self.max_depth = 1e-3, 10

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
        # self.save_masks()
        self.eval_dense_depth()

    def compute_errors(self, gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        ddd = np.log(pred) - np.log(gt)
        silog = np.sqrt((ddd ** 2).mean() - (ddd.mean() ** 2))
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, silog

    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01],
        #                     [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def eval_dense_depth(self):
        errors = []
        missing_cnt = 0
        for image, img_name, depth_name in self.dataloader:
            image = image.cuda()

            # 1. dispairty prediction
            if self.args.model['algo'] == "midas_pretrained":
                pred_disp = self.model(image)
            elif self.args.model['algo'] == "InstaDepthNet_d":
                zero_arr = torch.zeros((1, 1, image.shape[2], image.shape[3])).cuda()
                pred_disp, _, _ = self.model.model(image, zero_arr, zero_arr)

            # 2. read gt depth
            gt_depth = cv2.imread(depth_name[0], -1)
            if gt_depth is None:
                missing_cnt += 1
                continue

            if self.dataset == 'kitti':
                gt_depth = gt_depth.astype(np.float32) / 256.0
                top_margin = int(gt_depth.shape[0] - 352)
                left_margin = int((gt_depth.shape[1] - 1216) / 2)
                gt_depth = gt_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                name_splitted = img_name[0].split('/')
                img_name_2_save = f"{name_splitted[-4]}_{name_splitted[-1][:-4]}"

            # 3. convert disparity to depth (as done in MonoDepth2)
            if self.convert == "median":
                pred_disp = pred_disp.squeeze().cpu().detach().numpy()

                norm_pred_disp = (pred_disp - pred_disp.min()) / pred_disp.max()
                pred_depth = 1 / (norm_pred_disp + 1e-3)

                mask_valid = (gt_depth >= self.min_depth) & (gt_depth <= self.max_depth)
                ratio = np.median(gt_depth[mask_valid]) / np.median(pred_depth[mask_valid])
                pred_depth *= ratio

            plt.hist(pred_depth[mask_valid], color='gray', edgecolor='black', bins=50)
            plt.title('Histogram of pred_depth[mask_valid]')
            plt.xlabel('depth')
            plt.ylabel('distribution')
            plt.savefig(f"{self.folder2save}/distribution/depth/{img_name_2_save}.png")
            plt.close('all')

            pred_depth[pred_depth < self.min_depth] = self.min_depth
            pred_depth[pred_depth > self.max_depth] = self.max_depth

            error = self.compute_errors(gt_depth[mask_valid], pred_depth[mask_valid])
            errors.append(error)
            print(f"[{img_name_2_save}], {error}")
            d1 = error[-4] * 100
            plt.imsave(f"{self.folder2save}/pred_disp/{img_name_2_save}_{d1:.2f}.png", pred_disp, cmap='inferno')
            gt_disp = 1 / (gt_depth + 1e-3)
            gt_disp[gt_depth == 0] = 0
            plt.imsave(f"{self.folder2save}/gt_disp/{img_name_2_save}.png", gt_disp, cmap='inferno')
            rgb = UnNormalize()(image)
            rgb[rgb > 1] = 1
            plt.imsave(f"{self.folder2save}/rgb/{img_name_2_save}.png", rgb[0].permute(1, 2, 0).cpu().detach().numpy())

            # plt.imsave(f"{self.folder2save}/pred_depth/{img_name_2_save}_{d1:.2f}.png", pred_depth, cmap='inferno')
            # plt.imsave(f"{self.folder2save}/gt_depth/{img_name_2_save}.png", gt_depth, cmap='inferno')

        # 4. calculate accuracy
        print(f"computed error on {len(errors)} / {missing_cnt} missing")
        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3", "silog"))
        print(("{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
