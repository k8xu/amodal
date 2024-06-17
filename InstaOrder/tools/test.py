import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
import copy

sys.path.append('.')
from datasets import reader
import models
import inference as infer
import utils
from tqdm import tqdm
from utils.visualize_utils import draw_graph, get_mid_top_from_masks, put_instance_mask_and_ID

import matplotlib.pyplot as plt
import collections
from midas.midas_net import MidasNet
import torch

COLORS = [
    [252, 15, 15],  # Red
    [252, 165, 15],  # orange
    [252, 232, 15],  # yellow
    [14, 227, 39],  # light green
    [10, 138, 37],  # dark green
    [9, 219, 216],  # light blue
    [9, 111, 219],  # dark blue
    [185, 90, 232],  # light purple
    [201, 40, 175],  # purple
    [245, 49, 166]  # pink
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--load_model', required=True, type=str)
    parser.add_argument('--order_method', type=str)
    parser.add_argument('--order_th', default=0.1, type=float)
    parser.add_argument('--amodal_th', default=0.2, type=float)
    parser.add_argument('--test_num', default=-1, type=int)
    parser.add_argument('--pairs', default='all', type=str)
    parser.add_argument('--disp_select_method', default='', type=str)
    parser.add_argument('--save_pngs', default=1, type=int)
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
        if not args.order_method:
            self.args.order_method = self.args.model['algo']

        self.curr_step = 0
        if "pcnet_m.pth.tar" in self.args.load_model:
            self.ckpt_name = f"{self.args.load_model.split('/')[-1]}"
            self.model_name = self.ckpt_name

        elif self.args.model['algo'] == "midas_pretrained":
            self.model_name = f"{self.args.data['dataset']}_{self.args.model['algo']}_{self.args.disp_select_method}"
            self.ckpt_name = f"{self.model_name}"

        else:
            self.model_name = f'{self.args.load_model.split("/checkpoints/")[0].split("/")[-1]}'
            self.ckpt_name = f"{self.model_name}_{self.args.disp_select_method}"

        assert self.args.pairs in ["all", "nbor"]

        # set logger
        self.wb_logger = None
        if args.trainer['wandb']:
            import wandb
            self.run_name = self.ckpt_name
            wandb.init(project="InstaOrder", name=f'Test/{self.run_name}', config=args)

            self.wb_logger = wandb

        ### add
        self.BASE_DIR = self.args.data['base_dir']
        self.folder2save = f"{self.BASE_DIR}/data/out_pngs/{self.args.data['dataset']}/{self.ckpt_name}"
        os.makedirs('{}/logs'.format(self.folder2save), exist_ok=True)
        self.logger = utils.create_logger(
            'global_logger',
            '{}/logs/log_eval.txt'.format(self.folder2save))
        self.prepare_data()

    def prepare_data(self):
        config = self.args.data
        dataset = config['dataset']
        # self.COLORS = None
        self.COLORS = COLORS
        self.data_root = self.args.data['val_image_root']
        self.gt_ordering = "ann"

        if dataset == 'COCOA':
            self.args.annotation = self.args.data['val_annot_file']
            self.data_reader = reader.COCOADataset(self.args.data['val_annot_file'])
        elif dataset == 'InstaOrder':
            self.data_reader = reader.InstaOrderDataset(self.args.data['val_annot_file'])
        elif dataset == "KINS":
            self.data_reader = reader.KINSLVISDataset(dataset, self.args.data['val_annot_file'])
            self.gt_ordering = "man"

        self.data_length = self.data_reader.get_image_length()
        self.dataset = dataset
        if self.args.test_num != -1:
            self.data_length = self.args.test_num

    def prepare_model(self):

        # create model
        if self.args.model['algo'] == "midas_pretrained":
            self.model = MidasNet(self.args.load_model, non_negative=True)
            self.model.cuda()
            self.model.train()
            self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f"#parameters: {self.num_params}")

            self.model.eval()
        else:
            self.model = models.__dict__[self.args.model['algo']](self.args.model, dist_model=False)
            self.model.load_state(self.args.load_model)
            self.model.switch_to('train')
            self.num_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            self.logger.info(f"#parameters: {self.num_params}")
            self.model.switch_to('eval')

    def expand_bbox(self, bboxes):
        new_bboxes = []
        for bbox in bboxes:
            centerx = bbox[0] + bbox[2] / 2.
            centery = bbox[1] + bbox[3] / 2.
            size = max([np.sqrt(bbox[2] * bbox[3] * self.args.data['enlarge_box']), bbox[2] * 1.1, bbox[3] * 1.1])
            new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
            new_bboxes.append(new_bbox)
        return np.array(new_bboxes)

    def run(self):
        self.prepare_model()
        # self.save_masks()

        if self.args.data['trainval_dataset'] == "SupDepthOrderDataset":
            self.eval_depth_order()
        elif self.args.data['trainval_dataset'] in ["SupOcclusionOrderDataset", "PartialCompDataset"]:
            self.eval_occ_order()
        elif self.args.data['trainval_dataset'] == "SupDepthOccOrderDataset":
            self.eval_occ_depth_order()

    def save_masks(self):
        for i in tqdm(range(self.data_length), total=self.data_length):
            modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(i, with_gt=True)
            image = np.array(Image.open(os.path.join(self.data_root, image_fn)).convert('RGB'))
            folder2save = f"{self.BASE_DIR}/data/{self.args.data['dataset']}/masks"
            os.makedirs(folder2save, exist_ok=True)
            plt.imsave(f"{folder2save}/{image_fn.split('_')[-1].split('.')[0]}_rgb.png", image)
            mid_tops = get_mid_top_from_masks(modal)
            I_masked_all = put_instance_mask_and_ID(image, modal, mid_tops, self.COLORS, category)
            plt.imsave(f"{folder2save}/{image_fn.split('_')[-1].split('.')[0]}_masks.png", I_masked_all)

    def eval_occ_depth_order(self):
        recall_list, precision_list, f1_list = [], [], []

        WHDR_dict_per_ovls_eqs = collections.defaultdict(list)  # WHDR_all, WHDR_eq, WHDR_neq
        self.logger.info(f"WHDR\t\t distinct \t\t overlap \t\t all (all) \t\t ")

        for i in tqdm(range(self.data_length), total=self.data_length):
            modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(i, with_gt=True)

            # data
            image = np.array(Image.open(os.path.join(self.data_root, image_fn)).convert('RGB'))
            bboxes = self.expand_bbox(bboxes)

            # gt order
            if self.dataset == 'InstaOrder':
                gt_depth_order_ovl_count = self.data_reader.get_gt_ordering(i, 'depth')
                gt_depth_order, gt_overlap_matrix, gt_count_matrix = gt_depth_order_ovl_count
                gt_occ_order = self.data_reader.get_gt_ordering(i, 'occlusion', self.args.data['remove_occ_bidirec'])

            if self.args.order_method == "InstaOrderNet_od" or self.args.order_method == "InstaDepthNet_od":
                pred_occ_order, pred_depth_order = infer.infer_order_sup_occ_depth(
                    self.model, image, modal, bboxes, pairs=self.args.pairs, method=self.args.order_method,
                    patch_or_image=self.args.data['patch_or_image'], input_size=self.args.data['input_size'],
                    disp_select_method=self.args.disp_select_method)
            else:
                raise Exception('No such order method: {}'.format(self.args.order_method))
            # compute depth order score
            whdr_per_ovls_eqs = infer.eval_depth_order_whdr(pred_depth_order, copy.deepcopy(gt_depth_order_ovl_count))
            for ovl_eq_str in whdr_per_ovls_eqs.keys():
                WHDR_dict_per_ovls_eqs[ovl_eq_str].append(whdr_per_ovls_eqs[ovl_eq_str][0])

            # compute occlusion order score
            recall, precision, f1 = infer.eval_order_recall_precision_f1(pred_occ_order, gt_occ_order, self.args.zd)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)

            ### add
            self.logger.info(
                f"[{image_fn}]\t{whdr_per_ovls_eqs['ovlX_all'][0]:.3f} | {whdr_per_ovls_eqs['ovlO_all'][0]:.3f} | {whdr_per_ovls_eqs['ovlOX_all'][0]:.3f}"
            )
            self.logger.info(f"\t\t\trecall={recall:.3f} / precision={precision:.3f} / f1={f1:.3f}")

            if self.args.save_pngs == 1:
                folder2save = f"{self.BASE_DIR}/data/out_pngs/{self.args.data['dataset']}/{self.ckpt_name}"
                save_list = ['rgb', 'mask', 'depth_order', 'occ_order']
                dict_save = {}
                for save_str in save_list:
                    dict_save[save_str] = f"{folder2save}/{save_str}/"
                    os.makedirs(dict_save[save_str], exist_ok=True)
                # save rgb
                img_name = image_fn.split('_')[-1].split('.')[0]
                plt.imsave(f"{dict_save['rgb']}/{img_name}.png", image)
                # save mask
                mid_tops = get_mid_top_from_masks(modal)
                I_masked_all = put_instance_mask_and_ID(image, modal, mid_tops, self.COLORS)
                plt.imsave(f"{dict_save['mask']}/{img_name}.png", I_masked_all)
                # save gt-pred depth order
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                draw_graph(gt_depth_order, overlap_matrix=gt_overlap_matrix)
                plt.subplot(122)
                draw_graph(pred_depth_order)
                plt.savefig(
                    f"{dict_save['depth_order']}/{img_name}_WHDR_ovlX_all_ovlO={whdr_per_ovls_eqs['ovlX_all'][0]:.1f}-{whdr_per_ovls_eqs['ovlOX_all'][0]:.1f}-{whdr_per_ovls_eqs['ovlO_all'][0]:.1f}.png")

                plt.close('all')

                # save gt-pred occlusion order
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                draw_graph(gt_occ_order)
                plt.subplot(122)
                draw_graph(pred_occ_order)
                plt.savefig(f"{dict_save['occ_order']}/{img_name}_f1={f1:.1f}_P{self.args.pairs}.png")
                plt.close('all')

        self.logger.info("\n\n[MEAN WHDR]")
        for ovl_eq_str in WHDR_dict_per_ovls_eqs.keys():
            whdr_arr = np.array(WHDR_dict_per_ovls_eqs[ovl_eq_str])
            mask_valid = whdr_arr != -1
            mean = whdr_arr[mask_valid].sum() / (len(whdr_arr[mask_valid]) + 1e-6)
            ovl_str, eq_str = ovl_eq_str.split('_')
            self.wb_logger.log({f"val_{ovl_str}/WHDR_{eq_str}": mean}, step=self.curr_step)
            self.logger.info(f"{ovl_eq_str}: {mean}")

        mean_recall = sum(recall_list) / len(recall_list)
        mean_precision = sum(precision_list) / len(precision_list)
        mean_f1 = sum(f1_list) / len(f1_list)
        self.wb_logger.log({"val/recall": mean_recall,
                            "val/precision": mean_precision,
                            "val/f1": mean_f1,
                            "val/num_test_images": len(recall_list),
                            "val/iter": self.curr_step}, step=self.curr_step)

        self.logger.info(self.args.order_method)
        self.logger.info(f"\n\n[AVERAGE] recall={mean_recall:.3f} / precision={mean_precision:.3f} / f1={mean_f1:.3f}")

        self.wb_logger.log({"val/num_test_images": self.data_length,
                            "val/iter": self.curr_step}, step=self.curr_step)

    def eval_depth_order(self):
        WHDR_dict_per_ovls_eqs = collections.defaultdict(list)  # WHDR_all, WHDR_eq, WHDR_neq
        self.logger.info(f"WHDR\t\t\t distinct \t\t\t overlap \t\t\t all (all | neq | eq)")

        for i in tqdm(range(self.data_length), total=self.data_length):
            modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(i, with_gt=True)
            if self.args.data['use_category']:
                modal = modal * category[:, None, None]
            # data
            image = np.array(Image.open(os.path.join(self.data_root, image_fn)).convert('RGB'))
            bboxes = self.expand_bbox(bboxes)
            # gt order
            if self.dataset == 'InstaOrder':
                gt_depth_order_ovl_count = self.data_reader.get_gt_ordering(i, 'depth', rm_overlap=self.args.data[
                    'remove_depth_overlap'])
                gt_depth_order, gt_overlap_matrix, gt_count_matrix = gt_depth_order_ovl_count

            # infer order
            if self.args.order_method == 'area':
                closer = 'larger'
                if args.data['dataset'] == 'COCOA' or args.data['dataset'] == 'InstaOrder':
                    # closer = 'smaller'
                    closer = 'larger'

                pred_depth_order = infer.infer_depth_order_area(modal, closer=closer)

            elif self.args.order_method == 'yaxis':
                closer = 'higher'
                if args.data['dataset'] == 'COCOA' or args.data['dataset'] == 'InstaOrder':
                    closer = 'lower'
                pred_depth_order = infer.infer_depth_order_yaxis(modal, closer=closer)


            elif self.args.order_method in ['InstaOrderNet_d', 'midas_pretrained', 'InstaDepthNet_d']:
                pred_depth_order, disp_clipped = infer.infer_order_sup_depth(
                    self.model, image, modal, bboxes, pairs=self.args.pairs, method=self.args.order_method,
                    patch_or_image=self.args.data['patch_or_image'], input_size=self.args.data['input_size'],
                    disp_select_method=self.args.disp_select_method, use_rgb=self.args.model['use_rgb'])
            else:
                raise Exception('No such order method: {}'.format(self.args.order_method))

            whdr_per_ovls_eqs = infer.eval_depth_order_whdr(pred_depth_order, copy.deepcopy(gt_depth_order_ovl_count))
            for ovl_eq_str in whdr_per_ovls_eqs.keys():
                WHDR_dict_per_ovls_eqs[ovl_eq_str].append(whdr_per_ovls_eqs[ovl_eq_str][0])

            ### log
            self.logger.info(
                f"[{image_fn}]\t{whdr_per_ovls_eqs['ovlX_all'][0]:.3f} "  # | {whdr_per_ovls_eqs['ovlX_neq'][0]:.3f} | {whdr_per_ovls_eqs['ovlX_eq'][0]:.3f}"
                f" | {whdr_per_ovls_eqs['ovlO_all'][0]:.3f} "  # | {whdr_per_ovls_eqs['ovlO_neq'][0]:.3f} | {whdr_per_ovls_eqs['ovlO_eq'][0]:.3f}"
                f" | {whdr_per_ovls_eqs['ovlOX_all'][0]:.3f} "
                # | {whdr_per_ovls_eqs['ovlOX_neq'][0]:.3f} | {whdr_per_ovls_eqs['ovlOX_eq'][0]:.3f}"
            )

            if self.args.save_pngs == 1:
                folder2save = f"{self.BASE_DIR}/data/out_pngs/{self.args.data['dataset']}/{self.ckpt_name}"
                save_list = ['rgb', 'mask', 'order']
                dict_save = {}
                for save_str in save_list:
                    dict_save[save_str] = f"{folder2save}/{save_str}/"
                    os.makedirs(dict_save[save_str], exist_ok=True)
                # save rgb
                img_name = image_fn.split('_')[-1].split('.')[0]
                plt.imsave(f"{dict_save['rgb']}/{img_name}.png", image)
                # save mask
                mid_tops = get_mid_top_from_masks(modal)
                I_masked_all = put_instance_mask_and_ID(image, modal, mid_tops, self.COLORS)
                plt.imsave(f"{dict_save['mask']}/{img_name}.png", I_masked_all)
                # save gt-pred order
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                draw_graph(gt_depth_order, overlap_matrix=gt_overlap_matrix)

                plt.subplot(122)
                draw_graph(pred_depth_order)
                plt.savefig(
                    f"{dict_save['order']}/{img_name}_WHDR_ovlX_all_ovlO={whdr_per_ovls_eqs['ovlX_all'][0]:.1f}-{whdr_per_ovls_eqs['ovlOX_all'][0]:.1f}-{whdr_per_ovls_eqs['ovlO_all'][0]:.1f}.png")
                plt.close('all')

                if disp_clipped != None:
                    disp_dir = f"{folder2save}/disp/"
                    os.makedirs(disp_dir, exist_ok=True)
                    disp_clipped = torch.nn.functional.interpolate(disp_clipped[None, None, :, :], size=image.shape[:2],
                                                                   mode="bicubic", align_corners=False)
                    plt.imsave(f"{disp_dir}/{img_name}.png", disp_clipped.squeeze().cpu().detach().numpy(),
                               cmap='inferno')

                # mask_dir = f"{folder2save}/mask_black/"
                # print(mask_dir)
                # os.makedirs(mask_dir, exist_ok=True)
                # for mm in range(modal.shape[0]):
                #     plt.imsave(f"{mask_dir}/{img_name}-{mm}.png", modal[mm], cmap='gray')

        self.logger.info("\n\n[MEAN WHDR]")
        for ovl_eq_str in WHDR_dict_per_ovls_eqs.keys():
            whdr_arr = np.array(WHDR_dict_per_ovls_eqs[ovl_eq_str])
            mask_valid = whdr_arr != -1
            mean = whdr_arr[mask_valid].sum() / (len(whdr_arr[mask_valid]) + 1e-6)
            ovl_str, eq_str = ovl_eq_str.split('_')
            self.wb_logger.log({f"val_{ovl_str}/WHDR_{eq_str}": mean}, step=self.curr_step)
            if eq_str == "all":
                self.logger.info(f"{ovl_eq_str}: {mean}")

        self.wb_logger.log({"val/num_test_images": self.data_length,
                            "val/iter": self.curr_step}, step=self.curr_step)

        if self.curr_step == 0:
            for ovl_eq_str in WHDR_dict_per_ovls_eqs.keys():
                whdr_arr = np.array(WHDR_dict_per_ovls_eqs[ovl_eq_str])
                mask_valid = whdr_arr != -1
                mean = whdr_arr[mask_valid].sum() / (len(whdr_arr[mask_valid]) + 1e-6)
                ovl_str, eq_str = ovl_eq_str.split('_')
                self.wb_logger.log({f"val_{ovl_str}/WHDR_{eq_str}": mean}, step=100000)
                self.logger.info(f"{ovl_eq_str}: {mean}")

    def eval_occ_order(self):
        recall_list, precision_list, f1_list = [], [], []
        for i in tqdm(range(self.data_length), total=self.data_length):
            modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(i, with_gt=True)
            if self.args.data['use_category']:
                modal = modal * category[:, None, None]

            # data
            image = np.array(Image.open(os.path.join(self.data_root, image_fn)).convert('RGB'))
            bboxes = self.expand_bbox(bboxes)

            # gt order
            if self.dataset == 'InstaOrder':
                gt_occ_order = self.data_reader.get_gt_ordering(i, 'occlusion', self.args.data['remove_occ_bidirec'])
            else:
                gt_occ_order = self.data_reader.get_gt_ordering(i) if self.gt_ordering == "ann" \
                    else infer.infer_gt_order(modal, amodal_gt)

            # infer order
            if self.args.order_method == 'area':
                occluder = 'larger'
                if args.data['dataset'] == 'COCOA' or args.data['dataset'] == 'InstaOrder':
                    # occluder = 'smaller'
                    occluder = 'larger'

                pred_occ_order = infer.infer_occ_order_area(modal, occluder=occluder)

            elif self.args.order_method == 'yaxis':
                occluder = 'higher'
                if args.data['dataset'] == 'COCOA' or args.data['dataset'] == 'InstaOrder':
                    occluder = 'lower'
                pred_occ_order = infer.infer_occ_order_yaxis(modal, occluder=occluder)

            elif self.args.order_method == 'PartialCompletionMask':
                pred_occ_order = infer.infer_order(
                    self.model, image, modal, category, bboxes, pairs=self.args.pairs,
                    use_rgb=self.args.model['use_rgb'], th=self.args.order_th, dilate_kernel=0,
                    input_size=256, min_input_size=16, interp='nearest', debug_info=False)

            elif self.args.order_method in ['InstaOrderNet_o', 'OrderNet']:
                pred_occ_order = infer.infer_order_sup_occ(
                    self.model, image, modal, bboxes, pairs=self.args.pairs, method=self.args.order_method,
                    patch_or_image=self.args.data['patch_or_image'], input_size=self.args.data['input_size'],
                    use_rgb=self.args.model['use_rgb'])
            else:
                raise Exception('No such order method: {}'.format(self.args.order_method))

            recall, precision, f1 = infer.eval_order_recall_precision_f1(pred_occ_order, gt_occ_order, self.args.zd)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
            self.logger.info(f"[{image_fn}] recall={recall:.3f} / precision={precision:.3f} / f1={f1:.3f}")

            if self.args.save_pngs == 1:
                folder2save = f"{self.BASE_DIR}/data/out_pngs/{self.args.data['dataset']}/{self.ckpt_name}"
                save_list = ['rgb', 'mask', 'order']
                dict_save = {}
                for save_str in save_list:
                    dict_save[save_str] = f"{folder2save}/{save_str}/"
                    os.makedirs(dict_save[save_str], exist_ok=True)
                # save rgb
                img_name = image_fn.split('_')[-1].split('.')[0]
                plt.imsave(f"{dict_save['rgb']}/{img_name}.png", image)
                # save mask
                mid_tops = get_mid_top_from_masks(modal)
                I_masked_all = put_instance_mask_and_ID(image, modal, mid_tops, self.COLORS)
                plt.imsave(f"{dict_save['mask']}/{img_name}.png", I_masked_all)
                # save gt-pred order
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                draw_graph(gt_occ_order)
                plt.subplot(122)
                draw_graph(pred_occ_order)
                plt.savefig(f"{dict_save['order']}/{img_name}_f1={f1:.1f}_P{self.args.pairs}.png")
                plt.close('all')

        mean_recall = sum(recall_list) / len(recall_list)
        mean_precision = sum(precision_list) / len(precision_list)
        mean_f1 = sum(f1_list) / len(f1_list)
        self.logger.info(self.args.order_method)
        self.logger.info(f"\n\n[AVERAGE] recall={mean_recall:.3f} / precision={mean_precision:.3f} / f1={mean_f1:.3f}")

        if self.wb_logger is not None:
            self.wb_logger.log({"val/recall": mean_recall,
                                "val/precision": mean_precision,
                                "val/f1": mean_f1,
                                "val/num_test_images": len(recall_list),
                                "val/iter": self.curr_step}, step=self.curr_step)
            if "pcnet_m.pth.tar" in self.args.load_model:
                self.wb_logger.log({"val/recall": mean_recall,
                                    "val/precision": mean_precision,
                                    "val/f1": mean_f1,
                                    "val/num_test_images": len(recall_list),
                                    "val/iter": self.curr_step}, step=40000)


if __name__ == "__main__":
    args = parse_args()
    main(args)
