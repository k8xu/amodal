import numpy as np

try:
    import mc
except Exception:
    pass
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader
import inference as infer
import torch.distributed as dist


class SupOcclusionOrderDataset(Dataset):
    def __init__(self, config, phase, algo):
        self.algo = algo
        self.dataset = config['dataset']
        self.rm_bidirec = config['remove_occ_bidirec']

        if self.dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])
        elif self.dataset == 'InstaOrder':
            self.data_reader = reader.InstaOrderDataset(config['{}_annot_file'.format(phase)])
        else:
            self.data_reader = reader.KINSLVISDataset(
                self.dataset, config['{}_annot_file'.format(phase)])

        self.img_transform = transforms.Compose([
            transforms.Normalize(config['data_mean'], config['data_std'])
        ])
        self.sz = config['input_size']
        self.phase = phase

        self.config = config

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)

        if self.config['patch_or_image'] == "patch":
            self.get_pair_patch_or_image = self._get_pair
        elif self.config['patch_or_image'] == "image":
            self.get_pair_patch_or_image = self._get_pair_image
        elif self.config['patch_or_image'] == "resize":
            self.get_pair_patch_or_image = self._get_pair_resize
        # self.rank = dist.get_rank()

    def __len__(self):
        return self.data_reader.get_image_length()

    def _init_memcached(self):
        if not self.initialized:
            assert self.memcached_client is not None, "Please specify the path of your memcached_client"
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _load_image(self, fn):
        if self.memcached:
            try:
                img_value = mc.pyvector()
                self.mclient.Get(fn, img_value)
                img_value_str = mc.ConvertBuffer(img_value)
                img = utils.pil_loader(img_value_str)
            except:
                print('Read image failed ({})'.format(fn))
                raise Exception("Exit")
            else:
                return img
        else:
            return Image.open(fn).convert('RGB')

    def _get_pair_resize(self, modal, bboxes, idx1, idx2, imgfn, load_rgb=False, randshift=False):
        rgb = np.array(
            self._load_image(os.path.join(self.config['{}_image_root'.format(self.phase)], imgfn)))  # uint8
        rgb = cv2.resize(rgb, (self.sz, self.sz), interpolation=cv2.INTER_LINEAR)

        modal1 = cv2.resize(modal[idx1], (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        modal2 = cv2.resize(modal[idx2], (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            rgb = rgb[:, ::-1, :]
            modal1 = modal1[:, ::-1]
            modal2 = modal2[:, ::-1]

        rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)
        rgb = self.img_transform(rgb)  # CHW

        return modal1, modal2, rgb

    def _get_pair_image(self, modal, bboxes, idx1, idx2, imgfn, load_rgb=False, randshift=False):
        _, hh, ww = modal.shape
        bbox_hw = int(max(hh, ww))

        left = (bbox_hw - ww) // 2
        top = (bbox_hw - hh) // 2
        modal1_padded = np.zeros((bbox_hw, bbox_hw)).astype(modal.dtype)
        modal2_padded = np.zeros((bbox_hw, bbox_hw)).astype(modal.dtype)
        modal1_padded[top:top + hh, left:left + ww] = modal[idx1]
        modal2_padded[top:top + hh, left:left + ww] = modal[idx2]
        modal1 = cv2.resize(modal1_padded, (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        modal2 = cv2.resize(modal2_padded, (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        # flip
        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            flip = True
            modal1 = modal1[:, ::-1]
            modal2 = modal2[:, ::-1]
        else:
            flip = False

        if load_rgb:
            rgb = np.array(
                self._load_image(os.path.join(self.config['{}_image_root'.format(self.phase)], imgfn)))  # uint8
            rgb_padded = np.zeros((bbox_hw, bbox_hw, 3)).astype(rgb.dtype)
            rgb_padded[top:top + hh, left:left + ww, :] = rgb
            rgb = cv2.resize(rgb_padded, (self.sz, self.sz), interpolation=cv2.INTER_LINEAR)

            if flip:
                rgb = rgb[:, ::-1, :]

            rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)
            rgb = self.img_transform(rgb)  # CHW

        if load_rgb:
            return modal1, modal2, rgb
        else:
            return modal1, modal2, None

    def _get_pair(self, modal, bboxes, idx1, idx2, imgfn, load_rgb=False, randshift=False):
        bbox = utils.combine_bbox(bboxes[(idx1, idx2), :])
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * 2.), bbox[2] * 1.1, bbox[3] * 1.1])

        # shift & scale aug
        if self.phase == 'train':
            if randshift:
                centerx += np.random.uniform(*self.config['base_aug']['shift']) * size
                centery += np.random.uniform(*self.config['base_aug']['shift']) * size
            size /= np.random.uniform(*self.config['base_aug']['scale'])

        # crop
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        modal1 = cv2.resize(utils.crop_padding(modal[idx1], new_bbox, pad_value=(0,)),
                            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        modal2 = cv2.resize(utils.crop_padding(modal[idx2], new_bbox, pad_value=(0,)),
                            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        # flip
        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            flip = True
            modal1 = modal1[:, ::-1]
            modal2 = modal2[:, ::-1]
        else:
            flip = False

        if load_rgb:
            rgb = np.array(
                self._load_image(os.path.join(self.config['{}_image_root'.format(self.phase)], imgfn)))  # uint8
            rgb = cv2.resize(utils.crop_padding(rgb, new_bbox, pad_value=(0, 0, 0)), (self.sz, self.sz),
                             interpolation=cv2.INTER_CUBIC)
            if flip:
                rgb = rgb[:, ::-1, :]

            rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)
            rgb = self.img_transform(rgb)  # CHW

        if load_rgb:
            return modal1, modal2, rgb
        else:
            return modal1, modal2, None

    def _get_pair_ind(self, idx):
        modal, category, bboxes, amodal, image_fn = self.data_reader.get_image_instances(idx, with_gt=True)
        if self.config['use_category']:
            modal = modal * category[:, None, None]

        if self.dataset == "KINS":
            gt_occ_matrix = infer.infer_gt_order(modal, amodal)
        elif self.dataset == 'InstaOrder':
            gt_occ_matrix = self.data_reader.get_gt_ordering(idx, type="occlusion", rm_bidirec=self.rm_bidirec)
        else:
            gt_occ_matrix = self.data_reader.get_gt_ordering(idx)

        np.fill_diagonal(gt_occ_matrix, -1)
        pairs = np.where(gt_occ_matrix == 1)
        non_pairs = np.where(gt_occ_matrix == 0)

        if len(pairs[0]) == 0:
            return self._get_pair_ind(np.random.choice(len(self)))
        return modal, bboxes, image_fn, pairs, non_pairs, gt_occ_matrix

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()

        modal, bboxes, image_fn, pairs, non_pairs, gt_occ_matrix = self._get_pair_ind(idx)
        if self.algo == "OrderNet":
            """
            label meaning
            0: B over A
            1: A over B
            2: none
            3: bi-direc
            """
            if np.random.rand() < 0.7 or len(non_pairs[0]) == 0:
                ### pair
                randidx = np.random.choice(len(pairs[0]))
                idx1 = pairs[0][randidx]
                idx2 = pairs[1][randidx]
                label = 1

                b_over_a = gt_occ_matrix[idx2, idx1]
                if self.config['extend_bidirec'] and b_over_a:
                    label = 3

            else:
                ### non_pair
                randidx = np.random.choice(len(non_pairs[0]))
                idx1 = non_pairs[0][randidx]
                idx2 = non_pairs[1][randidx]
                label = 2

            modal1, modal2, rgb = self.get_pair_patch_or_image(
                modal, bboxes, idx1, idx2, image_fn,
                load_rgb=self.config['load_rgb'], randshift=True)

            if rgb is None:
                rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32)  # 3HW

            modal_tensor1 = torch.from_numpy(modal1.astype(np.float32)).unsqueeze(0)  # 1HW, float
            modal_tensor2 = torch.from_numpy(modal2.astype(np.float32)).unsqueeze(0)  # 1HW, float

            if np.random.rand() < 0.5:
                # label is 1 or 2
                return rgb, modal_tensor1, modal_tensor2, label
            else:
                # label is 0 or 2
                label = 0 if label == 1 else label
                return rgb, modal_tensor2, modal_tensor1, label

        ## bidirection
        elif self.algo == "InstaOrderNet_o":
            if np.random.rand() < 0.7 or len(non_pairs[0]) == 0:
                ### pair
                randidx = np.random.choice(len(pairs[0]))
                idx1 = pairs[0][randidx]
                idx2 = pairs[1][randidx]
            else:
                ### non_pair
                randidx = np.random.choice(len(non_pairs[0]))
                idx1 = non_pairs[0][randidx]
                idx2 = non_pairs[1][randidx]

            modal1, modal2, rgb = self.get_pair_patch_or_image(
                modal, bboxes, idx1, idx2, image_fn,
                load_rgb=self.config['load_rgb'], randshift=True)

            a_over_b = gt_occ_matrix[idx1, idx2]
            b_over_a = gt_occ_matrix[idx2, idx1]

            if rgb is None:
                rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32)  # 3HW

            modal_tensor1 = torch.from_numpy(modal1.astype(np.float32)).unsqueeze(0)  # 1HW, float
            modal_tensor2 = torch.from_numpy(modal2.astype(np.float32)).unsqueeze(0)  # 1HW, float
            if np.random.rand() < 0.5:
                return rgb, modal_tensor1, modal_tensor2, torch.tensor([b_over_a, a_over_b]).float()
            else:
                return rgb, modal_tensor2, modal_tensor1, torch.tensor([a_over_b, b_over_a]).float()
