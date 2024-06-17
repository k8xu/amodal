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
from utils.data_utils import transform_rgb, transform_resize


class SupDepthOccOrderDataset(Dataset):

    def __init__(self, config, phase, algo):
        self.algo = algo

        self.dataset = config['dataset']
        self.rm_bidirec = config['remove_occ_bidirec']
        self.rm_overlap = config['remove_depth_overlap']

        # print("\n\ndataset = ", self.dataset)
        if self.dataset == 'InstaOrder':
            self.data_reader = reader.InstaOrderDataset(config['{}_annot_file'.format(phase)])
        else:
            raise NotImplementedError

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
        return self.data_reader.get_geometric_length()

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
        new_bbox = [bbox_hw // 2, bbox_hw // 2, bbox_hw, bbox_hw]

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

    def _get_pair_ind(self, img_id):
        modal, category, bboxes, amodal, image_fn = self.data_reader.get_image_instances(img_id, with_gt=True)
        if self.dataset == 'InstaOrder':
            gt_depth_matrix = self.data_reader.get_gt_ordering(img_id, type="depth", rm_overlap=self.rm_overlap)
            gt_occ_matrix = self.data_reader.get_gt_ordering(img_id, type="occlusion", rm_bidirec=self.rm_bidirec)
        else:
            raise NotImplementedError

        return modal, bboxes, image_fn, gt_depth_matrix, gt_occ_matrix

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()

        img_id, depth_order = self.data_reader.get_imgId_and_depth(idx)
        modal, bboxes, image_fn, gt_order_overlap_count, gt_occ_matrix = self._get_pair_ind(img_id)
        gt_depth_matrix, gt_overlap_matrix, gt_count_matrix = gt_order_overlap_count

        split_char = "<" if "<" in depth_order else "="
        idx1, idx2 = list(map(int, depth_order.split(split_char)))

        modal1, modal2, rgb = self.get_pair_patch_or_image(
            modal, bboxes, idx1, idx2, image_fn,
            load_rgb=self.config['load_rgb'], randshift=True)
        if rgb is None:
            rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32)  # 3HW

        modal_tensor1 = torch.from_numpy(modal1.astype(np.float32)).unsqueeze(0)  # 1HW, float
        modal_tensor2 = torch.from_numpy(modal2.astype(np.float32)).unsqueeze(0)  # 1HW, float

        ####### depth_label:  A<B (0), A>B (1), A=B (2)
        if gt_depth_matrix[idx1, idx2] == -1:
            depth_label = -1
        elif gt_depth_matrix[idx1, idx2] == 1 and gt_depth_matrix[idx2, idx1] == 0:
            depth_label = 0
        elif gt_depth_matrix[idx1, idx2] == 2:
            depth_label = 2

        depth_count = gt_count_matrix[idx1, idx2]
        is_overlap = gt_overlap_matrix[idx1, idx2]

        ######## occ_label ########
        a_over_b = gt_occ_matrix[idx1, idx2]
        b_over_a = gt_occ_matrix[idx2, idx1]
        occ_label1 = torch.tensor([b_over_a, a_over_b]).float()
        occ_label2 = torch.tensor([a_over_b, b_over_a]).float()

        if np.random.rand() < 0.5:
            # -1(-1), 0(0), 2(2)
            return rgb, modal_tensor1, modal_tensor2, depth_label, depth_count, is_overlap, occ_label1
        else:
            # -1(-1), 0(1), 2(2)
            depth_label = 1 if depth_label == 0 else depth_label
            return rgb, modal_tensor2, modal_tensor1, depth_label, depth_count, is_overlap, occ_label2
