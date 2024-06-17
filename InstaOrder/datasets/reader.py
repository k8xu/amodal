import json
import numpy as np
import sys
from PIL import Image
from pycocotools.coco import COCO
import os
import cv2

sys.path.append('.')

import cvbase as cvb
import pycocotools.mask as maskUtils
import utils
import time
import torch
import torchvision.transforms as transforms
import csv


def read_KINS(ann):
    modal = maskUtils.decode(ann['inmodal_seg'])  # HW, uint8, {0, 1}
    bbox = ann['inmodal_bbox']  # luwh
    category = ann['category_id']
    if 'score' in ann.keys():
        score = ann['score']
    else:
        score = 1.
    return modal, bbox, category, score


def read_LVIS(ann, h, w):
    segm = ann["segmentation"]
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann["segmentation"]
    bbox = ann['bbox']  # luwh
    category = ann['category_id']
    return maskUtils.decode(rle), bbox, category


def read_COCOA(ann, h, w):
    if 'visible_mask' in ann.keys():
        rle = [ann['visible_mask']]
    else:
        rles = maskUtils.frPyObjects([ann['segmentation']], h, w)
        rle = maskUtils.merge(rles)
    modal = maskUtils.decode(rle).squeeze()
    if np.all(modal != 1):
        # if the object if fully occluded by others,
        # use amodal bbox as an approximated location,
        # note that it will produce random amodal results.
        amodal = maskUtils.decode(maskUtils.merge(
            maskUtils.frPyObjects([ann['segmentation']], h, w)))
        bbox = utils.mask_to_bbox(amodal)
    else:
        bbox = utils.mask_to_bbox(modal)

    return modal, bbox, 1  # category as constant 1


class KITTIDataset(object):
    def __init__(self, args):
        self.args = args

        with open(args.data['val_annot_file'], 'r') as f:
            self.filenames = f.readlines()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        img_name = os.path.join(f"{self.args.data['val_image_root']}/rawdata/{file_path.split()[0]}")
        image = np.array(Image.open(img_name).convert('RGB'))

        top_margin = int(image.shape[0] - 352)
        left_margin = int((image.shape[1] - 1216) / 2)
        image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

        image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32))
        img_transform = transforms.Compose([
            transforms.Normalize(self.args.data['data_mean'], self.args.data['data_std'])
        ])
        image = img_transform(image / 255.)

        depth_name = os.path.join(f"{self.args.data['val_image_root']}/data_depth_annotated/{file_path.split()[1]}")

        return image, img_name, depth_name


class NYUDataset(object):
    def __init__(self, args):
        self.args = args
        with open(args.data['val_annot_file'], 'r') as f:
            self.filenames = f.readlines()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_path = self.filenames[idx]
        img_name = os.path.join(f"{self.args.data['val_image_root']}/{file_path.split()[0]}")
        image = np.array(Image.open(img_name).convert('RGB'))

        # image = image[16:-16, 16:-16, :]
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32))
        img_transform = transforms.Compose([
            transforms.Normalize(self.args.data['data_mean'], self.args.data['data_std'])
        ])
        image = img_transform(image / 255.)

        depth_name = os.path.join(f"{self.args.data['val_image_root']}/{file_path.split()[1]}")

        return image, img_name, depth_name


class DIWDataset(object):
    def __init__(self, args):
        self.args = args
        self.n_sample, self.data_handle = self._read_csv(args.data['val_annot_file'])

    def __len__(self):
        return self.n_sample

    def _read_csv(self, _filename):
        f = open(_filename, 'r')
        csv_file_handle = list(csv.reader(f))
        _n_lines = len(csv_file_handle)

        _data = {}
        _line_idx = 0
        _sample_idx = 0
        while _line_idx < _n_lines:
            dic = {}
            dic['img_filename'] = csv_file_handle[_line_idx][0]
            dic['n_point'] = 1
            dic['img_filename_line_idx'] = _line_idx

            _line_idx += dic['n_point']
            _line_idx += 1

            _data[_sample_idx] = dic

            _sample_idx += 1

        print('number of sample =', len(_data))
        _data['csv_file_handle'] = csv_file_handle

        return _sample_idx, _data

    def _read_one_sample(self, _sample_idx, handle):
        _data = {}
        _data['img_filename'] = handle[_sample_idx]['img_filename']
        _data['n_point'] = handle[_sample_idx]['n_point']
        _line_idx = handle[_sample_idx]['img_filename_line_idx'] + 1

        for point_idx in range(0, handle[_sample_idx]['n_point']):
            A_y = int(handle['csv_file_handle'][_line_idx][0])
            A_x = int(handle['csv_file_handle'][_line_idx][1])
            B_y = int(handle['csv_file_handle'][_line_idx][2])
            B_x = int(handle['csv_file_handle'][_line_idx][3])
            _data['A_yx'] = [A_y, A_x]
            _data['B_yx'] = [B_y, B_x]

            _data['AB_depth_ordinal'] = handle['csv_file_handle'][_line_idx][4][0]
            _line_idx += 1

        return _data

    def __getitem__(self, idx):
        thumb_filename = self.data_handle[idx]['img_filename']  # "./DIW_test/xxx.thumb"
        thumb_filename = f"{self.args.data['val_image_root']}/{thumb_filename[1:]}"  # "{BASE}/DIW_test/xxx.thumb"
        img = np.array(Image.open(thumb_filename).convert('RGB'))

        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] != 3):
            img_temp = np.zeros((img.shape[0], img.shape[1], 3))
            img_temp[:, :, 0] = img
            img_temp[:, :, 1] = img
            img_temp[:, :, 2] = img
            img = img_temp

        image = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32))
        img_transform = transforms.Compose([
            transforms.Normalize(self.args.data['data_mean'], self.args.data['data_std'])
        ])
        image = img_transform(image / 255.)

        _line_idx = self.data_handle[idx]['img_filename_line_idx'] + 1
        assert _line_idx == (2 * idx + 1)
        A_y = int(self.data_handle['csv_file_handle'][_line_idx][0]) - 1
        A_x = int(self.data_handle['csv_file_handle'][_line_idx][1]) - 1
        B_y = int(self.data_handle['csv_file_handle'][_line_idx][2]) - 1
        B_x = int(self.data_handle['csv_file_handle'][_line_idx][3]) - 1
        ordinal = self.data_handle['csv_file_handle'][_line_idx][4][0]
        Ayx_Byx_ord = [[A_y, A_x], [B_y, B_x], ordinal]
        return img, image, Ayx_Byx_ord, thumb_filename


class COCOADataset(object):
    def __init__(self, annot_fn):
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']

        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.images_info)

    def get_gt_ordering(self, imgidx):
        num = len(self.annot_info[imgidx]['regions'])
        gt_order_matrix = np.zeros((num, num), dtype=np.int)
        order_str = self.annot_info[imgidx]['depth_constraint']
        if len(order_str) == 0:
            return gt_order_matrix
        order_str = order_str.split(',')
        for o in order_str:
            idx1, idx2 = o.split('-')
            idx1, idx2 = int(idx1) - 1, int(idx2) - 1
            if self.annot_info[imgidx]['regions'][idx2]['occlude_rate'] > 0.95:
                continue

            gt_order_matrix[idx1, idx2] = 1
            # gt_order_matrix[idx2, idx1] = -1
        return gt_order_matrix  # num x num

    def get_instance(self, idx, with_gt=False):
        imgidx, regidx = self.indexing[idx]
        # img
        img_info = self.images_info[imgidx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        # region
        reg = self.annot_info[imgidx]['regions'][regidx]
        modal, bbox, category = read_COCOA(reg, h, w)
        if with_gt:
            amodal = maskUtils.decode(maskUtils.merge(
                maskUtils.frPyObjects([reg['segmentation']], h, w)))
        else:
            amodal = None
        return modal, bbox, category, image_fn, amodal

    def get_image_instances(self, idx, with_id=False, with_gt=False, with_anns=False, ignore_stuff=False):
        ann_info = self.annot_info[idx]
        img_info = self.images_info[idx]
        image_fn = img_info['file_name']
        image_id = img_info['id']
        w, h = img_info['width'], img_info['height']
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        for reg in ann_info['regions']:
            if ignore_stuff and reg['isStuff']:
                continue
            modal, bbox, category = read_COCOA(reg, h, w)
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            if with_gt:
                amodal = maskUtils.decode(maskUtils.merge(
                    maskUtils.frPyObjects([reg['segmentation']], h, w)))
                ret_amodal.append(amodal)
        if with_anns and with_id:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), \
                   image_fn, ann_info, image_id
        elif with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), \
                   image_fn, ann_info, image_id
        elif with_id:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), \
                   image_fn, image_id
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), \
                   image_fn


class InstaOrderDataset(object):
    def __init__(self, annot_fn):
        data = cvb.load(annot_fn)
        # self.images_info = data['images']
        self.annot_info = data['annotations']
        data_types = ['train2017', 'val2017']

        for dtype in data_types:
            if dtype in annot_fn:
                data_type = dtype

        coco_annot_fn = annot_fn.replace(f"{annot_fn.split('/')[-1]}", f"instances_{data_type}.json")
        self.coco = COCO(coco_annot_fn)

    def get_image_length(self):
        return len(self.annot_info)

    def get_instance_length(self):
        self.indexing = []
        for img_id, ann in enumerate(self.annot_info):
            for inst_id in range(len(ann['instance_ids'])):
                self.indexing.append((img_id, inst_id))
        return len(self.indexing)

    def get_occlusion_length(self):
        self.occ_all_img_and_idx = []
        for img_id, ann in enumerate(self.annot_info):
            for occ_idx in range(len(ann['occlusion'])):
                self.occ_all_img_and_idx.append((img_id, occ_idx))
        return len(self.occ_all_img_and_idx)

    def get_geometric_length(self):
        self.depth_all_img_and_order = []
        for img_id, ann in enumerate(self.annot_info):
            for g_overlap_count in ann['depth']:
                self.depth_all_img_and_order.append((img_id, g_overlap_count['order']))
        return len(self.depth_all_img_and_order)

    def get_imgId_and_depth(self, depth_all_idx):
        return self.depth_all_img_and_order[depth_all_idx]

    def get_gt_ordering(self, imgidx, type, rm_bidirec=0, rm_overlap=0):
        num = len(self.annot_info[imgidx]['instance_ids'])

        assert type in ["depth", "occlusion"], "order type should be ond of depth or occlusion"
        if type == "occlusion":
            gt_occ_matrix = np.zeros((num, num), dtype=np.int)
            occ_str = self.annot_info[imgidx]['occlusion']
            if len(occ_str) == 0:
                return gt_occ_matrix

            for o in occ_str:
                if "&" in o['order'] and rm_bidirec == 1:
                    gt_occ_matrix[idx1, idx2] = -1
                    gt_occ_matrix[idx2, idx1] = -1


                elif "&" in o['order']:
                    idx1, idx2 = list(map(int, o['order'].split(' & ')[0].split('<')))
                    gt_occ_matrix[idx1, idx2] = 1
                    gt_occ_matrix[idx2, idx1] = 1

                else:
                    idx1, idx2 = list(map(int, o['order'].split('<')))
                    gt_occ_matrix[idx1, idx2] = 1
            return gt_occ_matrix  # num x num

        elif type == "depth":
            gt_depth_matrix = np.ones((num, num), dtype=np.int) * (-1)
            is_overlap_matrix = np.ones((num, num), dtype=np.int) * (-1)
            count_matrix = np.ones((num, num), dtype=np.int) * (-1)

            depth_str = self.annot_info[imgidx]['depth']

            if len(depth_str) == 0:
                return [gt_depth_matrix, is_overlap_matrix, count_matrix]

            for overlap_count in depth_str:
                depth_order = overlap_count['order']
                is_overlap = overlap_count['overlap']
                count = overlap_count['count']

                split_char = "<" if "<" in depth_order else "="
                idx1, idx2 = list(map(int, depth_order.split(split_char)))
                if rm_overlap and is_overlap:
                    is_overlap_matrix[idx1, idx2] = -1
                    is_overlap_matrix[idx2, idx1] = -1

                # set is_overlap_matrix
                elif is_overlap:
                    is_overlap_matrix[idx1, idx2] = 1
                    is_overlap_matrix[idx2, idx1] = 1
                else:
                    is_overlap_matrix[idx1, idx2] = 0
                    is_overlap_matrix[idx2, idx1] = 0

                # set gt_depth_matrix
                if split_char == "<":
                    gt_depth_matrix[idx1, idx2] = 1
                    gt_depth_matrix[idx2, idx1] = 0
                elif split_char == "=":
                    gt_depth_matrix[idx1, idx2] = 2
                    gt_depth_matrix[idx2, idx1] = 2
                # set count_matrix
                count_matrix[idx1, idx2] = count
                count_matrix[idx2, idx1] = count
            return [gt_depth_matrix, is_overlap_matrix, count_matrix]  # num x num

    def get_instance(self, idx, with_gt=False):
        # for PCNet-M
        imgidx, regidx = self.indexing[idx]

        # img
        image_id = self.annot_info[imgidx]['image_id']
        img_info = self.coco.loadImgs(image_id)[0]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']

        # region
        annId = self.annot_info[imgidx]['instance_ids'][regidx]
        ann = self.coco.loadAnns(annId)[0]
        modal, bbox, category = read_LVIS(ann, h, w)
        # reg = self.annot_info[imgidx]['regions'][regidx]
        # modal, bbox, category = read_COCOA(reg, h, w)
        amodal = None
        return modal, bbox, category, image_fn, amodal

    def get_image_instances(self, idx, with_id=False, with_gt=False, with_anns=False, ignore_stuff=False):
        ann_info = self.annot_info[idx]
        # img
        image_id = ann_info['image_id']
        img_info = self.coco.loadImgs(image_id)[0]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']

        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        annIds = [int(annId) for annId in ann_info['instance_ids']]

        for annId in annIds:
            ann = self.coco.loadAnns(annId)[0]
            modal, bbox, category = read_LVIS(ann, h, w)
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            # if with_gt:
            #     amodal = maskUtils.decode(maskUtils.merge(
            #         maskUtils.frPyObjects([reg['segmentation']], h, w)))
            #     ret_amodal.append(amodal)

        if with_anns and with_id:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), \
                   image_fn, ann_info, image_id
        elif with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), \
                   image_fn, ann_info, image_id
        elif with_id:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), \
                   image_fn, image_id
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), \
                   image_fn


class KINSLVISDataset(object):
    def __init__(self, dataset, annot_fn):
        self.dataset = dataset
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']
        self.category_info = data['categories']

        # make dict
        self.imgfn_dict = dict([(a['id'], a['file_name']) for a in self.images_info])
        self.size_dict = dict([(a['id'], (a['width'], a['height'])) for a in self.images_info])
        self.anns_dict = self.make_dict()
        self.img_ids = list(self.anns_dict.keys())

    def get_instance_length(self):
        return len(self.annot_info)

    def get_image_length(self):
        return len(self.img_ids)

    def get_instance(self, idx, with_gt=False):
        ann = self.annot_info[idx]
        # img
        imgid = ann['image_id']
        w, h = self.size_dict[imgid]
        image_fn = self.imgfn_dict[imgid]
        # instance
        if self.dataset == 'KINS':
            modal, bbox, category, _ = read_KINS(ann)
        elif self.dataset == 'LVIS':
            modal, bbox, category = read_LVIS(ann, h, w)
        else:
            raise Exception("No such dataset: {}".format(self.dataset))
        if with_gt:
            amodal = maskUtils.decode(
                maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
        else:
            amodal = None
        return modal, bbox, category, image_fn, amodal

    def make_dict(self):
        anns_dict = {}
        for ann in self.annot_info:
            image_id = ann['image_id']
            if not image_id in anns_dict:
                anns_dict[image_id] = [ann]
            else:
                anns_dict[image_id].append(ann)
        return anns_dict  # imgid --> anns

    def get_image_instances(self, idx, with_gt=False, with_anns=False):
        imgid = self.img_ids[idx]
        image_fn = self.imgfn_dict[imgid]
        w, h = self.size_dict[imgid]
        anns = self.anns_dict[imgid]
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        # ret_score = []
        for ann in anns:
            if self.dataset == 'KINS':
                modal, bbox, category, score = read_KINS(ann)
            elif self.dataset == 'LVIS':
                modal, bbox, category = read_LVIS(ann, h, w)
            else:
                raise Exception("No such dataset: {}".format(self.dataset))
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            # ret_score.append(score)
            if with_gt:
                amodal = maskUtils.decode(
                    maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
                ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(
                ret_amodal), image_fn, anns
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn


class MapillaryDataset(object):

    def __init__(self, root, annot_fn):
        with open(annot_fn, 'r') as f:
            annot = json.load(f)
        self.categories = annot['categories']
        self.annot_info = annot['images']
        self.root = root  # e.g., "data/manpillary/training"
        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.annot_info)

    def get_instance(self, idx, with_gt=False):
        assert not with_gt, \
            "Mapillary Vista has no ground truth for ordering or amodal masks."
        imgidx, regidx = self.indexing[idx]
        # img
        image_id = self.annot_info[imgidx]['image_id']
        image_fn = image_id + ".jpg"
        # region
        instance_map = np.array(
            Image.open("{}/instances/{}.png".format(
                self.root, image_id)), dtype=np.uint16)
        h, w = instance_map.shape[:2]
        reg_info = self.annot_info[imgidx]['regions'][regidx]
        modal = (instance_map == reg_info['instance_id']).astype(np.uint8)
        category = reg_info['category_id']
        bbox = np.array(utils.mask_to_bbox(modal))
        return modal, bbox, category, image_fn, None

    def get_image_instances(self, idx, with_gt=False, with_anns=False, ignore_stuff=False):
        assert not with_gt
        assert not ignore_stuff
        # img
        image_id = self.annot_info[idx]['image_id']
        image_fn = image_id + ".jpg"
        # region
        instance_map = np.array(
            Image.open("{}/instances/{}.png".format(
                self.root, image_id)), dtype=np.uint16)
        h, w = instance_map.shape[:2]
        instance_ids = np.unique(instance_map)
        category = instance_ids // 256
        num_instance = len(instance_ids)
        instance_ids_tensor = np.zeros((num_instance, h, w), dtype=np.uint16)
        instance_ids_tensor[...] = instance_ids[:, np.newaxis, np.newaxis]
        modal = (instance_ids_tensor == instance_map).astype(np.uint8)
        bboxes = []
        for i in range(modal.shape[0]):
            bboxes.append(utils.mask_to_bbox(modal[i, ...]))
        return modal, category, np.array(bboxes), None, image_fn


def mask_to_polygon(mask, tolerance=1.0, area_threshold=1):
    """Convert object's mask to polygon [[x1,y1, x2,y2 ...], [...]]
    Args:
        mask: object's mask presented as 2D array of 0 and 1
        tolerance: maximum distance from original points of polygon to approximated
        area_threshold: if area of a polygon is less than this value, remove this small object
    """
    from skimage import measure
    polygons = []
    # pad mask with 0 around borders
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_mask, 0.5)
    # Fix coordinates after padding
    contours = np.subtract(contours, 1)
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) > 2:
            contour = np.flip(contour, axis=1)
            reshaped_contour = []
            for xy in contour:
                reshaped_contour.append(xy[0])
                reshaped_contour.append(xy[1])
            reshaped_contour = [point if point > 0 else 0 for point in reshaped_contour]

            # Check if area of a polygon is enough
            rle = maskUtils.frPyObjects([reshaped_contour], mask.shape[0], mask.shape[1])
            area = maskUtils.area(rle)
            if sum(area) > area_threshold:
                polygons.append(reshaped_contour)
    return polygons
