import numpy as np

import torch
import torch.nn as nn

import utils
import inference as infer
from . import SingleStageModel
import torch.distributed as dist
from utils.common_utils import UnNormalize
import copy
from utils.data_utils import inv_transform
import matplotlib.pyplot as plt
from scipy import ndimage


# **** model for occlusion & depth order ****#
class InstaOrderNet_od(SingleStageModel):
    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(InstaOrderNet_od, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        # loss
        self.depth_criterion = nn.CrossEntropyLoss()
        self.occ_criterion = nn.BCELoss()

        # # get rank
        # self.world_size = dist.get_world_size()
        # self.rank = dist.get_rank()

    def set_input(self, rgb=None, modal1=None, modal2=None, depth_order=None, count=None, is_overlap=None,
                  occ_order=None):
        self.rgb = rgb.cuda()
        self.modal1 = modal1.cuda()
        self.modal2 = modal2.cuda()
        ### depth
        self.depth_order1 = depth_order.cuda()
        self.depth_order2 = copy.deepcopy(self.depth_order1)
        self.depth_order2 = torch.ones_like(self.depth_order1) - self.depth_order1  # order2: 0->1, 1->0, 2->2
        self.depth_order2[self.depth_order1 == 2] = 2
        self.count = count.cuda()
        self.is_overlap = is_overlap.cuda()

        ### occlusion
        self.occ_order1 = occ_order.cuda()
        self.occ_order2 = copy.deepcopy(self.occ_order1)
        self.occ_order2 = torch.index_select(self.occ_order2, 1, torch.tensor([1, 0]).cuda())  # exchange two columns

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            occ_out1, depth_out1 = self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1))
            occ_out2, depth_out2 = self.model(torch.cat([self.modal2, self.modal1, self.rgb], dim=1))
            depth_out1, depth_out2 = nn.functional.softmax(depth_out1), nn.functional.softmax(depth_out2)
            occ_out1, occ_out2 = nn.functional.sigmoid(occ_out1), nn.functional.sigmoid(occ_out2)

        loss_to_log, loss = self.calculate_loss(depth_out1, depth_out2, occ_out1, occ_out2)
        return loss_to_log, {'loss': loss}

    def calculate_loss(self, depth_out1, depth_out2, occ_out1, occ_out2):
        ### weighted loss for distinct vs overlap pairs
        overlap_weight = self.params['overlap_weight']
        distinct_weight = self.params['distinct_weight']
        overlap_mask, distinct_mask = (self.is_overlap == 1), (self.is_overlap == 0)
        loss_overlap, loss_distinct = 0, 0

        if overlap_mask.sum() > 0:
            loss_overlap = self.depth_criterion(depth_out1[overlap_mask], self.depth_order1[overlap_mask]) + \
                           self.depth_criterion(depth_out2[overlap_mask], self.depth_order2[overlap_mask])
        if distinct_mask.sum() > 0:
            loss_distinct = self.depth_criterion(depth_out1[distinct_mask], self.depth_order1[distinct_mask]) + \
                            self.depth_criterion(depth_out2[distinct_mask], self.depth_order2[distinct_mask])
        depth_loss = loss_overlap * overlap_weight + loss_distinct * distinct_weight

        occ_loss = self.occ_criterion(occ_out1, self.occ_order1) + \
                   self.occ_criterion(occ_out2, self.occ_order2)

        loss = (depth_loss + occ_loss) / self.world_size
        loss_to_log = {"loss_occ": occ_loss, "loss_depth": depth_loss}

        return loss_to_log, loss

    def step(self):
        occ_out1, depth_out1 = self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1))
        occ_out2, depth_out2 = self.model(torch.cat([self.modal2, self.modal1, self.rgb], dim=1))
        depth_out1, depth_out2 = nn.functional.softmax(depth_out1), nn.functional.softmax(depth_out2)
        occ_out1, occ_out2 = nn.functional.sigmoid(occ_out1), nn.functional.sigmoid(occ_out2)

        loss_to_log, loss = self.calculate_loss(depth_out1, depth_out2, occ_out1, occ_out2)

        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return loss_to_log, {'loss': loss}


# **** model for depth order **** #
class InstaDepthNet_od(SingleStageModel):
    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(InstaDepthNet_od, self).__init__(params, dist_model)

        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        # loss
        self.depth_criterion = nn.CrossEntropyLoss()
        self.occ_criterion = nn.BCELoss()

        self.SoftMax = nn.functional.softmax
        self.Sigmoid = nn.functional.sigmoid

    def set_input(self, rgb, modal1, modal2, depth_order, count, is_overlap, occ_order):
        self.rgb = rgb.cuda()
        self.modal1 = modal1.cuda()
        self.modal2 = modal2.cuda()

        # order2: 0->1, 1->0, 2->2
        self.depth_order1 = depth_order.cuda()
        self.depth_order2 = copy.deepcopy(self.depth_order1)
        self.depth_order2[self.depth_order1 == 0] = 1
        self.depth_order2[self.depth_order1 == 1] = 0

        # occorder
        self.occ_order1 = occ_order.cuda()
        self.occ_order2 = copy.deepcopy(self.occ_order1)
        self.occ_order2 = torch.index_select(self.occ_order2, 1, torch.tensor([1, 0]).cuda())  # exchange two columns

        self.count = count.cuda()
        self.is_overlap = is_overlap.cuda()

    def calculate_loss(self, disp1, disp2, depth_out1, depth_out2, occ_out1, occ_out2):
        ### weight loss for distinct pairs
        overlap_weight = self.params['overlap_weight']
        distinct_weight = self.params['distinct_weight']
        overlap_mask, distinct_mask = (self.is_overlap == 1), (self.is_overlap == 0)

        loss_overlap, loss_distinct = 0, 0
        if overlap_mask.sum() > 0:
            loss_overlap = (self.depth_criterion(depth_out1[overlap_mask], self.depth_order1[overlap_mask]) + \
                            self.depth_criterion(depth_out2[overlap_mask], self.depth_order2[overlap_mask])) \
                           * overlap_weight / self.world_size
        if distinct_mask.sum() > 0:
            loss_distinct = (self.depth_criterion(depth_out1[distinct_mask], self.depth_order1[distinct_mask]) + \
                             self.depth_criterion(depth_out2[distinct_mask], self.depth_order2[distinct_mask])) \
                            * distinct_weight / self.world_size
        loss_occ_order = 0
        if self.params['occ_order_weight'] != 0:
            loss_occ_order = (self.occ_criterion(occ_out1, self.occ_order1) +
                              self.occ_criterion(occ_out2, self.occ_order2)) / self.world_size

        loss_smooth = 0
        if self.params['smooth_weight'] != 0:
            loss_smooth = (self.get_smooth_loss(disp1, self.rgb) +
                           self.get_smooth_loss(disp2, self.rgb)) \
                          * self.params['smooth_weight'] / self.world_size
        loss_disp_order = 0
        if self.params['dorder_weight'] != 0:
            # modal1_eroded, modal2_eroded = torch.zeros_like(self.modal1).bool(), torch.zeros_like(self.modal2).bool()
            for bb in range(self.modal1.shape[0]):
                if distinct_mask[bb] == 0: continue
                eroded1 = ndimage.binary_erosion(self.modal1[bb, 0].cpu().detach().numpy()).astype(np.bool)
                eroded2 = ndimage.binary_erosion(self.modal2[bb, 0].cpu().detach().numpy()).astype(np.bool)
                modal_eroded1, modal_eroded2 = torch.from_numpy(eroded1).cuda(), torch.from_numpy(eroded2).cuda()
                if self.depth_order1[bb] == 0:
                    # modal1 closer than modal2 (a.k.a disp1[modal1] > disp1[modal2])
                    loss_disp_order += (disp1[bb, 0][modal_eroded1] <= (disp1[bb, 0][modal_eroded2]).max()).sum()
                    loss_disp_order += (disp1[bb, 0][modal_eroded1].min() <= (disp1[bb, 0][modal_eroded2])).sum()
                    loss_disp_order += (disp2[bb, 0][modal_eroded1] >= (disp2[bb, 0][modal_eroded2]).max()).sum()
                    loss_disp_order += (disp2[bb, 0][modal_eroded1].min() >= (disp2[bb, 0][modal_eroded2])).sum()
                elif self.depth_order1[bb] == 1:
                    # modal1 farther than modal2 (a.k.a disp1[modal1] < disp1[modal2])
                    loss_disp_order += (disp1[bb, 0][modal_eroded1] >= (disp1[bb, 0][modal_eroded2]).max()).sum()
                    loss_disp_order += (disp1[bb, 0][modal_eroded1].min() >= (disp1[bb, 0][modal_eroded2])).sum()
                    loss_disp_order += (disp2[bb, 0][modal_eroded1] <= (disp2[bb, 0][modal_eroded2]).max()).sum()
                    loss_disp_order += (disp2[bb, 0][modal_eroded1].min() <= (disp2[bb, 0][modal_eroded2])).sum()

            loss_disp_order = loss_disp_order / (disp1.shape[2] * disp1.shape[3]) \
                              * self.params['dorder_weight'] / self.world_size

        loss = loss_overlap + loss_distinct + loss_occ_order + loss_smooth + loss_disp_order
        loss_to_log = {"loss_overlap": loss_overlap, "loss_distinct": loss_distinct, "loss_occ": loss_occ_order,
                       "loss_smooth": loss_smooth, "loss_disp_order": loss_disp_order}

        return loss_to_log, loss

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            disp1, depth_out1, occ_out1 = self.model(self.rgb, self.modal1, self.modal2)
            disp2, depth_out2, occ_out2 = self.model(self.rgb, self.modal2, self.modal1)
            depth_out1, depth_out2 = self.SoftMax(depth_out1), self.SoftMax(depth_out2)
            occ_out1, occ_out2 = self.Sigmoid(occ_out1), self.Sigmoid(occ_out2)

            disp1, disp2 = disp1.unsqueeze(1).cuda(), disp2.unsqueeze(1).cuda()
        loss_to_log, loss = self.calculate_loss(disp1, disp2, depth_out1, depth_out2, occ_out1, occ_out2)
        return loss_to_log, {'loss': loss}

    def step(self):
        disp1, depth_out1, occ_out1 = self.model(self.rgb, self.modal1, self.modal2)
        disp2, depth_out2, occ_out2 = self.model(self.rgb, self.modal2, self.modal1)
        depth_out1, depth_out2 = self.SoftMax(depth_out1), self.SoftMax(depth_out2)
        occ_out1, occ_out2 = self.Sigmoid(occ_out1), self.Sigmoid(occ_out2)
        disp1, disp2 = disp1.unsqueeze(1).cuda(), disp2.unsqueeze(1).cuda()

        loss_to_log, loss = self.calculate_loss(disp1, disp2, depth_out1, depth_out2, occ_out1, occ_out2)
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return loss_to_log, {'loss': loss}

    def min_max_norm(self, disp):
        # normalize to 0~1
        min_disp, max_disp = disp.min(2, True)[0].min(3, True)[0], disp.max(2, True)[0].max(3, True)[0]
        return (disp - min_disp) / (max_disp + 1e-7)

    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        # normalize to 0~1
        disp = self.min_max_norm(disp)

        # normalize disparity first
        mean_disp = disp.mean(2, True).mean(3, True)
        disp = disp / (mean_disp + 1e-7)

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


# **** model for depth order **** #
class InstaDepthNet_d(SingleStageModel):
    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(InstaDepthNet_d, self).__init__(params, dist_model)

        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        # loss
        self.depth_criterion = nn.CrossEntropyLoss()

        self.SoftMax = nn.functional.softmax
        self.Sigmoid = nn.functional.sigmoid

    def set_input(self, rgb, modal1, modal2, depth_order, count, is_overlap):
        self.rgb = rgb.cuda()
        self.modal1 = modal1.cuda()
        self.modal2 = modal2.cuda()

        # order2: 0->1, 1->0, 2->2
        self.depth_order1 = depth_order.cuda()
        self.depth_order2 = copy.deepcopy(self.depth_order1)
        self.depth_order2[self.depth_order1 == 0] = 1
        self.depth_order2[self.depth_order1 == 1] = 0

        self.count = count.cuda()
        self.is_overlap = is_overlap.cuda()

    def calculate_loss(self, disp1, disp2, depth_out1, depth_out2):
        ### weight loss for distinct pairs
        overlap_weight = self.params['overlap_weight']
        distinct_weight = self.params['distinct_weight']
        overlap_mask, distinct_mask = (self.is_overlap == 1), (self.is_overlap == 0)

        loss_overlap, loss_distinct = 0, 0
        if overlap_mask.sum() > 0:
            loss_overlap = (self.depth_criterion(depth_out1[overlap_mask], self.depth_order1[overlap_mask]) + \
                            self.depth_criterion(depth_out2[overlap_mask], self.depth_order2[overlap_mask])) \
                           * overlap_weight / self.world_size
        if distinct_mask.sum() > 0:
            loss_distinct = (self.depth_criterion(depth_out1[distinct_mask], self.depth_order1[distinct_mask]) + \
                             self.depth_criterion(depth_out2[distinct_mask], self.depth_order2[distinct_mask])) \
                            * distinct_weight / self.world_size

        loss_smooth = 0
        if self.params['smooth_weight'] != 0:
            loss_smooth = (self.get_smooth_loss(disp1, self.rgb) +
                           self.get_smooth_loss(disp2, self.rgb)) \
                          * self.params['smooth_weight'] / self.world_size
        loss_disp_order = 0
        if self.params['dorder_weight'] != 0:
            # modal1_eroded, modal2_eroded = torch.zeros_like(self.modal1).bool(), torch.zeros_like(self.modal2).bool()
            for bb in range(self.modal1.shape[0]):
                if distinct_mask[bb] == 0: continue
                eroded1 = ndimage.binary_erosion(self.modal1[bb, 0].cpu().detach().numpy()).astype(np.bool)
                eroded2 = ndimage.binary_erosion(self.modal2[bb, 0].cpu().detach().numpy()).astype(np.bool)
                modal_eroded1, modal_eroded2 = torch.from_numpy(eroded1).cuda(), torch.from_numpy(eroded2).cuda()
                if self.depth_order1[bb] == 0:
                    # modal1 farther than modal2 (a.k.a disp1[modal1] < disp1[modal2])
                    loss_disp_order += (disp1[bb, 0][modal_eroded1] <= (disp1[bb, 0][modal_eroded2]).max()).sum()
                    loss_disp_order += (disp1[bb, 0][modal_eroded1].min() <= (disp1[bb, 0][modal_eroded2])).sum()
                    loss_disp_order += (disp2[bb, 0][modal_eroded1] >= (disp2[bb, 0][modal_eroded2]).max()).sum()
                    loss_disp_order += (disp2[bb, 0][modal_eroded1].min() >= (disp2[bb, 0][modal_eroded2])).sum()
                elif self.depth_order1[bb] == 1:
                    # modal1 closer than modal2 (a.k.a disp1[modal1] > disp1[modal2])
                    loss_disp_order += (disp1[bb, 0][modal_eroded1] >= (disp1[bb, 0][modal_eroded2]).max()).sum()
                    loss_disp_order += (disp1[bb, 0][modal_eroded1].min() >= (disp1[bb, 0][modal_eroded2])).sum()
                    loss_disp_order += (disp2[bb, 0][modal_eroded1] <= (disp2[bb, 0][modal_eroded2]).max()).sum()
                    loss_disp_order += (disp2[bb, 0][modal_eroded1].min() <= (disp2[bb, 0][modal_eroded2])).sum()

            loss_disp_order = loss_disp_order / (disp1.shape[2] * disp1.shape[3]) \
                              * self.params['dorder_weight'] / self.world_size

        loss = loss_overlap + loss_distinct + loss_smooth + loss_disp_order
        loss_to_log = {"loss_overlap": loss_overlap, "loss_distinct": loss_distinct,
                       "loss_smooth": loss_smooth, "loss_disp_order": loss_disp_order}

        return loss_to_log, loss

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            disp1, depth_out1, _ = self.model(self.rgb, self.modal1, self.modal2)
            disp2, depth_out2, _ = self.model(self.rgb, self.modal2, self.modal1)
            depth_out1, depth_out2 = self.SoftMax(depth_out1), self.SoftMax(depth_out2)

            disp1, disp2 = disp1.unsqueeze(1).cuda(), disp2.unsqueeze(1).cuda()
        loss_to_log, loss = self.calculate_loss(disp1, disp2, depth_out1, depth_out2)
        return loss_to_log, {'loss': loss}

    def step(self):
        disp1, depth_out1, _ = self.model(self.rgb, self.modal1, self.modal2)
        disp2, depth_out2, _ = self.model(self.rgb, self.modal2, self.modal1)
        depth_out1, depth_out2 = self.SoftMax(depth_out1), self.SoftMax(depth_out2)
        disp1, disp2 = disp1.unsqueeze(1).cuda(), disp2.unsqueeze(1).cuda()

        loss_to_log, loss = self.calculate_loss(disp1, disp2, depth_out1, depth_out2)
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return loss_to_log, {'loss': loss}

    def min_max_norm(self, disp):
        # normalize to 0~1
        min_disp, max_disp = disp.min(2, True)[0].min(3, True)[0], disp.max(2, True)[0].max(3, True)[0]
        return (disp - min_disp) / (max_disp + 1e-7)

    def get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        # normalize to 0~1
        disp = self.min_max_norm(disp)

        # normalize disparity first
        mean_disp = disp.mean(2, True).mean(3, True)
        disp = disp / (mean_disp + 1e-7)

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


class InstaOrderNet_d(SingleStageModel):
    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(InstaOrderNet_d, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # # get rank
        # self.world_size = dist.get_world_size()
        # self.rank = dist.get_rank()

    def set_input(self, rgb=None, modal1=None, modal2=None, depth_order=None, count=None, is_overlap=None):
        self.rgb = rgb.cuda()
        self.modal1 = modal1.cuda()
        self.modal2 = modal2.cuda()

        self.depth_order1 = depth_order.cuda()
        # order2: 0->1, 1->0, 2->2
        self.depth_order2 = copy.deepcopy(self.depth_order1)
        self.depth_order2[self.depth_order1 == 0] = 1
        self.depth_order2[self.depth_order1 == 1] = 0

        self.count = count.cuda()
        self.is_overlap = is_overlap.cuda()

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.use_rgb:
                output1 = nn.functional.softmax(self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1)))
                output2 = nn.functional.softmax(self.model(torch.cat([self.modal2, self.modal1, self.rgb], dim=1)))
            else:
                output1 = nn.functional.softmax(self.model(torch.cat([self.modal1, self.modal2], dim=1)))
                output2 = nn.functional.softmax(self.model(torch.cat([self.modal2, self.modal1], dim=1)))
        loss_to_log = {}
        if ret_loss:
            loss = (self.criterion(output1, self.depth_order1) + self.criterion(output2,
                                                                                self.depth_order2)) / self.world_size
            return loss_to_log, {'loss': loss}
        else:
            return loss_to_log

    def step(self):
        if self.use_rgb:
            output1 = nn.functional.softmax(self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1)))
            output2 = nn.functional.softmax(self.model(torch.cat([self.modal2, self.modal1, self.rgb], dim=1)))
        else:
            output1 = nn.functional.softmax(self.model(torch.cat([self.modal1, self.modal2], dim=1)))
            output2 = nn.functional.softmax(self.model(torch.cat([self.modal2, self.modal1], dim=1)))

        ### weight loss for distinct vs overlap pairs
        overlap_weight = self.params['overlap_weight']
        distinct_weight = self.params['distinct_weight']
        overlap_mask, distinct_mask = (self.is_overlap == 1), (self.is_overlap == 0)
        loss_overlap, loss_distinct = 0, 0
        if overlap_mask.sum() > 0:
            loss_overlap = self.criterion(output1[overlap_mask], self.depth_order1[overlap_mask]) + \
                           self.criterion(output2[overlap_mask], self.depth_order2[overlap_mask])
        if distinct_mask.sum() > 0:
            loss_distinct = self.criterion(output1[distinct_mask], self.depth_order1[distinct_mask]) + \
                            self.criterion(output2[distinct_mask], self.depth_order2[distinct_mask])

        loss = (loss_overlap * overlap_weight + loss_distinct * distinct_weight) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}


# **** model for occlusion order **** #
class OrderNet(SingleStageModel):
    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(OrderNet, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        # loss
        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, rgb=None, modal1=None, modal2=None, occ_order=None):
        self.rgb = rgb.cuda()
        self.modal1 = modal1.cuda()
        self.modal2 = modal2.cuda()

        self.occ_order1 = occ_order.cuda()
        self.occ_order2 = copy.deepcopy(self.occ_order1)

        # order2: 0->1, 1->0, 2->2
        self.occ_order2[self.occ_order1 == 0] = 1
        self.occ_order2[self.occ_order1 == 1] = 0
        self.occ_order2[self.occ_order1 == 2] = 2
        self.occ_order2[self.occ_order1 == 3] = 3

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.use_rgb:
                output1 = nn.functional.softmax(self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1)))
                output2 = nn.functional.softmax(self.model(torch.cat([self.modal2, self.modal1, self.rgb], dim=1)))
            else:
                output1 = nn.functional.softmax(self.model(torch.cat([self.modal1, self.modal2], dim=1)))
                output2 = nn.functional.softmax(self.model(torch.cat([self.modal2, self.modal1], dim=1)))
        loss_to_log = {}
        if ret_loss:
            loss = (self.criterion(output1, self.occ_order1) + self.criterion(output2,
                                                                              self.occ_order2)) / self.world_size
            return loss_to_log, {'loss': loss}
        else:
            return loss_to_log

    def step(self):
        if self.use_rgb:
            output1 = nn.functional.softmax(self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1)))
            output2 = nn.functional.softmax(self.model(torch.cat([self.modal2, self.modal1, self.rgb], dim=1)))
        else:
            output1 = nn.functional.softmax(self.model(torch.cat([self.modal1, self.modal2], dim=1)))
            output2 = nn.functional.softmax(self.model(torch.cat([self.modal2, self.modal1], dim=1)))
        loss = (self.criterion(output1, self.occ_order1) + self.criterion(output2, self.occ_order2)) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}


class InstaOrderNet_o(SingleStageModel):
    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(InstaOrderNet_o, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params.get("use_rgb", False)

        # loss
        self.criterion = nn.BCELoss()

        # get rank
        # self.world_size = dist.get_world_size()
        # self.rank = dist.get_rank()

    def set_input(self, rgb=None, modal1=None, modal2=None, occ_order=None):
        self.rgb = rgb.cuda()
        self.modal1 = modal1.cuda()
        self.modal2 = modal2.cuda()

        self.occ_order1 = occ_order.cuda()
        self.occ_order2 = copy.deepcopy(self.occ_order1)
        self.occ_order2 = torch.index_select(self.occ_order2, 1, torch.tensor([1, 0]).cuda())  # exchange two columns

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.use_rgb:
                output1 = nn.functional.sigmoid(self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1)))
                output2 = nn.functional.sigmoid(self.model(torch.cat([self.modal2, self.modal1, self.rgb], dim=1)))
            else:
                output1 = nn.functional.sigmoid(self.model(torch.cat([self.modal1, self.modal2], dim=1)))
                output2 = nn.functional.sigmoid(self.model(torch.cat([self.modal2, self.modal1], dim=1)))

        loss_to_log = {}
        if ret_loss:
            loss = (self.criterion(output1, self.occ_order1) +
                    self.criterion(output2, self.occ_order2)) / self.world_size
            return loss_to_log, {'loss': loss}
        else:
            return loss_to_log

    def step(self):
        if self.use_rgb:
            output1 = nn.functional.sigmoid(self.model(torch.cat([self.modal1, self.modal2, self.rgb], dim=1)))
            output2 = nn.functional.sigmoid(self.model(torch.cat([self.modal2, self.modal1, self.rgb], dim=1)))
        else:
            output1 = nn.functional.sigmoid(self.model(torch.cat([self.modal1, self.modal2], dim=1)))
            output2 = nn.functional.sigmoid(self.model(torch.cat([self.modal2, self.modal1], dim=1)))

        loss = (self.criterion(output1, self.occ_order1) + self.criterion(output2, self.occ_order2)) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
