import numpy as np
import cv2

import torch
import torch.nn as nn

import utils
import pdb
from skimage.morphology import convex_hull
from sklearn.metrics import precision_score, recall_score, f1_score
import collections
import matplotlib.pyplot as plt
from utils.data_utils import transform_rgb, transform_resize, get_closest_int_multiple_of
import copy


def extract_upper_tri_without_diagonal(A):
    # A: NxN matrix
    return A[np.triu_indices_from(A, k=1)]


def net_forward(model, image, inmodal_patch, eraser, use_rgb, th):
    if use_rgb:
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
        image = image.cuda()
    inmodal_patch = torch.from_numpy(inmodal_patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        if eraser is not None:
            eraser = torch.from_numpy(eraser.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
            if use_rgb:
                output = model.model(torch.cat([inmodal_patch, eraser], dim=1), image)
            else:
                output = model.model(torch.cat([inmodal_patch, eraser], dim=1))
        else:
            if use_rgb:
                output = model.model(torch.cat([inmodal_patch], dim=1), image)
            else:
                output = model.model(inmodal_patch)
        output = nn.functional.softmax(output, dim=1)
    output.detach_()
    return (output[0, 1, :, :] > th).cpu().numpy().astype(np.uint8)


def net_forward_OrderNet(model, image, inmodal1, inmodal2):
    inmodal1 = torch.from_numpy(inmodal1.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    inmodal2 = torch.from_numpy(inmodal2.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        output1 = nn.functional.softmax(model.model(torch.cat([inmodal1, inmodal2, image], dim=1)), dim=1)
        output2 = nn.functional.softmax(model.model(torch.cat([inmodal2, inmodal1, image], dim=1)), dim=1)
        output1.detach_()
        output2.detach_()

        prob_1_over_2 = (output1[:, 1] + output2[:, 0]) / 2  # average results
        prob_2_over_1 = (output1[:, 0] + output2[:, 1]) / 2  # average results
        prob_none = (output1[:, 2] + output2[:, 2]) / 2  # average results

        if output1.shape[-1] == 4:  # for OrderNet_ext
            prob_both = (output1[:, 3] + output2[:, 3]) / 2  # average results
        else:
            prob_both = torch.tensor(0)

    argidx = np.argmax(
        (prob_1_over_2.cpu().numpy().item(), prob_2_over_1.cpu().numpy().item(),
         prob_none.cpu().numpy().item(), prob_both.cpu().numpy().item()))
    if argidx == 0:
        # 1 over 2
        return True, False
    elif argidx == 1:
        # 2 over 1
        return False, True
    elif argidx == 2:
        # no occlusion order
        return False, False
    elif argidx == 3:
        # bidirec
        return True, True


def net_forward_midas_pretrained(pred_disp, inmodal1, inmodal2, disp_select_method):
    # return argidx same as net_forward_depth
    # method: mean or median
    # pred_disp, inmodal1, inmodal2: [H,W]
    pixel_depth = 1 / (pred_disp + 1e-6)
    inmodal1 = torch.from_numpy(inmodal1.astype(bool)).cuda()
    inmodal2 = torch.from_numpy(inmodal2.astype(bool)).cuda()

    masked1 = pixel_depth[inmodal1]
    masked2 = pixel_depth[inmodal2]
    clip_min1, clip_max1 = torch.quantile(masked1, 0.05), torch.quantile(masked1, 0.95)
    clip_min2, clip_max2 = torch.quantile(masked2, 0.05), torch.quantile(masked2, 0.95)

    clipped1 = torch.clip(masked1, clip_min1, clip_max1)
    clipped2 = torch.clip(masked2, clip_min2, clip_max2)
    if disp_select_method == 'median':
        depth1, depth2 = torch.median(clipped1), torch.median(clipped2)
    elif disp_select_method == 'mean':
        depth1, depth2 = torch.mean(clipped1), torch.mean(clipped2)

    if depth1 < depth2:
        return 0
    elif depth1 > depth2:
        return 1
    else:
        return 2


def net_forward_InstaDepthNet(model, image, inmodal1, inmodal2):
    inmodal1 = torch.from_numpy(inmodal1.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    inmodal2 = torch.from_numpy(inmodal2.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        disp1, depth_out1, occ_out1 = model.model(image, inmodal1, inmodal2)
        disp2, depth_out2, occ_out2 = model.model(image, inmodal2, inmodal1)

        depth_out1, depth_out2 = nn.functional.softmax(depth_out1), nn.functional.softmax(depth_out2)
        depth_out1.detach_()
        depth_out2.detach_()

        prob_1_closer_2 = (depth_out1[:, 0] + depth_out2[:, 1]) / 2  # average results
        prob_1_farther_2 = (depth_out1[:, 1] + depth_out2[:, 0]) / 2  # average results
        prob_1_equal_2 = (depth_out1[:, 2] + depth_out2[:, 2]) / 2  # average results
        argidx_depth = np.argmax(
            (prob_1_closer_2.cpu().numpy().item(), prob_1_farther_2.cpu().numpy().item(),
             prob_1_equal_2.cpu().numpy().item()))

        # get occlusion order
        if occ_out1 is not None:
            occ_out1, occ_out2 = nn.functional.sigmoid(occ_out1), nn.functional.sigmoid(occ_out2)
            occ_out1.detach_()
            occ_out2.detach_()
            prob_1_over_2 = (occ_out1[:, 1] + occ_out2[:, 0]) / 2  # average results
            prob_2_over_1 = (occ_out1[:, 0] + occ_out2[:, 1]) / 2  # average results

            is_1_over_2 = prob_1_over_2.cpu().numpy().item() > 0.5
            is_2_over_1 = prob_2_over_1.cpu().numpy().item() > 0.5
        else:
            is_1_over_2, is_2_over_1 = 0, 0
    return argidx_depth, is_1_over_2, is_2_over_1, disp1, disp2


def net_forward_occ_depth(model, image, inmodal1, inmodal2):
    inmodal1 = torch.from_numpy(inmodal1.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    inmodal2 = torch.from_numpy(inmodal2.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        occ_out1, depth_out1 = model.model(torch.cat([inmodal1, inmodal2, image], dim=1))
        occ_out2, depth_out2 = model.model(torch.cat([inmodal2, inmodal1, image], dim=1))

        # get depth order
        depth_out1, depth_out2 = nn.functional.softmax(depth_out1), nn.functional.softmax(depth_out2)
        depth_out1.detach_()
        depth_out2.detach_()

        prob_1_closer_2 = (depth_out1[:, 0] + depth_out2[:, 1]) / 2  # average results
        prob_1_farther_2 = (depth_out1[:, 1] + depth_out2[:, 0]) / 2  # average results
        prob_1_equal_2 = (depth_out1[:, 2] + depth_out2[:, 2]) / 2  # average results

        # get occlusion order
        occ_out1, occ_out2 = nn.functional.sigmoid(occ_out1), nn.functional.sigmoid(occ_out2)
        occ_out1.detach_()
        occ_out2.detach_()
        prob_1_over_2 = (occ_out1[:, 1] + occ_out2[:, 0]) / 2  # average results
        prob_2_over_1 = (occ_out1[:, 0] + occ_out2[:, 1]) / 2  # average results

    argidx_depth = np.argmax(
        (prob_1_closer_2.cpu().numpy().item(), prob_1_farther_2.cpu().numpy().item(),
         prob_1_equal_2.cpu().numpy().item()))
    is_1_over_2 = prob_1_over_2.cpu().numpy().item() > 0.5
    is_2_over_1 = prob_2_over_1.cpu().numpy().item() > 0.5

    return argidx_depth, is_1_over_2, is_2_over_1


def net_forward_depth(model, image, inmodal1, inmodal2, use_rgb):
    inmodal1 = torch.from_numpy(inmodal1.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    inmodal2 = torch.from_numpy(inmodal2.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        if use_rgb:
            output1 = nn.functional.softmax(model.model(torch.cat([inmodal1, inmodal2, image], dim=1)), dim=1)
            output2 = nn.functional.softmax(model.model(torch.cat([inmodal2, inmodal1, image], dim=1)), dim=1)
        else:
            output1 = nn.functional.softmax(model.model(torch.cat([inmodal1, inmodal2], dim=1)), dim=1)
            output2 = nn.functional.softmax(model.model(torch.cat([inmodal2, inmodal1], dim=1)), dim=1)
        output1.detach_()
        output2.detach_()

        prob_1_closer_2 = (output1[:, 0] + output2[:, 1]) / 2  # average results
        prob_1_farther_2 = (output1[:, 1] + output2[:, 0]) / 2  # average results
        prob_1_equal_2 = (output1[:, 2] + output2[:, 2]) / 2  # average results

    argidx = np.argmax(
        (prob_1_closer_2.cpu().numpy().item(), prob_1_farther_2.cpu().numpy().item(),
         prob_1_equal_2.cpu().numpy().item()))

    return argidx


def net_forward_occ(model, image, inmodal1, inmodal2, use_rgb):
    inmodal1 = torch.from_numpy(inmodal1.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    inmodal2 = torch.from_numpy(inmodal2.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        if use_rgb:
            output1 = torch.sigmoid(model.model(torch.cat([inmodal1, inmodal2, image], dim=1)))
            output2 = torch.sigmoid(model.model(torch.cat([inmodal2, inmodal1, image], dim=1)))
        else:
            output1 = torch.sigmoid(model.model(torch.cat([inmodal1, inmodal2], dim=1)))
            output2 = torch.sigmoid(model.model(torch.cat([inmodal2, inmodal1], dim=1)))
        output1.detach_()
        output2.detach_()

        prob_1_over_2 = (output1[:, 1] + output2[:, 0]) / 2  # average results
        prob_2_over_1 = (output1[:, 0] + output2[:, 1]) / 2  # average results
        is_1_over_2 = prob_1_over_2.cpu().numpy().item() > 0.5
        is_2_over_1 = prob_2_over_1.cpu().numpy().item() > 0.5
        return is_1_over_2, is_2_over_1


def recover_mask(mask, bbox, h, w, interp):
    size = bbox[2]
    if interp == 'linear':
        mask = (cv2.resize(mask.astype(np.float32), (size, size),
                           interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    woff, hoff = bbox[0], bbox[1]
    newbbox = [-woff, -hoff, w, h]
    return utils.crop_padding(mask, newbbox, pad_value=(0,))


def resize_mask(mask, size, interp):
    if interp == 'linear':
        return (cv2.resize(
            mask.astype(np.float32), (size, size),
            interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        return cv2.resize(
            mask, (size, size), interpolation=cv2.INTER_NEAREST)


def infer_amodal_hull(inmodal, bboxes, order_matrix, order_grounded=True):
    amodal = []
    num = inmodal.shape[0]
    for i in range(num):
        m = inmodal[i]
        hull = convex_hull.convex_hull_image(m).astype(np.uint8)
        if order_grounded:
            assert order_matrix is not None
            ancestors = get_ancestors(order_matrix, i)
            eraser = (inmodal[ancestors, ...].sum(axis=0) > 0).astype(np.uint8)  # union
            hull[(eraser == 0) & (m == 0)] = 0
        amodal.append(hull)
    return amodal


def infer_order_hull(inmodal):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=int)
    occ_value_matrix = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        for j in range(i + 1, num):
            # if bordering(inmodal[i], inmodal[j]):
            if True:
                amodal_i = convex_hull.convex_hull_image(inmodal[i])
                amodal_j = convex_hull.convex_hull_image(inmodal[j])
                occ_value_matrix[i, j] = ((amodal_i > inmodal[i]) & (inmodal[j] == 1)).sum()
                occ_value_matrix[j, i] = ((amodal_j > inmodal[j]) & (inmodal[i] == 1)).sum()
    order_matrix[occ_value_matrix > occ_value_matrix.transpose()] = -1
    order_matrix[occ_value_matrix < occ_value_matrix.transpose()] = 1
    order_matrix[(occ_value_matrix == 0) & (occ_value_matrix == 0).transpose()] = 0
    return order_matrix


def infer_occ_order_area(inmodal, occluder='smaller'):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=int)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                # if True:
                area_i = inmodal[i].sum()
                area_j = inmodal[j].sum()

                small_idx, big_idx = (i, j) if area_i < area_j else (j, i)

                if occluder == "smaller":
                    order_matrix[small_idx, big_idx] = 1
                else:
                    order_matrix[big_idx, small_idx] = 1

    return order_matrix


def infer_occ_order_yaxis(inmodal, occluder='lower'):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=int)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                # if True:
                center_i = [coord.mean() for coord in np.where(inmodal[i] == 1)]  # y, x
                center_j = [coord.mean() for coord in np.where(inmodal[j] == 1)]  # y, x
                lower, higher = (i, j) if center_i[0] < center_j[0] else (j, i)
                if occluder == "lower":
                    order_matrix[lower, higher] = 1
                else:
                    order_matrix[higher, lower] = 1

    return order_matrix


def infer_depth_order_area(inmodal, closer='smaller'):
    # smaller mask occludes larger mask
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=int)
    for i in range(num):
        for j in range(i + 1, num):
            # if bordering(inmodal[i], inmodal[j]):
            if True:
                area_i = inmodal[i].sum()
                area_j = inmodal[j].sum()

                small_idx, big_idx = (i, j) if area_i < area_j else (j, i)

                if closer == "smaller":
                    order_matrix[small_idx, big_idx] = 1
                else:
                    order_matrix[big_idx, small_idx] = 1

    return order_matrix


def infer_depth_order_yaxis(inmodal, closer='lower'):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=int)
    for i in range(num):
        for j in range(i + 1, num):
            # if bordering(inmodal[i], inmodal[j]):
            if True:
                center_i = [coord.mean() for coord in np.where(inmodal[i] == 1)]  # y, x
                center_j = [coord.mean() for coord in np.where(inmodal[j] == 1)]  # y, x
                higher, lower = (i, j) if center_i[0] < center_j[0] else (j, i)
                if closer == "lower":
                    order_matrix[lower, higher] = 1
                else:
                    order_matrix[higher, lower] = 1

    return order_matrix


def infer_order_sup_occ_depth(model, image, inmodal, bboxes, pairs, method, patch_or_image, input_size,
                              disp_select_method):
    num = inmodal.shape[0]
    depth_order = np.zeros((num, num), dtype=int)
    occ_order = np.zeros((num, num), dtype=int)
    disp_clipped = None
    for i in range(num):
        for j in range(i + 1, num):
            if pairs == "nbor" and not bordering(inmodal[i], inmodal[j]):
                continue
            # preprocess image and mask
            if patch_or_image == "patch":
                bbox = utils.combine_bbox(bboxes[(i, j), :])
                centerx = bbox[0] + bbox[2] / 2.
                centery = bbox[1] + bbox[3] / 2.
                size = max([np.sqrt(bbox[2] * bbox[3] * 2.), bbox[2] * 1.1, bbox[3] * 1.1])
                new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
                rgb = cv2.resize(utils.crop_padding(
                    image, new_bbox, pad_value=(0, 0, 0)),
                    (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                modal_i = resize_mask(utils.crop_padding(
                    inmodal[i], new_bbox, pad_value=(0,)),
                    input_size, 'nearest')
                modal_j = resize_mask(utils.crop_padding(
                    inmodal[j], new_bbox, pad_value=(0,)),
                    input_size, 'nearest')
                rgb = transform_rgb(rgb)

            elif patch_or_image == "image":
                _, hh, ww = inmodal.shape
                bbox_hw = int(max(hh, ww))

                left = (bbox_hw - ww) // 2
                top = (bbox_hw - hh) // 2
                modal_i_padded = np.zeros((bbox_hw, bbox_hw)).astype(inmodal.dtype)
                modal_j_padded = np.zeros((bbox_hw, bbox_hw)).astype(inmodal.dtype)
                modal_i_padded[top:top + hh, left:left + ww] = inmodal[i]
                modal_j_padded[top:top + hh, left:left + ww] = inmodal[j]
                modal_i = cv2.resize(modal_i_padded, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(modal_j_padded, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

                image_padded = np.zeros((bbox_hw, bbox_hw, 3)).astype(image.dtype)
                image_padded[top:top + hh, left:left + ww, :] = image
                rgb = cv2.resize(image_padded, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                rgb = transform_rgb(rgb)

            elif patch_or_image == "resize":
                rgb = transform_resize(image, input_size, input_size)
                rgb = torch.from_numpy(rgb).cuda().unsqueeze(0)
                modal_i = cv2.resize(inmodal[i], (rgb.shape[3], rgb.shape[2]), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(inmodal[j], (rgb.shape[3], rgb.shape[2]), interpolation=cv2.INTER_NEAREST)

            elif patch_or_image == "orig":
                _, hh, ww = inmodal.shape
                hh = get_closest_int_multiple_of(hh, 32)
                ww = get_closest_int_multiple_of(ww, 32)
                rgb = transform_resize(image, ww, hh)
                rgb = torch.from_numpy(rgb).cuda().unsqueeze(0)
                modal_i = cv2.resize(inmodal[i], (ww, hh), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(inmodal[j], (ww, hh), interpolation=cv2.INTER_NEAREST)

            # predict order
            if method == "InstaOrderNet_od":
                argidx_depth, i_over_j, j_over_i = net_forward_occ_depth(model, rgb, modal_i, modal_j)
            elif method == "InstaDepthNet_od":
                argidx_depth, i_over_j, j_over_i, _, _ = net_forward_InstaDepthNet(model, rgb, modal_i, modal_j)

            # depth order
            if argidx_depth == 0:
                # i_closer_j
                depth_order[i, j] = 1
                depth_order[j, i] = 0
            elif argidx_depth == 1:
                # i_farther_j:
                depth_order[i, j] = 0
                depth_order[j, i] = 1
            elif argidx_depth == 2:
                # i_equal_j:
                depth_order[i, j] = 2
                depth_order[j, i] = 2

            # occlusion order
            if i_over_j:
                occ_order[i, j] = 1
            if j_over_i:
                occ_order[j, i] = 1

    return occ_order, depth_order


def infer_order_sup_occ(model, image, inmodal, bboxes, pairs, method, patch_or_image, input_size=256, use_rgb=True):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=int)

    for i in range(num):
        for j in range(i + 1, num):

            if pairs == "nbor" and not bordering(inmodal[i], inmodal[j]):
                continue
            # preprocess image and mask
            if patch_or_image == "patch":
                bbox = utils.combine_bbox(bboxes[(i, j), :])
                centerx = bbox[0] + bbox[2] / 2.
                centery = bbox[1] + bbox[3] / 2.
                size = max([np.sqrt(bbox[2] * bbox[3] * 2.), bbox[2] * 1.1, bbox[3] * 1.1])
                new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
                rgb = cv2.resize(utils.crop_padding(
                    image, new_bbox, pad_value=(0, 0, 0)),
                    (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                modal_i = resize_mask(utils.crop_padding(
                    inmodal[i], new_bbox, pad_value=(0,)),
                    input_size, 'nearest')
                modal_j = resize_mask(utils.crop_padding(
                    inmodal[j], new_bbox, pad_value=(0,)),
                    input_size, 'nearest')
                rgb = transform_rgb(rgb)

            elif patch_or_image == "image":
                _, hh, ww = inmodal.shape
                bbox_hw = int(max(hh, ww))

                left = (bbox_hw - ww) // 2
                top = (bbox_hw - hh) // 2
                modal_i_padded = np.zeros((bbox_hw, bbox_hw)).astype(inmodal.dtype)
                modal_j_padded = np.zeros((bbox_hw, bbox_hw)).astype(inmodal.dtype)
                modal_i_padded[top:top + hh, left:left + ww] = inmodal[i]
                modal_j_padded[top:top + hh, left:left + ww] = inmodal[j]
                modal_i = cv2.resize(modal_i_padded, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(modal_j_padded, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

                image_padded = np.zeros((bbox_hw, bbox_hw, 3)).astype(image.dtype)
                image_padded[top:top + hh, left:left + ww, :] = image
                rgb = cv2.resize(image_padded, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                rgb = transform_rgb(rgb)

            elif patch_or_image == "resize":
                rgb = transform_resize(image, input_size, input_size)
                rgb = torch.from_numpy(rgb).cuda().unsqueeze(0)
                modal_i = cv2.resize(inmodal[i], (rgb.shape[3], rgb.shape[2]), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(inmodal[j], (rgb.shape[3], rgb.shape[2]), interpolation=cv2.INTER_NEAREST)

            elif patch_or_image == "orig":
                _, hh, ww = inmodal.shape
                hh = get_closest_int_multiple_of(hh, 32)
                ww = get_closest_int_multiple_of(ww, 32)
                rgb = transform_resize(image, ww, hh)
                rgb = torch.from_numpy(rgb).cuda().unsqueeze(0)
                modal_i = cv2.resize(inmodal[i], (ww, hh), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(inmodal[j], (ww, hh), interpolation=cv2.INTER_NEAREST)

            if method == "OrderNet":
                i_over_j, j_over_i = net_forward_OrderNet(model, rgb, modal_i, modal_j)
            elif method == "InstaOrderNet_o":
                i_over_j, j_over_i = net_forward_occ(model, rgb, modal_i, modal_j, use_rgb)
            else:
                print("method name should be one of {OrderNet or InstaOrderNet_o}")
                return

            if i_over_j:
                order_matrix[i, j] = 1
            if j_over_i:
                order_matrix[j, i] = 1

    return order_matrix


def infer_order_sup_depth(model, image, inmodal, bboxes, pairs, method, patch_or_image, input_size, disp_select_method,
                          use_rgb=True):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=int)
    have_depth = False
    disp_clipped = None
    for i in range(num):
        for j in range(i + 1, num):

            if pairs == "nbor" and not bordering(inmodal[i], inmodal[j]):
                continue

            # preprocess image and mask
            if patch_or_image == "patch":
                bbox = utils.combine_bbox(bboxes[(i, j), :])
                centerx = bbox[0] + bbox[2] / 2.
                centery = bbox[1] + bbox[3] / 2.
                size = max([np.sqrt(bbox[2] * bbox[3] * 2.), bbox[2] * 1.1, bbox[3] * 1.1])
                new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
                rgb = cv2.resize(utils.crop_padding(
                    image, new_bbox, pad_value=(0, 0, 0)),
                    (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                modal_i = resize_mask(utils.crop_padding(
                    inmodal[i], new_bbox, pad_value=(0,)),
                    input_size, 'nearest')
                modal_j = resize_mask(utils.crop_padding(
                    inmodal[j], new_bbox, pad_value=(0,)),
                    input_size, 'nearest')
                rgb = transform_rgb(rgb)

            elif patch_or_image == "image":
                _, hh, ww = inmodal.shape
                bbox_hw = int(max(hh, ww))

                left = (bbox_hw - ww) // 2
                top = (bbox_hw - hh) // 2
                modal_i_padded = np.zeros((bbox_hw, bbox_hw)).astype(inmodal.dtype)
                modal_j_padded = np.zeros((bbox_hw, bbox_hw)).astype(inmodal.dtype)
                modal_i_padded[top:top + hh, left:left + ww] = inmodal[i]
                modal_j_padded[top:top + hh, left:left + ww] = inmodal[j]
                modal_i = cv2.resize(modal_i_padded, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(modal_j_padded, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

                image_padded = np.zeros((bbox_hw, bbox_hw, 3)).astype(image.dtype)
                image_padded[top:top + hh, left:left + ww, :] = image
                rgb = cv2.resize(image_padded, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                rgb = transform_rgb(rgb)

            elif patch_or_image == "resize":
                rgb = transform_resize(image, input_size, input_size)
                rgb = torch.from_numpy(rgb).cuda().unsqueeze(0)
                modal_i = cv2.resize(inmodal[i], (rgb.shape[3], rgb.shape[2]), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(inmodal[j], (rgb.shape[3], rgb.shape[2]), interpolation=cv2.INTER_NEAREST)

            elif patch_or_image == "orig":
                _, hh, ww = inmodal.shape
                hh = get_closest_int_multiple_of(hh, 32)
                ww = get_closest_int_multiple_of(ww, 32)
                rgb = transform_resize(image, ww, hh)
                rgb = torch.from_numpy(rgb).cuda().unsqueeze(0)
                modal_i = cv2.resize(inmodal[i], (ww, hh), interpolation=cv2.INTER_NEAREST)
                modal_j = cv2.resize(inmodal[j], (ww, hh), interpolation=cv2.INTER_NEAREST)

            # predict order
            if method == "InstaOrderNet_d":
                argidx = net_forward_depth(model, rgb, modal_i, modal_j, use_rgb)

            elif method == "midas_pretrained":
                if not have_depth:
                    have_depth = True
                    with torch.no_grad():
                        disp = model.forward(rgb).squeeze()
                        clip_min1, clip_max1 = torch.quantile(disp, 0.05), torch.quantile(disp, 0.95)
                        disp_clipped = torch.clip(disp, clip_min1, clip_max1)
                argidx = net_forward_midas_pretrained(disp, modal_i, modal_j, disp_select_method=disp_select_method)

            elif method == "InstaDepthNet_d" or method == "InstaDepthNet_od":
                # median or mean with InstaDepthNet
                if disp_select_method != '':
                    if not have_depth:
                        have_depth = True
                        with torch.no_grad():
                            zero_arr = np.zeros_like(modal_i, dtype=modal_i.dtype)
                            _, _, _, disp, _ = net_forward_InstaDepthNet(model, rgb, zero_arr, zero_arr)
                            disp = disp.squeeze()
                            clip_min1, clip_max1 = torch.quantile(disp, 0.05), torch.quantile(disp, 0.95)
                            disp_clipped = torch.clip(disp.squeeze(), clip_min1, clip_max1)
                    argidx = net_forward_midas_pretrained(disp, modal_i, modal_j, disp_select_method=disp_select_method)
                ########
                else:
                    argidx, _, _, _, _ = net_forward_InstaDepthNet(model, rgb, modal_i, modal_j)


            else:
                print("method name should be one of {InstaOrderNet_d or midas_pretrained}")
                return

            if argidx == 0:
                # i_closer_j
                order_matrix[i, j] = 1
                order_matrix[j, i] = 0
            elif argidx == 1:
                # i_farther_j:
                order_matrix[i, j] = 0
                order_matrix[j, i] = 1
            elif argidx == 2:
                # i_equal_j:
                order_matrix[i, j] = 2
                order_matrix[j, i] = 2
    return order_matrix, disp_clipped  # , disp1_list, disp2_list


def infer_order(model, image, inmodal, category, bboxes, pairs, use_rgb=True, th=0.5, dilate_kernel=0, input_size=None,
                min_input_size=32, interp='nearest', debug_info=False):
    '''
    image: HW3, inmodal: NHW, category: N, bboxes: N4
    '''
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=int)
    ind = []
    for i in range(num):
        for j in range(i + 1, num):
            if pairs == "nbor" and not bordering(inmodal[i], inmodal[j]):
                continue
            ind.append([i, j])
            ind.append([j, i])
    pairnum = len(ind)
    if pairnum == 0:
        return order_matrix
    ind = np.array(ind)
    eraser_patches = []
    inmodal_patches = []
    amodal_patches = []
    ratios = []
    for i in range(pairnum):
        tid = ind[i, 0]
        eid = ind[i, 1]
        image_patch = utils.crop_padding(image, bboxes[tid], pad_value=(0, 0, 0))
        inmodal_patch = utils.crop_padding(inmodal[tid], bboxes[tid], pad_value=(0,))
        if input_size is not None:
            newsize = input_size
        elif min_input_size > bboxes[tid, 2]:
            newsize = min_input_size
        else:
            newsize = None
        if newsize is not None:
            inmodal_patch = resize_mask(inmodal_patch, newsize, interp)
        eraser = utils.crop_padding(inmodal[eid], bboxes[tid], pad_value=(0,))
        if newsize is not None:
            eraser = resize_mask(eraser, newsize, interp)
        if dilate_kernel > 0:
            eraser = cv2.dilate(eraser, np.ones((dilate_kernel, dilate_kernel), np.uint8),
                                iterations=1)
        # erase inmodal
        inmodal_patch[eraser == 1] = 0
        # gather
        inmodal_patches.append(inmodal_patch)
        eraser_patches.append(eraser)
        amodal_patches.append(net_forward(
            model, image_patch, inmodal_patch * category[tid], eraser, use_rgb, th))
        ratios.append(1. if newsize is None else bboxes[tid, 2] / float(newsize))

    occ_value_matrix = np.zeros((num, num), dtype=np.float32)
    for i, idx in enumerate(ind):
        occ_value_matrix[idx[0], idx[1]] = (
                ((amodal_patches[i] > inmodal_patches[i]) & (eraser_patches[i] == 1)
                 ).sum() * (ratios[i] ** 2))
    order_matrix[occ_value_matrix > occ_value_matrix.transpose()] = 0
    order_matrix[occ_value_matrix < occ_value_matrix.transpose()] = 1
    order_matrix[(occ_value_matrix == 0) & (occ_value_matrix == 0).transpose()] = 0
    if debug_info:
        return order_matrix, ind, inmodal_patches, eraser_patches, amodal_patches
    else:
        return order_matrix


def bordering(a, b):
    dilate_kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
    a_dilate = cv2.dilate(a.astype(np.uint8), dilate_kernel, iterations=1)
    return np.any((a_dilate == 1) & b)


def bbox_in(box1, box2):
    l1, u1, r1, b1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    l2, u2, r2, b2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    if l1 >= l2 and u1 >= u2 and r1 <= r2 and b1 <= b2:
        return True
    else:
        return False


def fullcovering(mask1, mask2, box1, box2):
    if not (mask1 == 0).all() and not (mask2 == 0).all():
        return 0
    if (mask1 == 0).all() and bbox_in(box1, box2):  # 1 covered by 2
        return 1
    elif (mask2 == 0).all() and bbox_in(box2, box1):
        return 2
    else:
        return 0


def infer_gt_order(inmodal, amodal):
    # inmodal = inmodal.numpy()
    # amodal = amodal.numpy()
    num = inmodal.shape[0]
    gt_order_matrix = np.zeros((num, num), dtype=int)
    for i in range(num):
        for j in range(i + 1, num):
            if not bordering(inmodal[i], inmodal[j]):
                continue
            occ_ij = ((inmodal[i] == 1) & (amodal[j] == 1)).sum()
            occ_ji = ((inmodal[j] == 1) & (amodal[i] == 1)).sum()
            # assert not (occ_ij > 0 and occ_ji > 0) # assertion error, why?
            if occ_ij == 0 and occ_ji == 0:  # bordering but not occluded
                continue
            if occ_ij >= occ_ji:
                gt_order_matrix[i, j] = 1
                gt_order_matrix[j, i] = 0
            else:
                gt_order_matrix[i, j] = 0
                gt_order_matrix[j, i] = 1
    return gt_order_matrix


def eval_order(order_matrix, gt_order_matrix):
    inst_num = order_matrix.shape[0]
    allpair_true = ((order_matrix == gt_order_matrix).sum() - inst_num) / 2
    allpair = (inst_num * inst_num - inst_num) / 2

    occpair_true = ((order_matrix == gt_order_matrix) & (gt_order_matrix != 0)).sum() / 2
    occpair = (gt_order_matrix != 0).sum() / 2

    err = np.where(order_matrix != gt_order_matrix)
    gt_err = gt_order_matrix[err]
    pred_err = order_matrix[err]
    show_err = np.concatenate([np.array(err).T + 1, gt_err[:, np.newaxis], pred_err[:, np.newaxis]], axis=1)
    return allpair_true, allpair, occpair_true, occpair, show_err


def calculate_whdr(order_matrix, gt_order_matrix, score_matrix, mask):
    if mask.sum() == 0:
        return -1
    whdr = ((gt_order_matrix[mask] != order_matrix[mask]) * score_matrix[mask]).sum() / score_matrix[mask].sum()
    return whdr * 100


def eval_depth_order_whdr(order_matrix, gt_order_ovl_count):
    gt_order_matrix, gt_overlap_matrix, gt_count_matrix = gt_order_ovl_count
    # extract upper triangle, without diagonal
    gt_order_matrix = extract_upper_tri_without_diagonal(gt_order_matrix)
    gt_overlap_matrix = extract_upper_tri_without_diagonal(gt_overlap_matrix)
    gt_count_matrix = extract_upper_tri_without_diagonal(gt_count_matrix)
    order_matrix = extract_upper_tri_without_diagonal(order_matrix)
    score_matrix = 2 / gt_count_matrix

    mask_ovls = collections.defaultdict(list)
    mask_ovls['ovlX'] = (gt_overlap_matrix == 0)
    mask_ovls['ovlO'] = (gt_overlap_matrix == 1)
    mask_ovls['ovlOX'] = mask_ovls['ovlX'] | mask_ovls['ovlO']

    mask_eqs = collections.defaultdict(list)
    mask_eqs['eq'] = (gt_order_matrix == 2)
    mask_eqs['neq'] = (gt_order_matrix == 0) | (gt_order_matrix == 1)
    mask_eqs['all'] = mask_eqs['eq'] | mask_eqs['neq']  # not np.ones bcs there might be invalid depth

    whdr_per_ovls = collections.defaultdict(list)  # all, eq, neq order
    for mask_ovl in mask_ovls.keys():
        for mask_eq in mask_eqs.keys():
            mask = mask_ovls[mask_ovl] & mask_eqs[mask_eq]
            whdr = calculate_whdr(order_matrix, gt_order_matrix, score_matrix, mask)
            save_str = f"{mask_ovl}_{mask_eq}"
            whdr_per_ovls[save_str].append(whdr)

    return whdr_per_ovls


def eval_order_recall_precision_f1(order_matrix, gt_order_matrix, zd):
    # order_matrix[order_matrix < 0] = 0
    # gt_order_matrix[gt_order_matrix < 0] = 0
    gt_order_ = gt_order_matrix[gt_order_matrix != -1].reshape(-1)
    order_ = order_matrix[gt_order_matrix != -1].reshape(-1)
    recall = recall_score(gt_order_, order_, average='binary', zero_division=zd)
    precision = precision_score(gt_order_, order_, average='binary', zero_division=zd)
    f1 = f1_score(gt_order_, order_, average='binary', zero_division=zd)
    return recall * 100, precision * 100, f1 * 100


def get_neighbors(graph, idx):
    return np.where(graph[idx, :] != 0)[0]


def get_ancestors(graph, idx):
    is_ancestor = np.zeros((graph.shape[0],), dtype=np.bool)
    visited = np.zeros((graph.shape[0],), dtype=np.bool)
    queue = {idx}
    while len(queue) > 0:
        q = queue.pop()
        if visited[q]:
            continue  # incase there exists cycles.
        visited[q] = True
        new_ancestor = np.where(graph[q, :] == -1)[0]
        is_ancestor[new_ancestor] = True
        queue.update(set(new_ancestor.tolist()))
    is_ancestor[idx] = False
    return np.where(is_ancestor)[0]


def infer_instseg(model, image, category, bboxes, new_bboxes, input_size, th, rgb=None):
    num = bboxes.shape[0]
    seg_patches = []
    for i in range(num):
        rel_bbox = [bboxes[i, 0] - new_bboxes[i, 0],
                    bboxes[i, 1] - new_bboxes[i, 1], bboxes[i, 2], bboxes[i, 3]]
        bbox_mask = np.zeros((new_bboxes[i, 3], new_bboxes[i, 2]), dtype=np.uint8)
        bbox_mask[rel_bbox[1]:rel_bbox[1] + rel_bbox[3], rel_bbox[0]:rel_bbox[0] + rel_bbox[2]] = 1
        bbox_mask = cv2.resize(bbox_mask, (input_size, input_size),
                               interpolation=cv2.INTER_NEAREST)
        bbox_mask_tensor = torch.from_numpy(
            bbox_mask.astype(np.float32) * category[i]).unsqueeze(0).unsqueeze(0).cuda()
        image_patch = cv2.resize(utils.crop_padding(image, new_bboxes[i], pad_value=(0, 0, 0)),
                                 (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        image_tensor = torch.from_numpy(
            image_patch.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).cuda()  # 13HW
        with torch.no_grad():
            output = model.model(torch.cat([image_tensor, bbox_mask_tensor], dim=1)).detach()
        if output.shape[2] != image_tensor.shape[2]:
            output = nn.functional.interpolate(
                output, size=image_tensor.shape[2:4],
                mode="bilinear", align_corners=True)  # 12HW
        output = nn.functional.softmax(output, dim=1)  # 12HW
        if rgb is not None:
            prob = output[0, ...].cpu().numpy()  # 2HW
            rgb_patch = cv2.resize(utils.crop_padding(rgb, new_bboxes[i], pad_value=(0, 0, 0)),
                                   (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            prob_crf = np.array(utils.densecrf(prob, rgb_patch)).reshape(*prob.shape)
            pred = (prob_crf[1, :, :] > th).astype(np.uint8)  # HW
        else:
            pred = (output[0, 1, :, :] > th).cpu().numpy().astype(np.uint8)  # HW
        seg_patches.append(pred)
    return seg_patches


def infer_amodal_sup(model, image, inmodal, category, bboxes, use_rgb=True, th=0.5,
                     input_size=None, min_input_size=16, interp='nearest', debug_info=False):
    num = inmodal.shape[0]
    inmodal_patches = []
    amodal_patches = []
    for i in range(num):
        image_patch = utils.crop_padding(image, bboxes[i], pad_value=(0, 0, 0))
        inmodal_patch = utils.crop_padding(inmodal[i], bboxes[i], pad_value=(0,))
        if input_size is not None:
            newsize = input_size
        elif min_input_size > bboxes[i, 2]:
            newsize = min_input_size
        else:
            newsize = None
        if newsize is not None:
            inmodal_patch = resize_mask(inmodal_patch, newsize, interp)
        inmodal_patches.append(inmodal_patch)
        amodal_patches.append(net_forward(
            model, image_patch, inmodal_patch * category[i], None, use_rgb, th))
    if debug_info:
        return inmodal_patches, amodal_patches
    else:
        return amodal_patches


def infer_amodal(model, image, inmodal, category, bboxes, order_matrix,
                 use_rgb=True, th=0.5, dilate_kernel=0,
                 input_size=None, min_input_size=16, interp='nearest',
                 order_grounded=True, debug_info=False):
    num = inmodal.shape[0]
    inmodal_patches = []
    eraser_patches = []
    amodal_patches = []
    for i in range(num):
        if order_grounded:
            ancestors = get_ancestors(order_matrix, i)
        else:
            ancestors = get_neighbors(order_matrix, i)
        image_patch = utils.crop_padding(image, bboxes[i], pad_value=(0, 0, 0))
        inmodal_patch = utils.crop_padding(inmodal[i], bboxes[i], pad_value=(0,))
        if input_size is not None:  # always
            newsize = input_size
        elif min_input_size > bboxes[i, 2]:
            newsize = min_input_size
        else:
            newsize = None
        if newsize is not None:
            inmodal_patch = resize_mask(inmodal_patch, newsize, interp)

        eraser = (inmodal[ancestors, ...].sum(axis=0) > 0).astype(np.uint8)  # union
        eraser = utils.crop_padding(eraser, bboxes[i], pad_value=(0,))
        if newsize is not None:
            eraser = resize_mask(eraser, newsize, interp)
        if dilate_kernel > 0:
            eraser = cv2.dilate(eraser, np.ones((dilate_kernel, dilate_kernel), np.uint8),
                                iterations=1)
        # erase inmodal
        inmodal_patch[eraser == 1] = 0
        # gather
        inmodal_patches.append(inmodal_patch)
        eraser_patches.append(eraser)
        amodal_patches.append(net_forward(
            model, image_patch, inmodal_patch * category[i], eraser, use_rgb, th))
    if debug_info:
        return inmodal_patches, eraser_patches, amodal_patches
    else:
        return amodal_patches


def patch_to_fullimage(patches, bboxes, height, width, interp):
    amodals = []
    for patch, bbox in zip(patches, bboxes):
        amodals.append(recover_mask(patch, bbox, height, width, interp))
    return np.array(amodals)
