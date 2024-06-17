import string
import networkx as nx
import numpy as np
import skimage.io as io
from skimage.draw import polygon2mask
import cv2
import copy
import os
import os.path
import pycocotools.mask as maskUtils
import json
import matplotlib.pyplot as plt
import string


def draw_graph(matrix, overlap_matrix=None, ind=None, pos=None):
    if overlap_matrix is None:
        overlap_matrix = np.zeros_like(matrix, dtype=np.bool)

    upper_lower = string.ascii_uppercase + string.ascii_lowercase + string.ascii_uppercase

    matrix[matrix < 0] = 0
    overlap_matrix[overlap_matrix < 0] = 0
    G = nx.DiGraph()
    for i in range(matrix.shape[0]):
        G.add_node(upper_lower[i])

    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_color='w')

    # draw edge
    matrix1 = matrix * overlap_matrix  # ovl
    matrix2 = matrix * np.logical_not(overlap_matrix)  # distinct

    edges = np.where(matrix1 >= 1)
    from_idx = edges[0].tolist()
    to_idx = edges[1].tolist()
    from_node1 = [upper_lower[i] for i in from_idx]
    to_node1 = [upper_lower[i] for i in to_idx]

    edges = np.where(matrix2 >= 1)
    from_idx = edges[0].tolist()
    to_idx = edges[1].tolist()
    from_node2 = [upper_lower[i] for i in from_idx]
    to_node2 = [upper_lower[i] for i in to_idx]
    edge_colors = ['green'] * len(from_node1) + ['black'] * len(from_node2)
    #     G.add_edges_from(list(zip(from_node1+from_node2, to_node1+to_node2)))

    for xxx in list(zip(from_node1 + from_node2, to_node1 + to_node2, edge_colors)):
        from_idx, to_idx, color = xxx
        G.add_edge(from_idx, to_idx, color=color)
    colors = nx.get_edge_attributes(G, 'color').values()
    nx.draw_networkx_edges(G, pos, edge_color=colors, arrowstyle='->', arrowsize=20, width=2)

    #     nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2, edge_color=edge_colors)
    return pos


def get_mid_top_loc(poly, image_h, image_w):
    # poly: (n, 2)
    top_idx = poly[:, 1].argmin()
    mid_top = (np.clip(int(poly[top_idx][0]), 25, image_w - 35),
               np.clip(int(poly[top_idx][1]), 25, image_h - 35))
    return mid_top


def get_mid_top_from_masks(masks):
    """
    :param masks: n,h,w
    :return:
    """
    mid_tops = []
    for i in range(masks.shape[0]):
        if masks[i].sum() == 0:
            mid_tops.append([-1, -1])
            continue
        contours, _ = cv2.findContours(np.uint8(masks[i]), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # [n1,1,2], [n2,1,2]
        poly = np.concatenate(contours)[:, 0]
        mid_tops.append(get_mid_top_loc(poly, masks.shape[-2], masks.shape[-1]))
    return mid_tops


def draw_polyline_with_mask(I, mask, c):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    I = cv2.drawContours(I, contours, -1, c, 3)
    return I


def fill_poly_with_mask(I, polyfilled, mask_instance, c, alpha=0.6):
    polyfilled[np.where(mask_instance == 1)] = c
    I = cv2.addWeighted(I, alpha, polyfilled, 1 - alpha, 0)
    return I


def write_text(img, text, location, c):
    cv2.rectangle(img, (location[0] - 5, location[1] - 10), (location[0] + 25, location[1] + 5), c, -1)
    cv2.putText(img, text, location, 0, 0.3, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img


def put_instance_mask_and_ID(I, masks, mid_tops, colors_rainbow=None, IDs=None):
    upper_lower = string.ascii_uppercase + string.ascii_lowercase + string.ascii_uppercase
    I_masked = copy.deepcopy(I)
    num_inst = len(masks) if type(masks) == list else masks.shape[0]  # masks: n,h,w
    colors = list((np.random.random(size=3 * num_inst).reshape(num_inst, -1) + 0.0001) * 255)
    if colors_rainbow != None:
        colors[:10] = colors_rainbow

    # show all annotations
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue

        I_masked = draw_polyline_with_mask(I_masked, mask, colors[i])
        I_masked = fill_poly_with_mask(I_masked, copy.deepcopy(I_masked), mask, colors[i], alpha=0.6)

    # # another loop to put instance ID on top
    # for i, mask in enumerate(masks):
    #     if mask.sum() == 0:
    #         continue
    #     # text = f"{str(i)}" if IDs == None else IDs[i]
    #     text = IDs[i]
    #     # text = upper_lower[i] if IDs == None else IDs[i]
    #     I_masked = write_text(I_masked, text, mid_tops[i], colors[i])

    return I_masked
