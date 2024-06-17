import os
import sys
import argparse
from PIL import Image
import numpy as np
import torch
import re
import math
from tqdm import tqdm
import shutil
import glob
import json
import cv2
import yaml

sys.path.append('InstaOrder')
import models
import inference as infer

sys.path.append("Grounded-Segment-Anything")
sys.path.append("Grounded-Segment-Anything/GroundingDINO")
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor 

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintMixedContextPipeline
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint
from lama.bin.predict_for_mc import *

import warnings
warnings.filterwarnings("ignore")


class QueryObject:
    def __init__(self, img_path, img, img_pil, mask_id, query_mask, output_img_dir):
        self.img_path = img_path
        self.img = img
        self.img_pil = img_pil
        self.mask_id = mask_id
        self.query_mask = query_mask
        self.output_img_dir = output_img_dir
        self.run_iter = True
        self.iter_id = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Run the Progressive Occlusion-aware Completion pipeline')
    parser.add_argument('--input_dir',         type=str,  help="Folder path to images")
    parser.add_argument('--img_filenames_txt', type=str,  default="./img_filenames.txt", help='Text file with image filenames in input_dir that you want to use')
    parser.add_argument('--classes_txt',       type=str,  default="./classes.txt",       help='Text file with semantic classes to segment')
    parser.add_argument('--output_dir',        type=str,  default="./output")
    parser.add_argument('--gdino_config',      type=str,  default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--gdino_ckpt',        type=str,  default="Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
    parser.add_argument('--sam_ckpt',          type=str,  default="Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
    parser.add_argument('--instaorder_ckpt',   type=str,  default="InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar")
    parser.add_argument('--lama_config_path',  type=str,  default="lama/big-lama/config.yaml")
    parser.add_argument('--lama_ckpt_path',    type=str,  default="lama/big-lama/models/best.ckpt")
    parser.add_argument('--save_interm',       type=bool, default=True, help='Whether to save intermediate images')
    parser.add_argument('--max_iter_id',       type=int,  default=5,    help='Maximum number of pipeline iterations')
    parser.add_argument('--mc_timestep',       type=int,  help='Timestep to composite for Mixed Context Diffusion Sampling')
    parser.add_argument('--mc_clean_bkgd_img', type=str,  default="images/gray_wallpaper.jpeg", help='Path to clean background image for Mixed Context Diffusion Sampling')
    return parser.parse_args()


def read_txt(file_path: str):
    with open(file_path, 'r') as f:
        files = f.read().splitlines()
    return files


def find_mask_sides(mask, val=1):
    """
    Determine the bounding box of a given value
    """
    mask[mask > 0] = 1
    x_arr, y_arr = np.where(mask == val)
    x_min, x_max = min(x_arr), max(x_arr)
    y_min, y_max = min(y_arr), max(y_arr)
    return x_min, x_max, y_min, y_max


def load_models(gdino_config, gdino_ckpt, instaorder_ckpt=None, lama_config_path=None, lama_ckpt_path=None, mc_timestep=None, device="cuda"):
    """
    Load Grounding DINO, Stable Diffusion inpainter, InstaOrder, and LaMa inpainter
    """
    loaded_models = []

    # Grounding DINO
    gdino_args = SLConfig.fromfile(gdino_config)
    gdino_args.device = device
    gdino_model = build_model(gdino_args)
    gdino_ckpt = torch.load(gdino_ckpt, map_location="cpu")
    gdino_model.load_state_dict(clean_state_dict(gdino_ckpt["model"]), strict=False)
    gdino_model.eval()
    loaded_models.append(gdino_model)

    if mc_timestep is None:
        # Stable Diffusion
        sd_inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    else:
        # Stable Diffusion for Mixed Context Diffusion Sampling
        sd_inpaint_model = StableDiffusionInpaintMixedContextPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    sd_inpaint_model.enable_attention_slicing()
    sd_inpaint_model = sd_inpaint_model.to("cuda")
    loaded_models.append(sd_inpaint_model)

    # InstaOrder
    if instaorder_ckpt is not None:
        instaorder_model_params = {
            'algo': 'InstaOrderNet_od',
            'total_iter': 60000,
            'lr_steps': [32000, 48000],
            'lr_mults': [0.1, 0.1],
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'optim': 'SGD',
            'warmup_lr': [],
            'warmup_steps': [],
            'use_rgb': True,
            'backbone_arch': 'resnet50_cls',
            'backbone_param': {'in_channels': 5, 'num_classes': [2, 3]},
            'overlap_weight': 0.1, 'distinct_weight': 0.9
        }
        instaorder_model = models.__dict__['InstaOrderNet_od'](instaorder_model_params)
        instaorder_model.load_state(instaorder_ckpt)
        instaorder_model.switch_to('eval')
        loaded_models.append(instaorder_model)

    # LaMa
    if lama_config_path is not None and lama_ckpt_path is not None:
        with open(lama_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        lama_model = load_checkpoint(train_config, lama_ckpt_path, strict=False, map_location='cpu')
        lama_model.freeze()
        lama_model.to(device)
        loaded_models.append(lama_model)

    return loaded_models


def transform_image(img_pil, save_interm=False, output_img_dir=None):
    """
    Transform PIL image to tensor
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(img_pil, None)
    return img_tensor


def run_gdino(gdino_model, img, caption, box_thresh=0.35, text_thresh=0.35, with_logits=True, device="cuda"):
    gdino_model = gdino_model.to(device)
    img = img.to(device)

    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = gdino_model(img[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # Filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_thresh
    logits_filt = logits_filt[filt_mask]  # (num_filt, 256)
    boxes_filt = boxes_filt[filt_mask]  # (num_filt, 4)
    logits_filt.shape[0]

    # Get predicted objects
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_thresh, gdino_model.tokenizer(caption), gdino_model.tokenizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def run_sam(img_pil, sam_ckpt, boxes_filt, pred_phrases=None, device="cuda"):
    img = np.array(img_pil)
    predictor = SamPredictor(build_sam(checkpoint=sam_ckpt).to(device))
    predictor.set_image(img)

    # Predict SAM masks
    size = img_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    boxes_filt = boxes_filt.cpu()
    try:
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, img.shape[:2]).to(device)
        masks, iou_predictions, _ = predictor.predict_torch(  # masks: [1, 1, 512, 512]
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
    except: return None, None  # If there is an error, then skip to the next image
    masks = masks.cpu().numpy().squeeze(1)  # Convert from torch tensor to numpy array
    return img, masks


def segment(gdino_model, run_sam, sam_ckpt, img_pil, img_tensor, classes, device="cuda"):
    """
    Run Grounding DINO on image
    """
    classes = ". ".join(classes)
    boxes_filt, pred_phrases = run_gdino(gdino_model, img_tensor, classes, device=device)
    img, masks = run_sam(img_pil, sam_ckpt, boxes_filt, pred_phrases=pred_phrases)

    # Separate predicted phrases into class names and prediction scores
    class_names = []
    pred_scores = []
    for pred_phrase in pred_phrases:
        class_name, pred_score, _ = re.split("\(|\)", pred_phrase)
        class_names.append(class_name)
        pred_scores.append(float(pred_score))

    return img, masks, class_names, pred_scores


def check_valid_query(
    img, mask_id, query_mask, class_names, pred_scores, classes,
    query_pred_score_thresh=0.35, query_mask_size_thresh=0.02, save_interm=False, output_img_dir=None,
):
    """
    Check whether the query object is suitable for amodal completion
    """
    query_mask = query_mask.astype(np.uint8)
    query_class = class_names[mask_id]
    pred_score = pred_scores[mask_id]

    if pred_score < query_pred_score_thresh or query_mask.sum() < query_mask_size_thresh * img.shape[0] * img.shape[1]: return
    if query_class not in classes: return

    return query_mask, query_class


def expand_bbox(bboxes):
    """
    Expand bbox in InstaOrder network
    """
    new_bboxes = []
    for bbox in bboxes:
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * 3.0), bbox[2] * 1.1, bbox[3] * 1.1])
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        new_bboxes.append(new_bbox)
    return np.array(new_bboxes)


def find_expand_bboxes_instaorder(masks):
    bboxes = np.zeros((len(masks), 4))
    for i, mask in enumerate(masks):
        mask[mask > 0] = 1
        x_min_obj, x_max_obj, y_min_obj, y_max_obj = find_mask_sides(mask)
        w = y_max_obj - y_min_obj; h = x_max_obj - x_min_obj
        bboxes[i, 0] = y_min_obj
        bboxes[i, 1] = x_min_obj
        bboxes[i, 2] = w
        bboxes[i, 3] = h
    bboxes = expand_bbox(bboxes)
    return bboxes


def analyze_masks(instaorder_model, img, masks, mask_id):
    """
    Analyze occlusion order using InstaOrder

    Occlusion order and depth order matrices
        occ_order[i, j] = 1 if j is over i (due to transpose)
        depth_order[i, j] = 1 if i is closer than j
        depth_order[i, j] = 2 if i is equal to j
    """
    modal = np.zeros((len(masks), masks[0].shape[0], masks[0].shape[1]))
    for i, mask in enumerate(masks):
        modal[i] = mask
    bboxes = find_expand_bboxes_instaorder(masks)

    pcolor_occ_order, pcolor_depth_order = infer.infer_order_sup_occ_depth(
        instaorder_model, img, modal, bboxes, pairs="all", method="InstaOrderNet_od",
        patch_or_image="resize", input_size=384, disp_select_method="")
    pcolor_occ_order = pcolor_occ_order.transpose()

    all_occluder_masks = []
    occ_indices = pcolor_occ_order[mask_id]
    occ_mask_indices = np.where(occ_indices == 1)[0]
    for occ_mask_index in occ_mask_indices:
        if occ_mask_index == mask_id: continue  # Skip occluder mask if it's the same as the query mask
        all_occluder_masks.append(masks[occ_mask_index])
    return all_occluder_masks


def aggregate_occluders(query_mask, all_occluder_masks, query_class, mask_id, iter_id, save_interm=False, output_img_dir=None):
    """
    Aggregate all occluders into a single mask
    """
    agg_occluder_mask = np.zeros((query_mask.shape))
    for occluder_mask in all_occluder_masks:
        agg_occluder_mask += occluder_mask
    agg_occluder_mask[agg_occluder_mask > 0] = 1

    # Ensure new occluders do not contain query mask
    query_occ_overlap = query_mask + agg_occluder_mask
    agg_occluder_mask[query_occ_overlap > 1] = 0  # Prevent occluder from containing query mask
    return agg_occluder_mask


def create_canvas(input_arr, size_multiplier, canvas_val):
    """
    Preprocess input image or mask by placing on blank canvas
    """
    input_height, input_width = input_arr.shape[0], input_arr.shape[1]
    canvas_shape = list(input_arr.shape)
    canvas_shape[0] = int(input_height * size_multiplier)
    canvas_shape[1] = int(input_width * size_multiplier)
    canvas_shape = tuple(canvas_shape)

    assert canvas_val >= 0
    if canvas_val > 0:
        canvas = np.ones(canvas_shape) * canvas_val
    else:
        canvas = np.zeros(canvas_shape)

    # Place input on canvas
    input_height_start = (canvas_shape[0] // 2) - (input_height // 2)
    input_height_end = input_height_start + input_height
    input_width_start = (canvas_shape[1] // 2) - (input_width // 2)
    input_width_end = input_width_start + input_width
    canvas[input_height_start : input_height_end, input_width_start : input_width_end] = input_arr
    return canvas


def check_touch_boundary(mask, gap_pixels=10):
    """
    Check whether the mask touches image boundary
    """
    H, W = mask.shape[0], mask.shape[1]
    x_max_obj, x_min_obj, y_min_obj, y_max_obj = find_mask_sides(mask)

    sides_touched = set()
    if (x_max_obj >= H - gap_pixels):
        sides_touched.add("bottom")
    if (x_min_obj <= gap_pixels):
        sides_touched.add("top")
    if (y_max_obj >= W - gap_pixels):
        sides_touched.add("right")
    if (y_min_obj <= gap_pixels):
        sides_touched.add("left")
    return sides_touched


def find_crop_region(query_mask, query_mask_canvas, pad_pixels=60, crop_buffer=60):
    """
    Apply conditional padding and determine cropping region
    """
    query_mask_canvas_height, query_mask_canvas_width = query_mask_canvas.shape
    crop_x_min, crop_x_max, crop_y_min, crop_y_max = find_mask_sides(query_mask_canvas)
    crop_x_min = max(0, crop_x_min - crop_buffer)
    crop_x_max = min(query_mask_canvas_height, crop_x_max + crop_buffer)
    crop_y_min = max(0, crop_y_min - crop_buffer)
    crop_y_max = min(query_mask_canvas_width, crop_y_max + crop_buffer)

    # Conditional padding
    sides_touched = check_touch_boundary(query_mask)
    if "top" in sides_touched:
        crop_x_min -= pad_pixels
    if "bottom" in sides_touched:
        crop_x_max += pad_pixels
    if "left" in sides_touched:
        crop_y_min -= pad_pixels
    if "right" in sides_touched:
        crop_y_max += pad_pixels
    
    # Compute target cropped region size
    crop_height = crop_x_max - crop_x_min
    crop_width = crop_y_max - crop_y_min
    crop_target_size = max(crop_height, crop_width)

    # Update cropped region to square
    if crop_width < crop_target_size:
        crop_target_size_diff = (crop_target_size - crop_width) / 2
        crop_y_min -= math.floor(crop_target_size_diff)
        crop_y_max += math.ceil(crop_target_size_diff)
    if crop_height < crop_target_size:
        crop_target_size_diff = (crop_target_size - crop_height) / 2
        crop_x_min -= math.floor(crop_target_size_diff)
        crop_x_max += math.ceil(crop_target_size_diff)
    return crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_target_size


def crop(inputs, query_mask, query_mask_canvas):
    """
    Apply crop region to all inputs
    """
    crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_target_size = find_crop_region(query_mask, query_mask_canvas)
    return [input_arr[crop_x_min : crop_x_max, crop_y_min : crop_y_max].astype(np.uint8) for input_arr in inputs], crop_x_min, crop_x_max, crop_y_min, crop_y_max


def compute_iou(mask1, mask2):
    overlap = mask1 + mask2
    overlap[overlap < 2] = 0; overlap[overlap == 2] = 1
    intersection = overlap.sum()

    union = mask1 + mask2
    union[union == 0] = 0; union[union > 0] = 1
    union = union.sum()
    return intersection / union


def filter_out_amodal_segmentation(crop_query_mask, amodal_masks):
    """
    Filter out the amodal segmentation from instance segmentation mask candidates
    """
    # When no seg masks detected, treat the original modal mask as the amodal mask
    amodal_segmentation = crop_query_mask
    max_iou = 0
    amodal_i = 0
    for i, amodal_mask in enumerate(amodal_masks):
        iou = compute_iou(crop_query_mask, amodal_mask.astype(np.uint8))
        if iou > max_iou:
            amodal_segmentation = amodal_mask
            max_iou = iou
            amodal_i = i
    return amodal_i, amodal_segmentation


def check_occlusion(
    amodal_completion,
    crop_query_mask,
    query_class,
    mask_id,
    gdino_model,
    sam_ckpt,
    instaorder_model,
    classes,
    query_obj,
    crop_x_min,
    crop_x_max,
    crop_y_min,
    crop_y_max,
    save_interm=False,
):
    """
    Check whether the query object remains occluded
    """
    amodal_completion_tensor = transform_image(amodal_completion)
    amodal_completion, amodal_segmentations, _, _ = segment(gdino_model, run_sam, sam_ckpt, amodal_completion, amodal_completion_tensor, classes)
    if amodal_segmentations is None:
        query_obj.run_iter = False
        query_obj.amodal_segmentation = None
        return query_obj  # If no masks are detected, then proceed to the next object

    amodal_i, amodal_segmentation = filter_out_amodal_segmentation(crop_query_mask, amodal_segmentations)
    query_obj.amodal_segmentation = amodal_segmentation.astype(np.uint8)

    new_occluder_masks = analyze_masks(instaorder_model, amodal_completion, amodal_segmentations, amodal_i)
    new_occluder_mask = aggregate_occluders(amodal_segmentation, new_occluder_masks, query_class, amodal_i, query_obj.iter_id, save_interm=save_interm, output_img_dir=query_obj.output_img_dir)
    
    # Update canvas with new amodal completion
    query_obj.img_canvas[crop_x_min : crop_x_max, crop_y_min : crop_y_max] = amodal_completion
    query_obj.query_mask_canvas[crop_x_min : crop_x_max, crop_y_min : crop_y_max] = amodal_segmentation
    query_obj.occ_mask_canvas[crop_x_min : crop_x_max, crop_y_min : crop_y_max] = new_occluder_mask
    query_obj.outpaint_mask_canvas[crop_x_min : crop_x_max, crop_y_min : crop_y_max] = np.zeros(amodal_segmentation.shape)

    amodal_sides_touched = check_touch_boundary(amodal_segmentation)
    if new_occluder_mask.sum() > 0 or len(amodal_sides_touched) > 0:
        query_obj.run_iter = True
    else:
        query_obj.run_iter = False
    return query_obj


def compute_offset(expanded_query_mask, init_outpainting_mask, amodal_segmentation):
    query_x_arr, query_y_arr = np.where(expanded_query_mask == 1)
    query_x_coord = min(query_x_arr); query_y_coord = min(query_y_arr)

    orig_x_arr, orig_y_arr = np.where(init_outpainting_mask == 0)
    orig_x_coord = min(orig_x_arr); orig_y_coord = min(orig_y_arr)

    amodal_seg_x_arr, amodal_seg_y_arr = np.where(amodal_segmentation == 1)
    amodal_x_coord = min(amodal_seg_x_arr); amodal_y_coord = min(amodal_seg_y_arr)

    x_offset = int(query_x_coord - amodal_x_coord - orig_x_coord)
    y_offset = int(query_y_coord - amodal_y_coord - orig_y_coord)
    return x_offset, y_offset


def run_mixed_context_diffusion(
    query_obj,
    sd_inpaint_model,
    lama_model,
    sd_img,
    sd_modal_mask,
    sd_occ_mask,
    sd_prompt,
    clean_bkgd_img,
    mc_timestep=35,
    sd_target_size=512,
):
    # Swap background
    clean_bkgd_mask = 1 - sd_modal_mask
    clean_bkgd_x_arr, clean_bkgd_y_arr = np.where(clean_bkgd_mask == 1)
    sd_img_syn = sd_img.copy()
    sd_img_syn[clean_bkgd_x_arr, clean_bkgd_y_arr] = clean_bkgd_img[clean_bkgd_x_arr, clean_bkgd_y_arr]

    # Create object-removed background image
    modal_occ_mask = sd_modal_mask + sd_occ_mask
    modal_occ_mask[modal_occ_mask > 0] = 1
    modal_occ_mask = np.expand_dims(modal_occ_mask, axis=-1)
    object_removed_bkgd_img = run_lama_inpainter(sd_img, modal_occ_mask, lama_model)

    modal_x_arr, modal_y_arr = np.where(sd_modal_mask == 1)
    object_removed_bkgd_img[modal_x_arr, modal_y_arr] = sd_img[modal_x_arr, modal_y_arr]

    sd_img = Image.fromarray(sd_img).convert("RGB")
    sd_img_syn = Image.fromarray(sd_img_syn).convert("RGB")
    sd_occ_mask = Image.fromarray(sd_occ_mask * 255).convert("L")
    sd_modal_mask = Image.fromarray(sd_modal_mask * 255).convert("L")
    object_removed_bkgd_img = Image.fromarray(object_removed_bkgd_img).convert("RGB")

    # Segment object in noisy image, and composite
    mc_output = sd_inpaint_model(
        prompt = sd_prompt,
        image = sd_img_syn,
        mask_image = sd_occ_mask,
        mixed_context = True,
        mixed_context_timestep = mc_timestep,
        mixed_context_up_ft_indices = [2], # Choose from [0, 1, 2, 3]
        query_mask_image = sd_modal_mask,
        object_removed_image = object_removed_bkgd_img,
    )
    amodal_completion = mc_output.images[0]
    return amodal_completion


def run_iteration(
    query_obj,
    output_dir,
    masks,
    classes,
    class_names,
    pred_scores,
    gdino_model,
    sam_ckpt,
    instaorder_model,
    sd_inpaint_model,
    lama_model,
    mc_timestep,
    mc_clean_bkgd_img,
    sd_target_size=512,
    save_interm=True,  # Whether to save intermediate images
):
    """
    Returns whether to run an additional iteration
    """
    # Check whether query object is valid for amodal completion
    img = query_obj.img
    mask_id = query_obj.mask_id
    query_mask = query_obj.query_mask

    check_query = check_valid_query(img, mask_id, query_mask, class_names, pred_scores, classes, save_interm=save_interm, output_img_dir=query_obj.output_img_dir)
    if check_query is None:
        query_obj.run_iter = False
        query_obj.amodal_segmentation = None
        return query_obj
    query_mask, query_class = check_query
    query_obj.query_class = query_class

    # Analyze masks to determine occluders
    occluder_masks = analyze_masks(instaorder_model, img, masks, mask_id)
    occ_mask = aggregate_occluders(query_mask, occluder_masks, query_class, mask_id, query_obj.iter_id, save_interm=save_interm, output_img_dir=query_obj.output_img_dir)

    # Check occlusion by image boundary
    sides_touched = check_touch_boundary(query_mask)
    query_obj.run_iter = True if (occ_mask.sum() > 0 or len(sides_touched) > 0) else False
    if not query_obj.run_iter:
        query_obj.amodal_segmentation = None
        return query_obj

    if query_obj.iter_id == 0:
        # Preprocess the img, query mask, and occluder mask
        img_canvas = create_canvas(img, size_multiplier=6, canvas_val=255)
        query_mask_canvas = create_canvas(query_mask, size_multiplier=6, canvas_val=0)
        occ_mask_canvas = create_canvas(occ_mask, size_multiplier=6, canvas_val=0)
        outpaint_mask_canvas = create_canvas(np.zeros((query_mask.shape)), size_multiplier=6, canvas_val=1)

        # Save init image and mask canvas
        query_obj.img_canvas = img_canvas
        query_obj.query_mask_canvas = query_mask_canvas
        query_obj.occ_mask_canvas = occ_mask_canvas
        query_obj.outpaint_mask_canvas = outpaint_mask_canvas

        query_obj.init_img_canvas = img_canvas.copy()
        query_obj.init_query_mask_canvas = query_mask_canvas.copy()
        query_obj.init_occ_mask_canvas = occ_mask_canvas.copy()
        query_obj.init_outpaint_mask_canvas = outpaint_mask_canvas.copy()

    # Crop image and mask canvas
    crop_inputs = [query_obj.img_canvas, query_obj.query_mask_canvas, query_obj.occ_mask_canvas, query_obj.outpaint_mask_canvas]
    crop_outputs, crop_x_min, crop_x_max, crop_y_min, crop_y_max = crop(crop_inputs, query_mask, query_obj.init_query_mask_canvas)
    crop_img, crop_query_mask, crop_occ_mask, crop_outpaint_mask = crop_outputs

    # Create input bundle for Stable Diffusion
    sd_img = crop_img
    sd_modal_mask = crop_query_mask
    sd_occ_mask = crop_occ_mask + crop_outpaint_mask
    sd_occ_mask[sd_occ_mask > 0] = 1
    sd_prompt = query_class

    # Run diffusion inpainting
    input_height, input_width = sd_img.shape[0], sd_img.shape[1]
    kernel = np.ones((5, 5), np.uint8)
    sd_occ_mask = cv2.dilate(sd_occ_mask, kernel, iterations=3).astype(np.uint8)
    if mc_timestep is None:
        # Convert to PIL Image
        sd_img = Image.fromarray(sd_img).convert("RGB").resize((sd_target_size, sd_target_size))
        sd_occ_mask = Image.fromarray(255 * sd_occ_mask).convert("L").resize((sd_target_size, sd_target_size), resample=Image.NEAREST)

        amodal_completion = sd_inpaint_model(
            image=sd_img,
            mask_image=sd_occ_mask,
            prompt=sd_prompt,
        ).images[0]
    else:
        clean_bkgd_img = np.array(Image.open(mc_clean_bkgd_img).convert("RGB"))
        amodal_completion = run_mixed_context_diffusion(
            query_obj,
            sd_inpaint_model,
            lama_model,
            sd_img,
            sd_modal_mask,
            sd_occ_mask,
            sd_prompt,
            clean_bkgd_img,
            mc_timestep,
            sd_target_size,
        )

    query_obj.iter_id += 1
    amodal_completion = amodal_completion.resize((input_width, input_height))
    query_obj.amodal_completion = amodal_completion

    # Occlusion check
    query_obj = check_occlusion(
        amodal_completion,
        sd_modal_mask,
        query_class,
        mask_id,
        gdino_model,
        sam_ckpt,
        instaorder_model,
        classes,
        query_obj,
        crop_x_min,
        crop_x_max,
        crop_y_min,
        crop_y_max,
        save_interm,
    )
    return query_obj


def run_pipeline(args):
    img_filenames = read_txt(args.img_filenames_txt)
    gdino_model, sd_inpaint_model, instaorder_model, lama_model = load_models(args.gdino_config, args.gdino_ckpt, args.instaorder_ckpt, args.lama_config_path, args.lama_ckpt_path, args.mc_timestep)
    classes = read_txt(args.classes_txt)
    os.makedirs(args.output_dir, exist_ok=True)

    for img_filename in tqdm(img_filenames, desc="Iterate images"):
        img_basename = img_filename.split(".")[0]
        img_path = os.path.join(args.input_dir, img_filename)
        img_pil = Image.open(img_path).convert('RGB')
        output_img_dir = os.path.join(args.output_dir, img_basename)

        if os.path.exists(output_img_dir):
            shutil.rmtree(output_img_dir)

        # Create output directories
        os.makedirs(output_img_dir, exist_ok=True)
        if args.save_interm:
            subdirs = ["amodal_completions", "amodal_segmentations"]
            for subdir in subdirs:
                os.makedirs(os.path.join(output_img_dir, subdir), exist_ok=True)

        # Perform instance segmentation
        img_tensor = transform_image(img_pil, save_interm=args.save_interm, output_img_dir=output_img_dir)
        img, masks, class_names, pred_scores = segment(gdino_model, run_sam, args.sam_ckpt, img_pil, img_tensor, classes)
        if masks is None: continue  # If no masks are detected, then proceed to the next image

        img_offsets_dict = {}

        for mask_id, query_mask in enumerate(masks):
            query_obj = QueryObject(img_path, img, img_pil, mask_id, query_mask, output_img_dir)

            while query_obj.run_iter:
                query_obj = run_iteration(
                    query_obj,
                    args.output_dir,
                    masks,
                    classes,
                    class_names,
                    pred_scores,
                    gdino_model,
                    args.sam_ckpt,
                    instaorder_model,
                    sd_inpaint_model,
                    lama_model,
                    args.mc_timestep,
                    args.mc_clean_bkgd_img,
                    save_interm=args.save_interm,
                )
                if query_obj.iter_id > args.max_iter_id: break

            # Post-processing
            if query_obj.amodal_segmentation is not None and query_obj.iter_id > 0:
                query_class = query_obj.query_class
                x_offset, y_offset = compute_offset(query_obj.query_mask_canvas, query_obj.init_outpaint_mask_canvas, query_obj.amodal_segmentation)
                img_offsets_dict[f'{query_class}_{query_obj.mask_id}'] = [x_offset, y_offset]
                img_offset_save_path = os.path.join(query_obj.output_img_dir, "offsets.json")
                with open(img_offset_save_path, 'w') as fp:
                    json.dump(img_offsets_dict, fp, sort_keys=True, indent=4)

                amodal_completion_to_save = query_obj.amodal_completion
                amodal_completion_to_save.save(os.path.join(query_obj.output_img_dir, "amodal_completions", f'{query_class}_{mask_id}.jpg'), quality=90)
                amodal_segmentation_to_save = Image.fromarray(query_obj.amodal_segmentation * 255).convert("RGB")
                amodal_segmentation_to_save.save(os.path.join(query_obj.output_img_dir, "amodal_segmentations", f'{query_class}_{mask_id}.png'))


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
