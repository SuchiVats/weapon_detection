# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:53:05 2025

@author: karth
"""
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import supervision as sv
import cv2
import numpy as np

from rfdetr.util.model import build_model_wrapper
from rfdetr.util.misc import collate_fn, NestedTensor
from rfdetr.util.box_ops import box_cxcywh_to_xyxy
from rfdetr.datasets import build_dataset


CLASS_NAMES = ["Axe", "Crowbar", "Knife", "Grenade", "Long_gun", "Pistol", "Sickle", "Sword"]

def draw_boxes_supervision(image_tensor, boxes, labels, scores, class_names, threshold=0.3):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    image_tensor = image_tensor * std + mean

    image = image_tensor.clamp(0, 1).mul(255).byte().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]  # BGR for OpenCV

    filtered_boxes, filtered_labels, filtered_scores = [], [], []
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        filtered_boxes.append(box.tolist())
        filtered_labels.append(int(label))
        filtered_scores.append(float(score))

    if not filtered_boxes:
        return image

    detections = sv.Detections(
        xyxy=np.array(filtered_boxes, dtype=np.float32),
        class_id=np.array(filtered_labels),
        confidence=np.array(filtered_scores)
    ).with_nms(threshold=0.3)  # Adjusted to match inference threshold

    color_palette = sv.ColorPalette.from_hex(["#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A", "#92CC17", "#3DDB86"])
    bbox_annotator = sv.BoxAnnotator(thickness=2, color=color_palette)
    return bbox_annotator.annotate(scene=image.copy(), detections=detections)

def run_inference(model, data_loader, postprocessors, device, threshold=0.3, vis_dir=None):
    model.eval()
    results = []

    os.makedirs(vis_dir, exist_ok=True) if vis_dir else None

    with torch.no_grad():
        for idx, (samples, targets) in enumerate(tqdm(data_loader, desc="Running Inference")):
            samples = samples.to(device)
            outputs = model(samples)

            prob = outputs["pred_logits"].softmax(-1)[0, :, :-1]
            keep = prob.max(-1).values > threshold
            boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][0, keep])
            scores, labels = prob[keep].max(-1)
            image_id = targets[0]["image_id"].item()

            height, width = samples.tensors.shape[-2:]
            boxes = boxes.cpu() * torch.tensor([width, height, width, height], dtype=torch.float32)

            for box, score, label in zip(boxes, scores, labels):
                results.append({
                    "image_id": image_id,
                    "class": CLASS_NAMES[label],
                    "score": score.item(),
                    "xmin": box[0].item(),
                    "ymin": box[1].item(),
                    "xmax": box[2].item(),
                    "ymax": box[3].item(),
                })

            if vis_dir:
                vis_image = draw_boxes_supervision(samples.tensors[0], boxes, labels, scores, CLASS_NAMES, threshold)
                vis_path = os.path.join(vis_dir, f"{image_id}.jpg")
                cv2.imwrite(vis_path, vis_image)

    return results




def main(args):
    args.encoder = args.backbone
    args.projector_scale = args.projector_scale or ["P4"]
    args.batch_size = getattr(args, "batch_size", 1)
    args.lr = getattr(args, "lr", 1e-4)
    args.epochs = getattr(args, "epochs", 1)
    args.output_dir = getattr(args, "output_dir", "checkpoints")
    args.dataset_file = getattr(args, "dataset_file", "coco")
    args.num_queries = getattr(args, "num_queries", 100)
    args.hidden_dim = getattr(args, "hidden_dim", 256)
    args.position_embedding = getattr(args, "position_embedding", "sine")
    args.vit_encoder_num_layers = getattr(args, "vit_encoder_num_layers", 12)
    args.pretrained_encoder = getattr(args, "pretrained_encoder", False)
    args.window_block_indexes = getattr(args, "window_block_indexes", [])
    args.out_feature_indexes = getattr(args, "out_feature_indexes", [2, 5, 8, 11])
    args.use_cls_token = getattr(args, "use_cls_token", False)
    args.shape = getattr(args, "shape", 640)
    args.decoder_norm = getattr(args, "decoder_norm", "LN")
    args.sa_nheads = getattr(args, "sa_nheads", 8)
    args.ca_nheads = getattr(args, "ca_nheads", 8)
    args.dec_n_points = getattr(args, "dec_n_points", 4)
    args.enc_layers = getattr(args, "enc_layers", 6)
    args.dec_layers = getattr(args, "dec_layers", 6)
    args.dim_feedforward = getattr(args, "dim_feedforward", 2048)
    args.dropout = getattr(args, "dropout", 0.1)
    args.activation = getattr(args, "activation", "relu")
    args.nheads = getattr(args, "nheads", 8)
    args.cls_loss_coef = getattr(args, "cls_loss_coef", 2.0)
    args.bbox_loss_coef = getattr(args, "bbox_loss_coef", 5.0)
    args.giou_loss_coef = getattr(args, "giou_loss_coef", 2.0)
    args.aux_loss = getattr(args, "aux_loss", False)
    args.drop_path = getattr(args, "drop_path", 0.0)
    args.focal_alpha = getattr(args, "focal_alpha", 0.25)
    args.group_detr = getattr(args, "group_detr", 1)
    args.two_stage = getattr(args, "two_stage", False)
    args.lite_refpoint_refine = getattr(args, "lite_refpoint_refine", False)
    args.bbox_reparam = getattr(args, "bbox_reparam", False)
    args.freeze_encoder = getattr(args, "freeze_encoder", False)
    args.layer_norm = getattr(args, "layer_norm", False)
    args.rms_norm = getattr(args, "rms_norm", False)
    args.backbone_lora = getattr(args, "backbone_lora", False)
    args.force_no_pretrain = getattr(args, "force_no_pretrain", False)
    args.num_select = getattr(args, "num_select", 300)
    args.sum_group_losses = getattr(args, "sum_group_losses", False)
    args.use_varifocal_loss = getattr(args, "use_varifocal_loss", False)
    args.use_position_supervised_loss = getattr(args, "use_position_supervised_loss", False)
    args.ia_bce_loss = getattr(args, "ia_bce_loss", False)

    model, _, postprocessors, _, resolution = build_model_wrapper(args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    print("Loading test dataset...")
    test_dataset = build_dataset("test", args, resolution)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    results = run_inference(model, test_loader, postprocessors, args.device, args.threshold, vis_dir=args.vis_dir)

    print(f"Saving results to: {args.output_csv}")
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--backbone", default="facebook/dinov2-base")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--projector_scale", nargs="+", default=["P4"])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_csv", default="inference_results.csv")
    parser.add_argument("--vis_dir", default="inference_vis")
    args = parser.parse_args()
    main(args)
