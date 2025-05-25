import argparse
from rfdetr.models.lwdetr import build_lw_detr
import os

def build_model_wrapper(cli_args):
    args = argparse.Namespace()

    # Basic CLI -> config mappings
    args.backbone = cli_args.backbone
    args.encoder = getattr(cli_args, "encoder", cli_args.backbone)
    args.coco_path = getattr(cli_args, "coco_path", os.path.dirname(cli_args.data))

    args.device = cli_args.device
    args.lr = cli_args.lr
    args.num_classes = cli_args.num_classes
    args.batch_size = cli_args.batch_size
    args.output_dir = cli_args.output_dir
    args.epochs = cli_args.epochs
    args.dataset_file = getattr(cli_args, "dataset_file", "coco")
    args.multi_scale = getattr(cli_args, "multi_scale", False)
    args.expanded_scales = getattr(cli_args, "expanded_scales", [0.1, 0.5, 1.0, 1.5])



    # Model flags
    args.encoder_only = getattr(cli_args, "encoder_only", False)
    args.backbone_only = getattr(cli_args, "backbone_only", False)
    args.pretrain_weights = getattr(cli_args, "pretrain_weights", None)

    args.num_queries = getattr(cli_args, "num_queries", 100)
    args.cls_loss_coef = getattr(cli_args, "cls_loss_coef", 2.0)
    args.bbox_loss_coef = getattr(cli_args, "bbox_loss_coef", 5.0)
    args.giou_loss_coef = getattr(cli_args, "giou_loss_coef", 2.0)
    args.aux_loss = getattr(cli_args, "aux_loss", False)
    args.drop_path = getattr(cli_args, "drop_path", 0.0)
    args.focal_alpha = getattr(cli_args, "focal_alpha", 0.25)
    args.group_detr = getattr(cli_args, "group_detr", 1)
    args.two_stage = getattr(cli_args, "two_stage", False)
    args.lite_refpoint_refine = getattr(cli_args, "lite_refpoint_refine", False)
    args.bbox_reparam = getattr(cli_args, "bbox_reparam", False)

    args.hidden_dim = getattr(cli_args, "hidden_dim", 256)
    args.position_embedding = getattr(cli_args, "position_embedding", "sine")
    args.freeze_encoder = getattr(cli_args, "freeze_encoder", False)
    args.layer_norm = getattr(cli_args, "layer_norm", False)
    args.rms_norm = getattr(cli_args, "rms_norm", False)
    args.backbone_lora = getattr(cli_args, "backbone_lora", False)
    args.force_no_pretrain = getattr(cli_args, "force_no_pretrain", False)
    args.projector_scale = getattr(cli_args, "projector_scale", ["P3", "P4", "P5"])

    # Transformer-related
    args.sa_nheads = getattr(cli_args, "sa_nheads", 8)
    args.ca_nheads = getattr(cli_args, "ca_nheads", 8)
    args.dim_feedforward = getattr(cli_args, "dim_feedforward", 2048)
    args.enc_layers = getattr(cli_args, "enc_layers", 6)
    args.dec_layers = getattr(cli_args, "dec_layers", 6)
    args.dropout = getattr(cli_args, "dropout", 0.1)
    args.activation = getattr(cli_args, "activation", "relu")
    args.nheads = getattr(cli_args, "nheads", 8)
    args.pre_norm = getattr(cli_args, "pre_norm", False)
    args.num_feature_levels = getattr(cli_args, "num_feature_levels", 4)
    args.decoder_norm = getattr(cli_args, "decoder_norm", "LN")
    args.dec_n_points = getattr(cli_args, "dec_n_points", 4)

    # DINOv2-specific
    args.vit_encoder_num_layers = getattr(cli_args, "vit_encoder_num_layers", 12)
    args.pretrained_encoder = getattr(cli_args, "pretrained_encoder", False)
    args.window_block_indexes = getattr(cli_args, "window_block_indexes", [])
    args.out_feature_indexes = getattr(cli_args, "out_feature_indexes", [2, 5, 8, 11])
    args.use_cls_token = getattr(cli_args, "use_cls_token", False)
    args.shape = getattr(cli_args, "shape", 640)

    # Matcher related
    args.set_cost_class = getattr(cli_args, "set_cost_class", 2.0)
    args.set_cost_bbox = getattr(cli_args, "set_cost_bbox", 5.0)
    args.set_cost_giou = getattr(cli_args, "set_cost_giou", 2.0)

    # Evaluation helpers
    args.use_varifocal_loss = getattr(cli_args, "use_varifocal_loss", False)
    args.use_position_supervised_loss = getattr(cli_args, "use_position_supervised_loss", False)
    args.ia_bce_loss = getattr(cli_args, "ia_bce_loss", False)
    args.num_select = getattr(cli_args, "num_select", 300)
    args.sum_group_losses = getattr(cli_args, "sum_group_losses", False)

    print("CLI arg backbone:", cli_args.backbone)
    print("Final args.backbone going to build_model:", args.backbone)

    model, criterion, postprocessors = build_lw_detr(args)
    resolution = 640
    return model, criterion, postprocessors, args, resolution
