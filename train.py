import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from rfdetr.util.model import build_model_wrapper
from rfdetr.util.misc import NestedTensor, collate_fn, validate_dataset
from rfdetr.datasets import build_dataset
import torch.nn.functional as F

def get_args_parser():
    parser = argparse.ArgumentParser("RF-DETR Training", add_help=False)
    parser.add_argument("--coco_path", type=str, required=True)
    parser.add_argument("--dataset_file", default="coco", type=str)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="checkpoints", type=str)
    parser.add_argument("--backbone", default="facebook/dinov2-base", type=str)
    parser.add_argument("--encoder_only", action="store_true")
    parser.add_argument("--backbone_only", action="store_true")
    parser.add_argument("--pretrain_weights", default=None, type=str)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--num_queries", default=25, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--position_embedding", default="sine", type=str)
    parser.add_argument("--vit_encoder_num_layers", default=12, type=int)
    parser.add_argument("--pretrained_encoder", action="store_true")
    parser.add_argument("--window_block_indexes", nargs="*", type=int, default=[])
    parser.add_argument("--out_feature_indexes", nargs="*", default=[2, 5, 8, 11], type=int)
    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--shape", default=640, type=int)
    parser.add_argument("--decoder_norm", default="LN", type=str)
    parser.add_argument("--sa_nheads", default=8, type=int)
    parser.add_argument("--ca_nheads", default=8, type=int)
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--cls_loss_coef", default=2.0, type=float)
    parser.add_argument("--bbox_loss_coef", default=5.0, type=float)
    parser.add_argument("--giou_loss_coef", default=2.0, type=float)
    parser.add_argument("--aux_loss", action="store_true")
    parser.add_argument("--drop_path", default=0.1, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--group_detr", default=1, type=int)
    parser.add_argument("--two_stage", action="store_true")
    parser.add_argument("--lite_refpoint_refine", action="store_true")
    parser.add_argument("--bbox_reparam", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--rms_norm", action="store_true")
    parser.add_argument("--backbone_lora", action="store_true")
    parser.add_argument("--force_no_pretrain", action="store_true")
    parser.add_argument("--projector_scale", nargs="+", default=["P4"])
    parser.add_argument("--num_select", default=300, type=int)
    parser.add_argument("--sum_group_losses", action="store_true")
    parser.add_argument("--use_varifocal_loss", action="store_true")
    parser.add_argument("--use_position_supervised_loss", action="store_true")
    parser.add_argument("--ia_bce_loss", action="store_true")
    return parser

def is_valid_nested_tensor(sample):
    return (
        isinstance(sample, NestedTensor)
        and hasattr(sample, "mask")
        and sample.mask is not None
        and isinstance(sample.mask, torch.Tensor)
        and sample.mask.numel() > 0
        and sample.mask.ndim == 3
    )

def main(args):
    print(f"Using device: {args.device}")
    print(f"Loading dataset config: {args.data}")

    if not isinstance(args.projector_scale, list):
        args.projector_scale = ["P4"]

    if not hasattr(args, "encoder") or args.encoder is None:
        args.encoder = args.backbone

    model, criterion, postprocessors, args, resolution = build_model_wrapper(args)
    model.to(args.device)

    print("Building datasets...")
    dataset_train = build_dataset("train", args, resolution)
    dataset_val = build_dataset("val", args, resolution)
    validate_dataset(dataset_train)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(device=args.device)

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        skipped_batches = 0

        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}", unit="batch")
        for samples, targets in pbar:
            if samples is None or targets is None or not is_valid_nested_tensor(samples):
                print(" Skipping batch: invalid NestedTensor or mask")
                skipped_batches += 1
                continue

            samples = samples.to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            with autocast(device_type=args.device):
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                loss_dict = {k: v for k, v in loss_dict.items() if "bbox" in k or "giou" in k}
                loss = sum(loss_dict[k] for k in loss_dict)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())

        avg_train_loss = total_loss / max(1, len(data_loader_train) - skipped_batches)

        model.eval()
        total_val_loss = 0

        val_pbar = tqdm(data_loader_val, desc=f"Val Epoch {epoch+1}", unit="batch")
        for samples, targets in val_pbar:
            if samples is None or targets is None or not is_valid_nested_tensor(samples):
                continue

            samples = samples.to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            for i in range(len(samples.mask)):
                mask = samples.mask[i]
                tensor = samples.tensors[i]
                if mask.shape[-2:] != tensor.shape[-2:]:
                    resized = F.interpolate(mask[None].float(), size=tensor.shape[-2:]).to(torch.bool)[0]
                    samples.mask[i] = resized

            with torch.no_grad():
                with autocast(device_type=args.device):
                    outputs = model(samples)
                    loss_dict = criterion(outputs, targets)
                    loss_dict = {k: v for k, v in loss_dict.items() if "bbox" in k or "giou" in k}
                    loss = sum(loss_dict[k] for k in loss_dict)
                total_val_loss += loss.item()
                val_pbar.set_postfix(val_loss=loss.item())

        avg_val_loss = total_val_loss / len(data_loader_val)
        os.makedirs(args.output_dir, exist_ok=True)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth"))

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Skipped: {skipped_batches}")
        print(" Loss breakdown:")
        for k, v in loss_dict.items():
            print(f"  🔸 {k}: {v.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("RF-DETR Training Script", parents=[get_args_parser()])
    args = parser.parse_args()

    args.encoder_only = False
    args.backbone_only = False
    args.projector_scale = ["P4"]
    args.aux_loss = True
    args.dropout = 0.3
    args.drop_path = 0.1
    args.epochs = 50
    args.lr = 5e-5

    main(args)