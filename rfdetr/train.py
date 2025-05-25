import os
import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from rfdetr.util.model import build_model_wrapper
from rfdetr.util.misc import NestedTensor, collate_fn
from rfdetr.datasets import build_dataset


def get_args_parser():
    parser = argparse.ArgumentParser("RF-DETR Training", add_help=False)
    parser.add_argument("--coco_path", type=str, required=True)
    parser.add_argument("--dataset_file", default="coco", type=str)

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="checkpoints", type=str)
    parser.add_argument("--backbone", default="facebook/dinov2-base", type=str)

    parser.add_argument("--encoder_only", action="store_true")
    parser.add_argument("--backbone_only", action="store_true")
    parser.add_argument("--pretrain_weights", default=None, type=str)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--num_queries", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--position_embedding", default="sine", type=str)

    parser.add_argument("--vit_encoder_num_layers", default=12, type=int)
    parser.add_argument("--pretrained_encoder", action="store_true")
    parser.add_argument("--window_block_indexes", nargs="*", type=int, default=[])
    parser.add_argument("--out_feature_indexes", nargs="*", default=[2, 5, 8, 11], type=int)
    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--shape", default=640, type=int)

    parser.add_argument("--decoder_norm", default="LN", type=str, choices=["LN", "Identity"])
    parser.add_argument("--sa_nheads", default=8, type=int)
    parser.add_argument("--ca_nheads", default=8, type=int)
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--nheads", default=8, type=int)

    parser.add_argument("--cls_loss_coef", default=2.0, type=float)
    parser.add_argument("--bbox_loss_coef", default=5.0, type=float)
    parser.add_argument("--giou_loss_coef", default=2.0, type=float)
    parser.add_argument("--aux_loss", action="store_true")
    parser.add_argument("--drop_path", default=0.0, type=float)
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

    parser.add_argument("--projector_scale", nargs="+", default=["P3", "P4", "P5"])
    parser.add_argument("--num_select", default=300, type=int)
    parser.add_argument("--sum_group_losses", action="store_true")
    parser.add_argument("--use_varifocal_loss", action="store_true")
    parser.add_argument("--use_position_supervised_loss", action="store_true")
    parser.add_argument("--ia_bce_loss", action="store_true")

    return parser


def main(args):
    print(f"Using device: {args.device}")
    print(f"Loading {args.data}: {args.data}")

    if not hasattr(args, "projector_scale") or not isinstance(args.projector_scale, list):
        print("Fixing projector_scale: resetting to ['P3', 'P4', 'P5']")
        args.projector_scale = ["P3", "P4", "P5"]
    else:
        print(f"Parsed projector_scale: {args.projector_scale}")

    if not hasattr(args, "encoder") or args.encoder is None:
        print("Setting args.encoder to args.backbone...")
        args.encoder = args.backbone

    print(f"CLI arg backbone: {args.backbone}")
    print(f"Final args.backbone going to build_model: {args.backbone}")

    model, criterion, postprocessors, args, resolution = build_model_wrapper(args)
    model.to(args.device)

    print("Building datasets...")
    dataset_train = build_dataset("train", args, resolution)
    dataset_val = build_dataset("val", args, resolution)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("Starting WandB logging...")
    wandb.init(project="tattoo-rfdetr", config=vars(args))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for samples, targets in data_loader_train:
            if isinstance(samples, list) and len(samples) > 0:
                tensor_list = [s.tensors for s in samples if hasattr(s, "tensors")]
                mask_list = [s.mask for s in samples if hasattr(s, "mask") and s.mask is not None]
                if len(tensor_list) > 0 and len(mask_list) > 0:
                    tensors = torch.stack(tensor_list)
                    masks = torch.stack(mask_list)
                    samples = NestedTensor(tensors, masks)
                else:
                    print(" Skipping batch due to missing tensors or masks.")
                    continue
            else:
                print(" Skipping batch due to empty or invalid sample format.")
                continue

            samples = samples.to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] for k in loss_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(data_loader_train)

        model.eval()
        total_val_loss = 0
        coco_evaluator = postprocessors["bbox"].coco_evaluator
        for samples, targets in data_loader_val:
            if isinstance(samples, list) and len(samples) > 0:
                tensor_list = [s.tensors for s in samples if hasattr(s, "tensors")]
                mask_list = [s.mask for s in samples if hasattr(s, "mask") and s.mask is not None]
                if len(tensor_list) > 0 and len(mask_list) > 0:
                    tensors = torch.stack(tensor_list)
                    masks = torch.stack(mask_list)
                    samples = NestedTensor(tensors, masks)
                else:
                    continue
            else:
                continue

            samples = samples.to(args.device)
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] for k in loss_dict)
            total_val_loss += loss.item()

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        val_map = coco_evaluator.coco_eval["bbox"].stats[0]

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": total_val_loss / len(data_loader_val),
            "val_mAP": val_map
        })

        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth"))
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val mAP: {val_map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RF-DETR Training Script", parents=[get_args_parser()])
    args = parser.parse_args()

    if not hasattr(args, "encoder_only"):
        args.encoder_only = False
    if not hasattr(args, "backbone_only"):
        args.backbone_only = False
    if not hasattr(args, "projector_scale") or not isinstance(args.projector_scale, list):
        args.projector_scale = ["P3", "P4", "P5"]

    main(args)
