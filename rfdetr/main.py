
# rfdetr/main.py

import argparse
from pathlib import Path

from rfdetr.util.model import build_model_wrapper

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--subcommand', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--infer_dir', type=str, default=None)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--opset_version', type=int, default=17)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--shape', type=int, nargs=2, default=(640, 640))
    parser.add_argument('--data', type=str, default=None)
    return parser

def populate_args(**kwargs):
    return argparse.Namespace(**kwargs)

def download_pretrain_weights(pretrain_weights: str, redownload=False):
    import os
    from rfdetr.util.files import download_file
    HOSTED_MODELS = {
        "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
        "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth"
    }
    if pretrain_weights in HOSTED_MODELS:
        if redownload or not os.path.exists(pretrain_weights):
            print(f"Downloading pretrained weights for {pretrain_weights}")
            download_file(HOSTED_MODELS[pretrain_weights], pretrain_weights)

def main(**kwargs):
    model = build_model_wrapper(kwargs)

    if "data" in kwargs:
        args, resolution = model.load_data(kwargs["data"])
        from rfdetr.datasets import build_dataset
        from torch.utils.data import DataLoader
        print(" Loading datasets...")
        dataset_train = build_dataset(image_set='train', args=args, resolution=resolution)
        dataset_val = build_dataset(image_set='val', args=args, resolution=resolution)

        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=model.collate_fn)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=model.collate_fn)

        model.fit(train_dataloader=dataloader_train, val_dataloader=dataloader_val, epochs=kwargs.get("epochs", 10))
        model.save("output_model")
    else:
        print(" No dataset path provided via --data")
    model.train(callbacks={
        "on_fit_epoch_end": [],
        "on_train_batch_start": [],
        "on_train_end": []
    }, **kwargs)

def distill(**kwargs):
    raise NotImplementedError("Distillation training is not implemented yet.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = vars(args)

    if args.subcommand == 'distill':
        distill(**config)
    elif args.subcommand == 'export_model':
        from rfdetr.deploy.export import main as export_main
        if args.batch_size != 1:
            config['batch_size'] = 1
            print("Only batch_size 1 is supported for onnx export. Forcibly setting batch_size = 1.")
        export_main(**config)
    else:
        if hasattr(args, 'backbone') and args.backbone:
            config['encoder'] = args.backbone
        main(**config)
