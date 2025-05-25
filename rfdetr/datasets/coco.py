# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import torch
import torch.utils.data
import torchvision

import rfdetr.datasets.transforms as T


class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return T.resize(img, target, self.size)

def compute_multi_scale_scales(resolution, expanded_scales=False):
    if resolution == 640:
        patch_size = 16
    elif resolution % (14 * 4) == 0:
        patch_size = 14
    elif resolution % (16 * 4) == 0:
        patch_size = 16
    else:
        raise ValueError(f"Resolution {resolution} is not divisible by 16*4 or 14*4")

    base_num_patches_per_window = resolution // (patch_size * 4)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    return [scale * patch_size * 4 for scale in scales]


def collate_fn_remove_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCoco()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if target["boxes"].nelement() == 0:
            return None

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


class ConvertCoco(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, resolution, multi_scale=False, expanded_scales=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    fixed_res = (resolution // 14) * 14

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            ResizeTransform((fixed_res, fixed_res)),
            normalize,
        ])

    if image_set in ['val', 'test', 'val_speed']:  # 🟢 Include 'test' here
        return T.Compose([
            ResizeTransform((fixed_res, fixed_res)),
            normalize,
        ])

    raise ValueError(f'Unknown image_set: {image_set}')



def make_coco_transforms_square_div_64(image_set, resolution, multi_scale=False, expanded_scales=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [resolution]
    if multi_scale:
        scales = compute_multi_scale_scales(resolution, expanded_scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([resolution]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args, resolution):
    root = Path(args.coco_path)
    assert root.exists(), f'provided dataset path {root} does not exist'

    # Updated to match Roboflow/YOLO structure
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set, resolution,
                multi_scale=getattr(args, "multi_scale", False),
                expanded_scales=getattr(args, "expanded_scales", False),
            )
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set, resolution,
                multi_scale=getattr(args, "multi_scale", False),
                expanded_scales=getattr(args, "expanded_scales", False),
            )
        )

    return dataset



def build_roboflow(image_set, args, resolution):
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root /  "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)

    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(image_set, resolution, multi_scale=args.multi_scale))
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, resolution, multi_scale=args.multi_scale))

    return dataset
