# flash_x_datasets.py
#
#
#
#           low_res:  (32,  2048)   float32  range 0~100m
#
#
#   KITTI/CARLA: ScaleTensor(1/80)

import os
import numpy as np
import torch
import torchvision.transforms as transforms

from util.datasets import (
    register_dataset,
    RangeMapFolder,
    PairDataset,
    ScaleTensor,
    FilterInvalidPixels,
    LogTransform,
)


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────

def npy_loader_range_only(path: str) -> np.ndarray:
    """
    Handles both 2-channel npy (range + intensity) and 1-channel npy.
    Always returns the range channel (index 0) only.
    shape: (H, W, 2) -> (H, W) / (H, W) -> (H, W)
    """
    data = np.load(path).astype(np.float32)
    if data.ndim == 3:          # (H, W, 2)
        return data[..., 0]
    return data


def npy_loader_2d(path: str) -> np.ndarray:
    """
    For CARLA: returns the (H, W) 2D array as is.
    The original TULIP npy_loader indexes [..., 0], which crashes on a 2D array,
    so a separate loader is needed.
    """
    data = np.load(path).astype(np.float32)
    assert data.ndim == 2, f"CARLA loader: expected 2D array, got shape {data.shape}"
    return data


# ─────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────

@register_dataset('kitti_paired')
def build_kitti_paired_dataset(is_train, args):
    """
    KITTI paired: loads LR/HR files directly (no on-the-fly downsampling).
    Paths: {data_path}/{train|test}/high_res/*.npy
                                    /low_res/*.npy
    """
    split = 'train' if is_train else 'val'

    t_lr = [transforms.ToTensor(), ScaleTensor(1/80)]
    t_hr = [transforms.ToTensor(), ScaleTensor(1/80)]

    if args.log_transform:
        t_lr.append(LogTransform())
        t_hr.append(LogTransform())

    root_lr = os.path.join(args.data_path_low_res,  split, 'low_res')
    root_hr = os.path.join(args.data_path_high_res, split, 'high_res')

    ds_lr = RangeMapFolder(root_lr, transform=transforms.Compose(t_lr),
                           loader=npy_loader_range_only, class_dir=False)
    ds_hr = RangeMapFolder(root_hr, transform=transforms.Compose(t_hr),
                           loader=npy_loader_range_only, class_dir=False)

    assert len(ds_lr) == len(ds_hr), \
        f"KITTI LR({len(ds_lr)}) vs HR({len(ds_hr)}) file count mismatch"

    return PairDataset(ds_lr, ds_hr)


@register_dataset('kitti_object_paired')
def build_kitti_object_paired_dataset(is_train, args):
    """KITTI-object paired (same 2-ch format as kitti_paired; split=train/test)."""
    split = 'train' if is_train else 'test'
    t_lr = [transforms.ToTensor(), ScaleTensor(1/80)]
    t_hr = [transforms.ToTensor(), ScaleTensor(1/80)]
    if args.log_transform:
        t_lr.append(LogTransform())
        t_hr.append(LogTransform())
    root_lr = os.path.join(args.data_path_low_res,  split, 'low_res')
    root_hr = os.path.join(args.data_path_high_res, split, 'high_res')
    ds_lr = RangeMapFolder(root_lr, transform=transforms.Compose(t_lr),
                           loader=npy_loader_range_only, class_dir=False)
    ds_hr = RangeMapFolder(root_hr, transform=transforms.Compose(t_hr),
                           loader=npy_loader_range_only, class_dir=False)
    assert len(ds_lr) == len(ds_hr), \
        f"KITTI-object LR({len(ds_lr)}) vs HR({len(ds_hr)}) file count mismatch"
    return PairDataset(ds_lr, ds_hr)


@register_dataset('carla_paired')
def build_carla_paired_dataset(is_train, args):
    """
    CARLA paired: 2D arrays (H, W), no channel axis.
    Paths: {data_path}/{train|test}/high_res/*.npy
                                    /low_res/*.npy
    """
    split = 'train' if is_train else 'val'

    t_lr = [transforms.ToTensor(), ScaleTensor(1/80),
            FilterInvalidPixels(min_range=2/80, max_range=1)]
    t_hr = [transforms.ToTensor(), ScaleTensor(1/80),
            FilterInvalidPixels(min_range=2/80, max_range=1)]

    if args.log_transform:
        t_lr.append(LogTransform())
        t_hr.append(LogTransform())

    root_lr = os.path.join(args.data_path_low_res,  split, 'low_res')
    root_hr = os.path.join(args.data_path_high_res, split, 'high_res')

    ds_lr = RangeMapFolder(root_lr, transform=transforms.Compose(t_lr),
                           loader=npy_loader_2d, class_dir=False)
    ds_hr = RangeMapFolder(root_hr, transform=transforms.Compose(t_hr),
                           loader=npy_loader_2d, class_dir=False)

    assert len(ds_lr) == len(ds_hr), \
        f"CARLA LR({len(ds_lr)}) vs HR({len(ds_hr)}) file count mismatch"

    return PairDataset(ds_lr, ds_hr)


@register_dataset('durlar_paired')
def build_durlar_paired_dataset(is_train, args):
    """
    DurLAR paired:
      HR: (128, 2048, 2) -> range channel only (npy_loader_range_only)
      LR: (32,  2048)    -> already single channel (npy_loader_range_only also works)
    Paths: {data_path}/{train|test}/high_res/*.npy
                                    /low_res/*.npy
    """
    split = 'train' if is_train else 'val'

    t_lr = [transforms.ToTensor(), ScaleTensor(1/120),
            FilterInvalidPixels(min_range=0.3/120, max_range=1)]
    t_hr = [transforms.ToTensor(), ScaleTensor(1/120),
            FilterInvalidPixels(min_range=0.3/120, max_range=1)]

    if args.log_transform:
        t_lr.append(LogTransform())
        t_hr.append(LogTransform())

    root_lr = os.path.join(args.data_path_low_res,  split, 'low_res')
    root_hr = os.path.join(args.data_path_high_res, split, 'high_res')

    ds_lr = RangeMapFolder(root_lr, transform=transforms.Compose(t_lr),
                           loader=npy_loader_2d, class_dir=False)
    ds_hr = RangeMapFolder(root_hr, transform=transforms.Compose(t_hr),
                           loader=npy_loader_2d, class_dir=False)

    assert len(ds_lr) == len(ds_hr), \
        f"DurLAR LR({len(ds_lr)}) vs HR({len(ds_hr)}) file count mismatch"

    return PairDataset(ds_lr, ds_hr)