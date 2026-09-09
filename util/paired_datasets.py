# flash_x_datasets.py
#
# 기존 TULIP datasets.py에 추가할 paired dataset builders.
# 원본 kitti/durlar/carla builder는 전혀 수정하지 않음.
#
# 데이터 구조 (확인된 사실):
#   KITTI   high_res: (64,  1024, 2) float32  range 0~80m  + intensity  ← [...,0]만 사용
#           low_res:  (16,  1024, 2) float32  range 0~80m  + intensity  ← [...,0]만 사용
#
#   CARLA   high_res: (128, 2048)   float32  range 0~100m              ← 2D, 채널 없음
#           low_res:  (32,  2048)   float32  range 0~100m
#
#   DurLAR  high_res: (128, 2048, 2) float32 range 0~95m  + intensity  ← [...,0]만 사용
#           low_res:  (32,  2048)   float32  range 0~92m               ← 1채널
#
# 전처리 파이프라인:
#   KITTI/CARLA: ScaleTensor(1/80)
#   DurLAR:      ScaleTensor(1/120)  ← 센서 스펙 120m
#   모두:        log_transform 선택적, DownsampleTensor 없음 (LR 파일 직접 로드)

import os
import numpy as np
import torch
import torchvision.transforms as transforms

# TULIP 원본 코드의 클래스/함수를 그대로 재사용
from util.datasets import (
    register_dataset,
    RangeMapFolder,
    PairDataset,
    ScaleTensor,
    FilterInvalidPixels,
    LogTransform,
)


# ─────────────────────────────────────────────
# Loader 함수들
# ─────────────────────────────────────────────

def npy_loader_range_only(path: str) -> np.ndarray:
    """
    2채널 npy (range + intensity) 또는 1채널 npy를 모두 처리.
    항상 range 채널(index 0)만 반환.
    shape: (H, W, 2) → (H, W) / (H, W) → (H, W)
    """
    data = np.load(path).astype(np.float32)
    if data.ndim == 3:          # (H, W, 2)
        return data[..., 0]     # range 채널만
    return data                 # (H, W) 이미 1채널


def npy_loader_2d(path: str) -> np.ndarray:
    """
    CARLA용: (H, W) 2D array 그대로 반환.
    TULIP 원본 npy_loader는 [..., 0]을 하므로
    2D array에 적용하면 crash → 별도 loader 필요.
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
    KITTI paired: LR/HR 파일 직접 로드 (on-the-fly downsample 없음)
    경로: {data_path}/{train|test}/high_res/*.npy
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
        f"KITTI LR({len(ds_lr)}) vs HR({len(ds_hr)}) 파일 수 불일치"

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
        f"KITTI-object LR({len(ds_lr)}) vs HR({len(ds_hr)}) 파일 수 불일치"
    return PairDataset(ds_lr, ds_hr)


@register_dataset('carla_paired')
def build_carla_paired_dataset(is_train, args):
    """
    CARLA paired: 2D array (H, W), 채널 없음
    경로: {data_path}/{train|test}/high_res/*.npy
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
        f"CARLA LR({len(ds_lr)}) vs HR({len(ds_hr)}) 파일 수 불일치"

    return PairDataset(ds_lr, ds_hr)


@register_dataset('durlar_paired')
def build_durlar_paired_dataset(is_train, args):
    """
    DurLAR paired:
      HR: (128, 2048, 2) → range 채널만 추출 (npy_loader_range_only)
      LR: (32,  2048)    → 이미 1채널 (npy_loader_range_only도 동작)
    경로: {data_path}/{train|test}/high_res/*.npy
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
        f"DurLAR LR({len(ds_lr)}) vs HR({len(ds_hr)}) 파일 수 불일치"

    return PairDataset(ds_lr, ds_hr)