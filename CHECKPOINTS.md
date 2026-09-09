# Pretrained Checkpoints

| Model | Dataset | Upsampling | Download |
|-------|---------|-----------|----------|
| FLASH | KITTI | 16 → 64 | **[link — TODO]()** |

The released KITTI checkpoint reaches **voxel IoU ≈ 0.40 @ 0.1 m** on the KITTI test split.

> Replace the link above with your hosting URL (Google Drive / Hugging Face / GitHub release asset).

### Inference
```bash
python inference.py --checkpoint flash_kitti.pth \
    --input sample_low_res.npy --dataset kitti --out_dir out --save_ply
```
Loads directly into the FLASH model built by `inference.py` / `train.py` (dual-domain: Frequency-Aware attention + Multi-Scale Fusion).
