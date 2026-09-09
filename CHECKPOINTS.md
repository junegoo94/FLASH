# Pretrained Checkpoints

| Model | Dataset | Upsampling | Download |
|-------|---------|-----------|----------|
| FLASH | KITTI | 16 → 64 | [Hugging Face](https://huggingface.co/jmgoo1118/FLASH/resolve/main/flash_kitti.pth) |

The released KITTI checkpoint reaches **voxel IoU ≈ 0.41 @ 0.1 m** on the KITTI test split.

### Download
```bash
wget https://huggingface.co/jmgoo1118/FLASH/resolve/main/flash_kitti.pth
```
or in Python:
```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("jmgoo1118/FLASH", "flash_kitti.pth")
```

### Inference
```bash
python inference.py --checkpoint flash_kitti.pth \
    --input sample_low_res.npy --dataset kitti --out_dir out --save_ply
```
Loads directly into the FLASH model built by `inference.py` / `train.py` (dual-domain: Frequency-Aware attention + Multi-Scale Fusion).
