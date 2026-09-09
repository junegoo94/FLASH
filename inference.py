#!/usr/bin/env python3
"""FLASH inference: super-resolve a low-resolution LiDAR range image and (optionally) back-project
to a point cloud.

Example:
    python inference.py --checkpoint checkpoints/flash_kitti.pth \
        --input sample_low_res.npy --dataset kitti --out_dir out --save_ply
"""
import os, argparse, numpy as np, torch
from model.flash_improved_abl import flash_improved_base
from util.evaluation import img_to_pcd_kitti, img_to_pcd_carla, img_to_pcd_durlar

CFG = {   # per-sensor: input/target size, max range (m), back-projector
    "kitti":  dict(isz=(16, 1024), tsz=(64, 1024),  scale=80.0,  bp=img_to_pcd_kitti),
    "carla":  dict(isz=(32, 2048), tsz=(128, 2048), scale=80.0,  bp=img_to_pcd_carla),
    "durlar": dict(isz=(32, 2048), tsz=(128, 2048), scale=120.0, bp=img_to_pcd_durlar),
}

def build_flash(isz, tsz, device):
    # FLASH = the dual-domain model with Frequency-Aware attention + Multi-Scale Fusion.
    m = flash_improved_base(
        img_size=isz, target_img_size=tsz, patch_size=(1, 4), window_size=(2, 8), in_chans=1,
        circular_padding=True, pixel_shuffle=True, log_transform=True,
        use_fix1=False, use_fix2=False, use_fix3=False)
    return m.to(device).eval()

def load_checkpoint(model, path):
    ck = torch.load(path, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {path}  (missing {len(missing)}, unexpected {len(unexpected)})")
    return model

def write_ply(path, pts):
    pts = np.asarray(pts, np.float32)[:, :3]; n = len(pts)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\nend_header\n").encode())
        f.write(pts.astype("<f4").tobytes())

@torch.no_grad()
def super_resolve(model, lr_norm, tsz, device):
    """lr_norm: (H, W) range normalised to [0,1]. Returns high-res range in [0,1]."""
    x = torch.log1p(torch.from_numpy(lr_norm)[None, None].to(device))
    target = torch.zeros(1, 1, *tsz, device=device)
    with torch.cuda.amp.autocast(enabled=(device != "cpu")):
        pred, _, _ = model(x, target, eval=True)
    pred = torch.expm1(pred)
    pred = torch.where((pred >= 2 / 80) & (pred <= 1), pred, torch.zeros_like(pred))  # valid-range filter
    return pred.squeeze().float().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="low-res range image .npy (H x W, metres)")
    ap.add_argument("--dataset", default="kitti", choices=list(CFG))
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--save_ply", action="store_true", help="also save the back-projected point cloud")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    c = CFG[args.dataset]; S = c["scale"]
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    lr = np.load(args.input).astype(np.float32); lr = lr[..., 0] if lr.ndim == 3 else lr
    model = load_checkpoint(build_flash(c["isz"], c["tsz"], device), args.checkpoint)

    hr = super_resolve(model, lr / S, c["tsz"], device)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    np.save(f"{args.out_dir}/{stem}_sr.npy", (hr * S).astype(np.float32))
    print(f"saved {args.out_dir}/{stem}_sr.npy  ({hr.shape[0]}x{hr.shape[1]})")

    if args.save_ply:
        pts = np.asarray(c["bp"](hr, maximum_range=S))[:, :3]
        d = np.linalg.norm(pts, axis=1); pts = pts[(d > 1) & (d < S)]
        write_ply(f"{args.out_dir}/{stem}_sr.ply", pts)
        print(f"saved {args.out_dir}/{stem}_sr.ply  ({len(pts)} points)")

if __name__ == "__main__":
    main()
