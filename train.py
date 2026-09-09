#!/usr/bin/env python3
"""Train FLASH (dual-domain: Frequency-Aware attention + Multi-Scale Fusion) on paired range images.

Data layout (range images as .npy, in metres):
    data/kitti/{train,val}/{low_res,high_res}/*.npy   # low: 16x1024, high: 64x1024

Example:
    python train.py --data_root data/kitti --epochs 600 --batch_size 16 --output_dir checkpoints/
"""
import os, glob, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from model.flash_improved_abl import flash_improved_base

class PairedRange(Dataset):
    """Loads paired low-/high-res range .npy, scales to [0,1], log-compresses (log1p)."""
    def __init__(self, root, split, scale, min_r, max_r=1.0):
        self.lo = sorted(glob.glob(f"{root}/{split}/low_res/*.npy"))
        self.hi = sorted(glob.glob(f"{root}/{split}/high_res/*.npy"))
        assert len(self.lo) == len(self.hi) and len(self.lo) > 0, f"no/uneven data in {root}/{split}"
        self.s, self.mn, self.mx = scale, min_r, max_r
    def __len__(self): return len(self.lo)
    def _prep(self, p):
        a = np.load(p).astype(np.float32); a = a[..., 0] if a.ndim == 3 else a
        a = a * self.s                                   # to [0,1]
        a = np.where((a >= self.mn) & (a <= self.mx), a, 0.0)   # filter invalid
        return torch.log1p(torch.from_numpy(a))[None]     # (1,H,W), log-compressed
    def __getitem__(self, i): return self._prep(self.lo[i]), self._prep(self.hi[i])

def build_flash(isz, tsz):
    return flash_improved_base(
        img_size=isz, target_img_size=tsz, patch_size=(1, 4), window_size=(2, 8), in_chans=1,
        circular_padding=True, pixel_shuffle=True, log_transform=True,
        use_fix1=False, use_fix2=False, use_fix3=False)     # all off == FLASH

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/kitti")
    ap.add_argument("--dataset", default="kitti", choices=["kitti", "carla", "durlar"])
    ap.add_argument("--output_dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    SZ = {"kitti": ((16, 1024), (64, 1024), 1/80., 2/80.),
          "carla": ((32, 2048), (128, 2048), 1/80., 2/80.),
          "durlar": ((32, 2048), (128, 2048), 1/120., 0.3/120.)}[args.dataset]
    isz, tsz, scale, min_r = SZ
    dev = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    ds = PairedRange(args.data_root, "train", scale, min_r)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True)
    model = build_flash(isz, tsz).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev != "cpu"))
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
        start = ck.get("epoch", 0) + 1
        print(f"resumed from {args.resume} @ epoch {start}")
    print(f"FLASH | {args.dataset} {isz}->{tsz} | train {len(ds)} | {sum(p.numel() for p in model.parameters()):,} params")

    for ep in range(start, args.epochs):
        model.train(); tot = 0.0
        for low, high in dl:
            low, high = low.to(dev), high.to(dev)
            with torch.cuda.amp.autocast(enabled=(dev != "cpu")):
                _, loss, _ = model(low, high)             # FLASH computes the L1 log-range loss internally
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        print(f"epoch {ep:4d}  loss {tot/len(dl):.4f}")
        if (ep + 1) % args.save_every == 0 or ep == args.epochs - 1:
            torch.save({"model": model.state_dict(), "epoch": ep},
                       f"{args.output_dir}/flash_{args.dataset}_ep{ep}.pth")
            print(f"  saved {args.output_dir}/flash_{args.dataset}_ep{ep}.pth")

if __name__ == "__main__":
    main()
