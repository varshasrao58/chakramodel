import argparse
import sys
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from train_pranet import KvasirSEGDataset, DiceFocalLoss
from metrics.seg_metrics import dice, iou
from transformer_segmenter import ChakraTransformerSegmenter

def main():
    parser = argparse.ArgumentParser(description="Train ChakraTransformer (Research Track)")
    parser.add_argument('--batch-size', type=int, default=2, help='Per-device batch size')
    parser.add_argument('--grad-accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (AdamW)')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--backbone', default='vit_large_patch16_384', help='timm ViT backbone')
    parser.add_argument('--resume', type=Path, help='Checkpoint produced by a previous run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-amp', action='store_false', dest='amp',
                        help='Disable CUDA automatic mixed precision')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("=== ChakraTransformer Training (High-Accuracy Research Track) ===")
    print(f"Batch Size: {args.batch_size}, Accumulation: {args.grad_accum}, "
          f"Epochs: {args.epochs}, LR: {args.lr}, AMP: {args.amp}")
    print(f"Initializing {args.backbone}...")

    assert torch.cuda.is_available(), "CUDA must be available to train! CPU training is disabled."
    device = torch.device("cuda")
    model = ChakraTransformerSegmenter(backbone_name=args.backbone).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = DiceFocalLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print(f"Model initialized on {device}. Loading dataset...")
    
    root = REPO_ROOT
    images_dir = root / "data" / "kvasir-seg" / "images"
    masks_dir  = root / "data" / "kvasir-seg" / "masks"
    
    if not images_dir.exists():
        print("[ERROR] Kvasir-SEG not found. Run: python src/download_kvasir.py")
        sys.exit(1)
        
    img_size = 384  # Upgraded resolution for ViT-Large
    
    full_dataset = KvasirSEGDataset(images_dir, masks_dir, img_size=img_size, augment=True)
    val_dataset  = KvasirSEGDataset(images_dir, masks_dir, img_size=img_size, augment=False)

    all_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif"}])
    n_train   = int(0.8 * len(all_paths))

    train_set = torch.utils.data.Subset(full_dataset, range(n_train))
    val_set   = torch.utils.data.Subset(val_dataset,  range(n_train, len(full_dataset)))

    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, **loader_options
    )

    weights_dir = root / "weights"
    weights_dir.mkdir(exist_ok=True)
    best_dice = 0.0
    best_path = weights_dir / "chakra_transformer_best.pth"
    latest_path = weights_dir / "chakra_transformer_latest.pth"
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_dice = checkpoint["best_dice"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {checkpoint['epoch']} with best Dice {best_dice:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(imgs)
                loss = criterion(logits, masks) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += loss.item() * args.grad_accum
        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_dice_scores = []
        val_iou_scores  = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(imgs)
                probs  = torch.sigmoid(logits).cpu().numpy()
                masks_np = masks.numpy()
                for i in range(len(probs)):
                    pred = (probs[i, 0] > 0.5).astype(np.uint8) * 255
                    gt   = (masks_np[i, 0] > 0.5).astype(np.uint8) * 255
                    val_dice_scores.append(dice(pred, gt))
                    val_iou_scores.append(iou(pred, gt))

        mean_dice = float(np.mean(val_dice_scores))
        mean_iou  = float(np.mean(val_iou_scores))
        lr_now    = optimizer.param_groups[0]["lr"]

        print(f"  Epoch [{epoch:3d}/{args.epochs}] | Loss: {train_loss:.4f} | Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f} | LR: {lr_now:.6f}", flush=True)

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), best_path)
            print(f"  *** New best Dice: {best_dice:.4f} -> saved to {best_path}")

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_dice': best_dice,
        }, latest_path)

    print(f"\nTraining Complete! Best Val Dice: {best_dice:.4f}. Weights saved: {best_path}")

if __name__ == '__main__':
    main()
