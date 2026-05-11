"""
Maritime Semantic Segmentation — Training Script v3
====================================================
Architecture : Model architecture and encoder backbone are fully configurable through smp and timm.

Models: (change in the model section (4))

smp.UnetPlusPlus
smp.DeepLabV3Plus
smp.PSPNet
smp.FPN
smp.Linknet

Encoders: (from command line):

resnet34
timm-resnest50d
timm-efficientnet-b0
timm-efficientnet-b2
timm-mobilenetv3_large_100

Dataset      : 1693 images, pre-split into train / validate / test folders
Classes: Obstacle/Env (ignored in metrics), Water, Sky, Ship, Unknown (ignored in metrics), Land, Buoy

Tracked metrics: IoU per class, mIoU, pixelAccuracy, Precision/Recall for ships & buoys, inference time, peak VRAM
parameter count, checkpoint file size (inference is with baytch size 1)

Example:
  # UNet++ + ResNet34
  python train.py --data_root ~/maritime/dataset \
                  --output_dir ~/maritime/output/unetplusplus_resnet34 \
                  --encoder resnet34 \
                  --epochs 100 --batch_size 8
"""

import time
import argparse
import random
import json
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import LovaszLoss, FocalLoss #this is the final version, Dice Loss was used before wasnt optimal

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchmetrics.classification import (
    MulticlassJaccardIndex, #iou
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)


# ──────────────────────────────────────────────
# 0.  Korratavus (Reproducibility)
# ──────────────────────────────────────────────
def set_seed(seed: int = 42): #each time same randomly picked initial values
    random.seed(seed) #python random module
    np.random.seed(seed) #numpy random module
    torch.manual_seed(seed) #pytorch random module
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #no automatic algorithm


# ──────────────────────────────────────────────
# 1.  Seadistus (Config)
# ──────────────────────────────────────────────
NUM_CLASSES    = 8         #class 5 is obsolete and not used
IGNORE_CLASSES = [0, 4]      # Obstacle/Env and Unknown
LOSS_IGNORE    = 255          # pixels ignored have 255 as value and are ignored in loss computation

CLASS_NAMES = {
    0: "Obstacle/Env",
    1: "Water",
    2: "Sky",
    3: "Ship",
    4: "Unknown",
    6: "Land",
    7: "Buoy",
}

VALID_CLASSES = [1, 2, 3, 6, 7]
F1_CLASSES = [3, 7]   # Ship, Buoy F1 will be tracked
PIXEL_COUNTS = torch.tensor([
    1_459_111,    # 0  Obstacle/Env
    139_322_138,  # 1  Water
    154_307_671,  # 2  Sky
    4_802_620,    # 3  Ship
    5_422_140,    # 4  Unknown
    1,            # 5  unused
    26_075_524,   # 6  Land
    1_468_140,    # 7  Buoy
], dtype=torch.float32)

RARE_CLASSES = [3, 7]   # rare classes which will be boosted later


# ──────────────────────────────────────────────
# 2.  Andmestik (Dataset)
# ──────────────────────────────────────────────
class MaritimeDataset(Dataset):
    """
    !!!Expects pre-split folder structure!!!
        data_root/
            train/      images/  masks/
            validate/   images/  masks/
            test/       images/  masks/
    !!!Mask pixel values: 0,1,2,3,4,6,7!!!
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        assert len(image_paths) == len(mask_paths), \ #if pics and masks align
            f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self): #number of pics
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask  = np.array(Image.open(self.mask_paths[idx]))
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]
        return image, mask.long()


def load_split(data_root: Path, split: str):
    """Load image/mask path lists for train | validate | test."""
    img_dir  = data_root / split / "images"
    mask_dir = data_root / split / "masks"

    assert img_dir.exists(),  f"Missing: {img_dir}"
    assert mask_dir.exists(), f"Missing: {mask_dir}"

    stems = sorted([
        p.stem for p in img_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    assert len(stems) > 0, f"No images in {img_dir}"

    imgs, masks = [], []
    for s in stems:
        for ext in (".png", ".jpg", ".jpeg"): # must be png/jpg/jpeg
            ip = img_dir / (s + ext)
            if ip.exists():
                imgs.append(ip)
                break
        mp = mask_dir / (s + ".png") # mask can only be ong
        assert mp.exists(), f"Mask missing for: {s}"
        masks.append(mp)

    print(f"  {split:10s}: {len(imgs)} images") #how many imagepairs found
    return imgs, masks


def get_transforms(split: str):
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4), #to combat different light
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3), # maybe turn higher due to night issues
            A.GaussianBlur(blur_limit=(3, 5), p=0.15), #focus of camera
            A.GaussNoise(std_range=(0.01, 0.05), p=0.15), #sensor noise
            A.RandomFog(fog_coef_range=(0.05, 0.25), p=0.1),
            A.RandomRain(slant_range=(-5, 5), drop_length=10, p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=8, border_mode=cv2.BORDER_REFLECT, p=0.4),#camera angle, made to imitate the movemenet of waves a bit
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def compute_sample_weights(mask_paths, rare_classes=RARE_CLASSES, # if picture has rare class then 20x higher probability to be picked for training
                           boost=20.0): # 6.0 -> 12.0 -> 20.0 so far in testing (25.0 ei funka enam)
    weights = []
    for p in mask_paths:
        mask     = np.array(Image.open(p))
        has_rare = any((mask == c).any() for c in rare_classes)
        weights.append(boost if has_rare else 1.0)
    return weights


# ──────────────────────────────────────────────
# 3.  Kadu (Loss)
# ──────────────────────────────────────────────
def build_loss(device):
    """
    Combined loss (katse 4): 0.4 Focal + 0.4 Lovász + 0.2 CrossEntropy.
    """
    weights = 1.0 / PIXEL_COUNTS # total weights
    weights = weights / weights[2]   # sky = 1x
    weights[0] = 0.0                 # ignore classes
    weights[4] = 0.0
    weights[5] = 0.0
    weights = weights.clamp(max=200.0).to(device) #current limit to avoid destabilise (250.0->200.0)

    focal_loss = FocalLoss( #focusing on harder pixels, gamma 2.0->3.0->4.0
        mode="multiclass",
        gamma=4.0,
        ignore_index=LOSS_IGNORE,
        normalized=True,
    )
    lovasz_loss = LovaszLoss( #focusing on IoU gain, especially boosting smaller classes
        mode="multiclass",
        ignore_index=LOSS_IGNORE,
    )
    ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=LOSS_IGNORE) #stabiliser for lovasz + focal

    def combined(pred, target):
        t = target.clone()
        for c in IGNORE_CLASSES:
            t[t == c] = LOSS_IGNORE #ignored classes to 255 to not change loss
        return (
            0.4 * focal_loss(pred, t) + 0.4 * lovasz_loss(pred, t) + 0.2 * ce_loss(pred, t)) #prev 0.5 focal + 0.3 dice + 0.2 CE

    return combined

# ──────────────────────────────────────────────
# 4.  Mudel (Model)
# ──────────────────────────────────────────────
def build_model(encoder: str = "resnet34", pretrained: bool = True): #encoder is default
    model = smp.UnetPlusPlus( #look intro for others, some characteristics might depend on the chosen model
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model


# ──────────────────────────────────────────────
# 5.  Mõõdikud (Metrics)
# ──────────────────────────────────────────────
class SegMetrics:
    def __init__(self, device):
        self.iou = MulticlassJaccardIndex(
            num_classes=NUM_CLASSES,
            average=None, #ious differently at start
        ).to(device)
        self.acc = MulticlassAccuracy(
            num_classes=NUM_CLASSES,
            average="micro", #average pixel acc
        ).to(device)
        self.precision = MulticlassPrecision( #currently all, in the end only prec and recall.
            num_classes=NUM_CLASSES,
            average=None,
        ).to(device)
        self.recall = MulticlassRecall(
            num_classes=NUM_CLASSES,
            average=None,
        ).to(device)
        self.device = device

    def update(self, preds, targets):
        pred_labels = preds.argmax(1) #score for each pixel, argmax top score
        valid = torch.ones_like(targets, dtype=torch.bool)
        for c in IGNORE_CLASSES:
            valid &= (targets != c) #filter out ignored pixels
        valid &= (targets != 5)

        pred_labels  = pred_labels[valid]
        targets_filt = targets[valid]

        if targets_filt.numel() == 0:
            return

        self.iou.update(pred_labels, targets_filt) #update metrics based on prediction
        self.acc.update(pred_labels, targets_filt)
        self.precision.update(pred_labels, targets_filt)
        self.recall.update(pred_labels, targets_filt)

    def compute(self):
        per_class_iou = self.iou.compute().cpu().numpy()
        pixel_acc     = self.acc.compute().item()
        per_class_p   = self.precision.compute().cpu().numpy()
        per_class_r   = self.recall.compute().cpu().numpy()

        valid_iou = [per_class_iou[c] for c in VALID_CLASSES]
        mean_iou  = float(np.mean(valid_iou)) #miou

        # P - precision , R - recall
        f1_stats = {}
        for c in F1_CLASSES:
            p = float(per_class_p[c])
            r = float(per_class_r[c])
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0 #ship/buoy F1 calculate
            f1_stats[c] = {"precision": p, "recall": r, "f1": f1}

        return mean_iou, per_class_iou, pixel_acc, f1_stats #return accuracy metrics

    def reset(self): # clear last epoch
        self.iou.reset()
        self.acc.reset()
        self.precision.reset()
        self.recall.reset()

# ──────────────────────────────────────────────
# 6.  Ressursimõõdikud (Resource metrics) #BATCH SIZE 1!!!!!
# ──────────────────────────────────────────────
def measure_inference_time(model, device, h=384, w=512, warmup=10, runs=100):
    model.eval()
    dummy = torch.randn(1, 3, h, w).to(device) #random pilt for test
    with torch.no_grad():
        for _ in range(warmup): #warmup since at first kernel compile + memory alloc takes time
            model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(dummy)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / runs * 1000 #average for 100 runs
    return ms, 1000.0 / ms


def measure_vram(model, device, h=384, w=512): #during inference not training!
    model.eval()
    dummy = torch.randn(1, 3, h, w).to(device)
    torch.cuda.reset_peak_memory_stats(device) #resetprevioys
    with torch.no_grad():
        model(dummy) # one inference
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1e6


# ──────────────────────────────────────────────
# 7.  Treenimine/Valideerimine (Train/Validate)
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"): #makes it faster for and less needy fo VRAM
            preds = model(images)
            loss  = loss_fn(preds, masks)

        scaler.scale(loss).backward() #calculate gradient
        scaler.unscale_(optimizer) #back to regular scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #clip if explosion
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad() #make faster
def validate(model, loader, loss_fn, metrics, device): #essentialy check emtrics
    model.eval()
    metrics.reset() #reset last statistics
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with autocast("cuda"):
            preds = model(images)
            loss  = loss_fn(preds, masks)
        total_loss += loss.item()
        metrics.update(preds, masks)

    mean_iou, per_class_iou, pixel_acc, f1_stats = metrics.compute()
    return total_loss / len(loader), mean_iou, per_class_iou, pixel_acc, f1_stats


# ──────────────────────────────────────────────
# 8.  Vahemudelite salvestus (Checkpoint helpers)
# ──────────────────────────────────────────────
def save_checkpoint(state, path): #training can be restarted as well
    torch.save(state, path) #salvestab mis mainis state all toodud
    print(f"  Checkpoint saved -> {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


# ──────────────────────────────────────────────
# 9.  Logija (Logger) #maybe doesnt serve the intended purpose as the logs were not used at all during the work
# ──────────────────────────────────────────────
class Logger: #essentially logging all information throughout epochs
    def __init__(self, path):
        self.path    = path
        self.history = {
            "train_loss":    [],
            "val_loss":      [],
            "miou":          [],
            "pixel_acc":     [],
            "per_class_iou": [],
            # Ship and Buoy precision, recall, F1
            "precision_ship": [],
            "recall_ship":    [],
            "f1_ship":        [],
            "precision_buoy": [],
            "recall_buoy":    [],
            "f1_buoy":        [],
        }

    def log(self, train_loss, val_loss, miou, pixel_acc,
            per_class_iou, f1_stats):
        self.history["train_loss"].append(round(train_loss, 5))
        self.history["val_loss"].append(round(val_loss, 5))
        self.history["miou"].append(round(miou, 5))
        self.history["pixel_acc"].append(round(pixel_acc, 5))
        self.history["per_class_iou"].append(
            [round(float(v), 5) for v in per_class_iou])

        for cls_idx, key in [(3, "ship"), (7, "buoy")]: #logib key abil mõlema klassi prec/recall/f1
            stats = f1_stats[cls_idx]
            self.history[f"precision_{key}"].append(round(stats["precision"], 5))
            self.history[f"recall_{key}"].append(round(stats["recall"], 5))
            self.history[f"f1_{key}"].append(round(stats["f1"], 5))

        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)


# ──────────────────────────────────────────────
# 10.  Main
# ──────────────────────────────────────────────
def parse_args(): #käsurea initsialiseerimine
    p = argparse.ArgumentParser(description="Maritime segmentation")
    p.add_argument("--data_root",   type=str, required=True,
                   help="Root with train/ validate/ test/ subfolders") #dataset folder
    p.add_argument("--output_dir",  type=str, default="./output") #creates a folder
    p.add_argument("--encoder", type=str, default="resnet34",
           help="Encoder backbone from SMP") #mostly have to use timm
    p.add_argument("--epochs",      type=int,   default=100) #epoch count
    p.add_argument("--batch_size",  type=int,   default=8) #batch size, default 8
    p.add_argument("--lr",          type=float, default=1e-4) #learning rate
    p.add_argument("--num_workers", type=int,   default=4) #optional
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = output_dir / "best_model.pth"
    last_ckpt  = output_dir / "last_model.pth"
    log_path   = output_dir / "history.json"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #gives system characteristics
    print(f"\n{'='*60}")
    print(f"  Architecture : UNet++ / {args.encoder}")
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM total   : {vram_total:.1f} GB")
    print(f"{'='*60}\n")

    # Data
    print("Loading dataset splits...")
    tr_imgs, tr_masks = load_split(data_root, "train")
    vl_imgs, vl_masks = load_split(data_root, "validate")
    te_imgs, te_masks = load_split(data_root, "test")

    train_ds = MaritimeDataset(tr_imgs, tr_masks, get_transforms("train"))
    val_ds   = MaritimeDataset(vl_imgs, vl_masks, get_transforms("val"))
    test_ds  = MaritimeDataset(te_imgs, te_masks, get_transforms("val"))

    sample_w = compute_sample_weights(tr_masks)
    sampler  = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    print(f"\nBuilding the model with encoder: {args.encoder}")
    model        = build_model(args.encoder).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"  Total parameters     : {total_params/1e6:.2f} M")
    print(f"  Trainable parameters : {train_params/1e6:.2f} M")

    # Efficiency
    print("\nMeasuring deployment efficiency (batch size = 1)...")
    ms_pre, fps_pre = measure_inference_time(model, device)
    vram_pre        = measure_vram(model, device)
    print(f"  Inference time : {ms_pre:.1f} ms/frame  ({fps_pre:.1f} FPS)")
    print(f"  Peak VRAM      : {vram_pre:.0f} MB  [inference only]")

    # Loss / Optim / Scheduler
    loss_fn   = build_loss(device)
    optimizer = torch.optim.AdamW( #adamw only used
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6) #drop on a cosinus way from learning rate to 1e-6
    scaler  = GradScaler("cuda", enabled=not args.no_amp)
    metrics = SegMetrics(device)
    logger  = Logger(log_path)

    # Resume
    start_epoch = 0
    best_miou   = 0.0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, best_miou = load_checkpoint(
            args.resume, model, optimizer, scheduler)
        start_epoch += 1

    # Training loop (treeni->valideeri->scheduler.step()->prindi->salvesta)
    print(f"\nStarting training for {args.epochs} epochs...\n")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device)
        val_loss, miou, per_class_iou, pixel_acc, f1_stats = validate(
            model, val_loader, loss_fn, metrics, device)
        scheduler.step() # learning rate changer

        elapsed = time.time() - t0 # how long for train + val
        lr_now  = scheduler.get_last_lr()[0] #get last learning rate
        is_best = miou > best_miou #flag et kas on parim mudel

        print(f"Epoch [{epoch+1:03d}/{args.epochs}]  " #every epoch 
              f"t={elapsed:.0f}s  lr={lr_now:.2e}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"mIoU={miou:.4f}  "
              f"px_acc={pixel_acc:.4f}"
              f"{'  *' if is_best else ''}")

        print("  Per-class IoU:") #each class iou
        for c in VALID_CLASSES:
            bar = "█" * int(per_class_iou[c] * 20)
            print(f"    {CLASS_NAMES[c]:15s} (idx {c}): "
                  f"{per_class_iou[c]:.3f}  {bar}")

        print("  Ship / Buoy  Precision | Recall | F1:") #rare class extra metricsD
        for c in F1_CLASSES:
            s = f1_stats[c]
            print(f"    {CLASS_NAMES[c]:15s} (idx {c}): "
                  f"P={s['precision']:.3f}  R={s['recall']:.3f}  "
                  f"F1={s['f1']:.3f}")
        print()

        state = { #previously mentioned state which is saved to keep the checkpoints
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "args":      vars(args),
        }
        if is_best:
            best_miou = miou
            save_checkpoint(state, best_ckpt) #best model saving
        save_checkpoint(state, last_ckpt) #saves last as well
        logger.log(train_loss, val_loss, miou, pixel_acc,
                   per_class_iou, f1_stats)

    # ── Final test evaluation ───────────────────
    print("=" * 60)
    print("Running final evaluation on test set...") #test set unchanged throughout
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True)

    load_checkpoint(best_ckpt, model) #best model against test set
    model.to(device)

    _, test_miou, test_per_class, test_px_acc, test_f1 = validate(
        model, test_loader, loss_fn, metrics, device)

    ms, fps  = measure_inference_time(model, device) #best model inference time
    vram_mb  = measure_vram(model, device)
    ckpt_mb  = best_ckpt.stat().st_size / 1e6

    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS")
    print(f"  Architecture : UNet++ (change manually) / {args.encoder}") #this has to be manually changed on new models
    print(f"{'='*60}")
    print(f"  mIoU           : {test_miou:.4f}")
    print(f"  Pixel accuracy : {test_px_acc:.4f}")
    print(f"\n  Per-class IoU:")
    for c in VALID_CLASSES:
        print(f"    {CLASS_NAMES[c]:15s} (idx {c}): "
              f"{test_per_class[c]:.4f}")
    print(f"\n  Ship / Buoy  Precision | Recall | F1:")
    for c in F1_CLASSES:
        s = test_f1[c]
        print(f"    {CLASS_NAMES[c]:15s} (idx {c}): "
              f"P={s['precision']:.4f}  R={s['recall']:.4f}  "
              f"F1={s['f1']:.4f}")
    print(f"\n  Deployment efficiency (batch size = 1):")
    print(f"    Parameters     : {total_params/1e6:.2f} M")
    print(f"    Inference time (batch size = 1): {ms:.1f} ms/frame  ({fps:.1f} FPS)")
    print(f"    Peak VRAM      : {vram_mb:.0f} MB")
    print(f"    Checkpoint size: {ckpt_mb:.1f} MB")
    print(f"{'='*60}\n")

    results = {
        "architecture":  "UnetPlusPlus (change manually if changed in 4.)",
        "encoder":       args.encoder,
        "test_miou":     round(test_miou, 5),
        "pixel_acc":     round(test_px_acc, 5),
        "per_class_iou": {
            CLASS_NAMES[c]: round(float(test_per_class[c]), 5)
            for c in VALID_CLASSES
        },
        "f1_stats": {
            CLASS_NAMES[c]: {
                "precision": round(test_f1[c]["precision"], 5),
                "recall":    round(test_f1[c]["recall"],    5),
                "f1":        round(test_f1[c]["f1"],        5),
            }
            for c in F1_CLASSES
        },
        "efficiency": {
            "parameters_M":  round(total_params / 1e6, 2),
            "inference_ms":  round(ms, 2),
            "fps":           round(fps, 1),
            "peak_vram_mb":  round(vram_mb, 0),
            "checkpoint_mb": round(ckpt_mb, 1),
            "note": "VRAM and FPS measured at batch size 1 (deployment conditions)"
        }
    }
    results_path = output_dir / "test_results.json" #final test logs
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {results_path}")


if __name__ == "__main__":
    main()