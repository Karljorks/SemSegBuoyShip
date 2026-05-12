"""
predict.py — Run inference and save coloured overlay images

Usage:
    python predict.py \
        --checkpoint ~/maritime/output/unetpp_mobilenetv3/best_model.pth \
        --input_dir  ~/maritime/dataset/test/images \
        --output_dir ~/maritime/predictions_unetpp \
        --encoder    timm-mobilenetv3_large_100

    # With ground-truth masks (enables per-image mIoU display):
    python predict.py \
        --checkpoint ~/maritime/output/unetpp_mobilenetv3/best_model.pth \
        --input_dir  ~/maritime/dataset/test/images \
        --mask_dir   ~/maritime/dataset/test/masks \
        --output_dir ~/maritime/predictions_unetpp \
        --encoder    timm-mobilenetv3_large_100
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
NUM_CLASSES    = 8
IGNORE_CLASSES = [0, 4, 5]
VALID_CLASSES  = [1, 2, 3, 6, 7]

CLASS_NAMES = {
    0: "Obstacle/Env",
    1: "Water",
    2: "Sky",
    3: "Ship",
    4: "Unknown",
    6: "Land",
    7: "Buoy",
}

PALETTE = {
    0: (255, 140,   0),   # Obstacle/Env  — orange
    1: (  0, 120, 255),   # Water         — blue
    2: ( 30, 170, 255),   # Sky           — light blue
    3: (220,  50,  50),   # Ship          — red
    4: (120, 120, 120),   # Unknown       — grey
    5: (  0,   0,   0),   # Unused        — black
    6: ( 60, 180,  60),   # Land          — green
    7: (255, 230,   0),   # Buoy          — yellow
}

transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def mask_to_rgb(mask: np.ndarray) -> np.ndarray: #convert to RGB
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in PALETTE.items():
        rgb[mask == cls_id] = colour
    return rgb

def blend(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray: #blend image with mask 50% opacity
    return (image * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)

def compute_miou(pred: np.ndarray, gt: np.ndarray) -> float: #calculate mIoU for display
    ious = []
    for c in VALID_CLASSES:
        valid_mask = np.ones_like(gt, dtype=bool)
        for ig in IGNORE_CLASSES:
            valid_mask &= (gt != ig)

        pred_c = (pred == c) & valid_mask
        gt_c   = (gt   == c) & valid_mask

        intersection = (pred_c & gt_c).sum()
        union        = (pred_c | gt_c).sum()

        if union == 0:
            continue
        ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


def make_legend(width: int, cell_h: int = 28) -> Image.Image: #legend underneath the picture with the classes and colors
    classes = [(k, v) for k, v in CLASS_NAMES.items() if k not in [4, 5]]
    height  = cell_h * len(classes) + 10
    legend  = Image.new("RGB", (width, height), (30, 30, 30))
    draw    = ImageDraw.Draw(legend)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (cls_id, name) in enumerate(classes):
        y   = 5 + i * cell_h
        col = PALETTE[cls_id]
        draw.rectangle([8, y + 3, 28, y + cell_h - 3], fill=col)
        draw.text((36, y + 5), name, fill=(255, 255, 255), font=font)

    return legend


def make_side_by_side( #assemble final output, header shows miou and ilename, panel shows pictures, legend at bottom
    image_np:   np.ndarray,
    blended_np: np.ndarray,
    miou:       float | None,
    img_name:   str,
) -> Image.Image:
    h, w     = image_np.shape[:2]
    HEADER_H = 44
    GAP      = 6

    total_w = w * 2 + GAP
    header  = Image.new("RGB", (total_w, HEADER_H), (20, 20, 20))
    draw_h  = ImageDraw.Draw(header)

    try:
        font_lg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg

    draw_h.text((10, 8),  img_name,               fill=(220, 220, 220), font=font_lg)
    draw_h.text((10, 26), "Original",              fill=(160, 160, 160), font=font_sm)
    draw_h.text((w + GAP + 10, 26), "Prediction Overlay",
                fill=(160, 160, 160), font=font_sm)

    if miou is not None:
        miou_text = f"mIoU: {miou:.4f}"
        col       = (100, 220, 100) if miou >= 0.90 else \
                    (255, 200,  60) if miou >= 0.80 else (220, 80, 80)
        bbox = draw_h.textbbox((0, 0), miou_text, font=font_lg)
        tw   = bbox[2] - bbox[0]
        draw_h.text((total_w - tw - 12, 14), miou_text, fill=col, font=font_lg)

    orig_pil    = Image.fromarray(image_np)
    overlay_pil = Image.fromarray(blended_np)

    panel = Image.new("RGB", (total_w, h), (15, 15, 15))
    panel.paste(orig_pil,    (0,       0))
    panel.paste(overlay_pil, (w + GAP, 0))

    legend = make_legend(total_w)

    full_h   = HEADER_H + h + legend.height
    combined = Image.new("RGB", (total_w, full_h), (15, 15, 15))
    combined.paste(header, (0, 0))
    combined.paste(panel,  (0, HEADER_H))
    combined.paste(legend, (0, HEADER_H + h))

    return combined

def build_model(encoder: str) -> smp.UnetPlusPlus: #at the moment is U++, this can be changed the same way as the training code
    return smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )

def main():
    p = argparse.ArgumentParser(description="UNet++ Maritime Inference") #argparse values
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model.pth")
    p.add_argument("--input_dir",  required=True,
                   help="Directory of input images")
    p.add_argument("--output_dir", default="./predictions",
                   help="Where to save outputs")
    p.add_argument("--mask_dir",   default=None,
                   help="(Optional) Ground-truth mask directory — enables mIoU display")
    p.add_argument("--encoder",    default="timm-mobilenetv3_large_100",
                   help="Must match the encoder used during training")
    p.add_argument("--alpha",      type=float, default=0.5,
                   help="Overlay blend strength (0=original, 1=full mask)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading checkpoint: {args.checkpoint}") #loading the model, must be .pth file
    model = build_model(args.encoder)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    epoch    = ckpt.get("epoch", "?")
    best_iou = ckpt.get("best_miou", None)
    print(f"  Encoder  : {args.encoder}")
    print(f"  Epoch    : {epoch + 1 if isinstance(epoch, int) else epoch}")
    if best_iou is not None:
        print(f"  Best val mIoU: {best_iou:.4f}")
    print()

    img_dir = Path(args.input_dir)
    paths   = sorted(
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.jpeg"))
    )
    print(f"Found {len(paths)} images in {img_dir}\n") #scans the images in test (or other if wished) folder

    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    miou_all = []

    for img_path in paths:
        image_np = np.array(Image.open(img_path).convert("RGB"))
        aug      = transform(image=image_np)
        tensor   = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad(), autocast("cuda" if device.type == "cuda" else "cpu"):
            logits    = model(tensor)
            pred_mask = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        miou = None
        if mask_dir is not None:
            gt_path = mask_dir / (img_path.stem + ".png") #masks must be .png
            if gt_path.exists():
                gt_mask = np.array(Image.open(gt_path))
                miou    = compute_miou(pred_mask, gt_mask)
                miou_all.append(miou)
        mask_rgb   = mask_to_rgb(pred_mask) #generate visualisation
        blended_np = blend(image_np, mask_rgb, args.alpha)
        combined   = make_side_by_side(image_np, blended_np, miou, img_path.stem)

        combined.save(output_dir / f"{img_path.stem}_compare.png")
        Image.fromarray(pred_mask).save(output_dir / f"{img_path.stem}_mask.png")

        miou_str = f"  mIoU={miou:.4f}" if miou is not None else ""
        print(f"  {img_path.stem}{miou_str}")
    print(f"\nSaved to: {output_dir}")
    print("\nColour legend:")
    for cls_id, name in CLASS_NAMES.items():
        r, g, b = PALETTE[cls_id]
        print(f"  idx {cls_id}  {name:15s}: RGB({r:3d},{g:3d},{b:3d})")

    if miou_all:
        print(f"\nDataset mIoU (over {len(miou_all)} images): {np.mean(miou_all):.4f}")


if __name__ == "__main__":
    main()