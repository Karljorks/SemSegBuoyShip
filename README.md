# Maritime Semantic Segmentation

Dense semantic segmentation for maritime scenes (water, sky, ship, buoy, land). Model architecture and encoder backbone are fully configurable.

---

## Requirements

```bash
pip install torch torchvision segmentation-models-pytorch albumentations torchmetrics opencv-python pillow numpy
```

---

## Dataset Structure

```
dataset/
├── train/      images/  masks/
├── validate/   images/  masks/
└── test/       images/  masks/
```

Masks are single-channel PNGs with pixel values 0–7 (class IDs). Image and mask filenames must share the same stem (e.g. `frame_001.jpg` ↔ `frame_001.png`).

---

## Quick Start

```bash
# ResNet-50 encoder — recommended
python train.py \
  --data_root ~/maritime/dataset \
  --output_dir ~/maritime/output \
  --encoder resnet50 \
  --epochs 100 --batch_size 8

# ResNet-101 — higher accuracy, more VRAM
python train.py \
  --data_root ~/maritime/dataset \
  --output_dir ~/maritime/output \
  --encoder resnet101 \
  --epochs 100 --batch_size 8
```

---

## Configuration

| Flag | Default | Description |
|---|---|---|
| `--data_root` | *(required)* | Dataset root with `train/`, `validate/`, `test/` |
| `--output_dir` | `./output` | Where checkpoints and logs are saved |
| `--encoder` | `resnet50` | Encoder backbone (any SMP-supported encoder) |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `8` | Batch size — reduce to `4` on smaller GPUs |
| `--lr` | `1e-3` | Initial learning rate |
| `--num_workers` | `4` | DataLoader workers |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--no_amp` | off | Disable mixed precision (use if you see NaN losses) |
| `--seed` | `42` | Random seed |

---

## Resuming Training

```bash
python train.py \
  --data_root ~/maritime/dataset \
  --output_dir ~/maritime/output \
  --resume ~/maritime/output/last_model.pth
```

---

## Switching Models

The architecture is defined in `build_model()`. To swap UNet++ for a different segmentation head, change the `smp` class and update `decoder_channels` to match.

**DeepLabV3+** (atrous convolutions, strong on large objects):
```python
model = smp.DeepLabV3Plus(
    encoder_name=encoder,
    encoder_weights="imagenet" if pretrained else None,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None,
)
```

**FPN** (fast, good multi-scale performance):
```python
model = smp.FPN(
    encoder_name=encoder,
    encoder_weights="imagenet" if pretrained else None,
    decoder_dropout=0.2,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None,
)
```

Any [SMP-supported architecture](https://github.com/qubvel/segmentation_models.pytorch) works — the rest of the training loop (loss, metrics, checkpointing) is architecture-agnostic.

---

## Outputs

```
output_dir/
├── best_model.pth      # Highest validation mIoU
├── last_model.pth      # Final epoch
├── history.json        # Per-epoch training curves
└── test_results.json   # Final test metrics and efficiency stats
```
