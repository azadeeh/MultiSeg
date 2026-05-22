# MultiSeg: Lightweight 3D Brain Tumor Segmentation via Axial Transformer and Modality-Correlated Feature Fusion

> **Thesis Project** — Islamic Azad University, Science and Research Branch  
> Faculty of Ibn Sina · Department of Artificial Intelligence and Robotics  
> Author: **Azadeh Rastgoo** | Advisor: **Dr. Mehdi Mazinani** | Consultant: **Dr. Abbas Kouchari**

---

## Overview

**MultiSeg** is a lightweight hybrid 3D architecture for automated brain tumor segmentation on multi-modal MRI images, evaluated on the BraTS 2021 dataset.

The key insight of this work: **encoding clinical knowledge directly into the architecture is more parameter-efficient than simply scaling model capacity.** MultiSeg achieves state-of-the-art Dice scores for Whole Tumor (WT) and Tumor Core (TC) with only **20.68M parameters** — the lightest model among all compared methods.

---

## Results

| Method | Params | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET |
|---|---|---|---|---|---|---|---|
| nnU-Net [29] | 31.2M | 0.913 | 0.868 | 0.812 | — | — | — |
| TransBTS [36] | 33.0M | 0.902 | 0.854 | 0.813 | — | — | — |
| SwinBTS [41] | 42.2M | 0.918 | 0.887 | 0.831 | — | — | — |
| CKD-TransBTS [43] | 35.6M | 0.921 | 0.896 | **0.849** | — | — | **2.97** |
| **MultiSeg (Ours)** | **20.68M** | **0.926** | **0.904** | 0.820 | 5.06 | 4.98 | 5.22 |

> All results on **BraTS 2021** test set (150 samples). ↑ Higher is better for Dice. ↓ Lower is better for HD95.

---

## Architecture

MultiSeg is built on four core design principles:

**1. Dual-Branch Encoder**  
Processes modality pairs separately — structural pair (T1/T1Gd) in Branch A, and tissue pair (T2/FLAIR) in Branch B — explicitly encoding clinical knowledge into the architecture.

**2. 3D Axial Self-Attention**  
Models long-range spatial dependencies along each axis (D, H, W) independently, with lower computational cost compared to full 3D self-attention.

**3. Cross-Axial Attention Bottleneck (MCCA)**  
Fuses information across branches in the bottleneck via modality-correlated cross-attention.

**4. SkipGate + TCFC Decoder**  
Adaptive gating on skip connections filters noise from background regions. TCFC blocks calibrate features before upsampling.

**5. Teacher EMA Knowledge Distillation**  
Stabilizes training under class imbalance via a linearly warmed-up Teacher EMA schedule, acting as a temporal regularizer.

---

## Computational Specs

| Metric | Value |
|---|---|
| Parameters (total) | 20.68M |
| Parameters (trainable) | 20.68M |
| FLOPs (per patch) | 430.9 GFLOPs |
| Inference time (no TTA) | ~456 ms |
| Inference time (TTA ×8) | ~3.65 s |
| Peak GPU RAM | ~9.97 GB |

---

## Dataset

**BraTS 2021** — 1,251 multi-modal MRI cases (T1, T1Gd, T2, FLAIR).

| Split | Samples |
|---|---|
| Train | 1000 |
| Validation | 100 |
| Test | 150 |

Three tumor sub-regions are segmented:
- **WT** — Whole Tumor (labels 1+2+3)
- **TC** — Tumor Core (labels 1+3)
- **ET** — Enhancing Tumor (label 3)

---

## Installation

```bash
git clone https://github.com/<your-username>/MultiSeg.git
cd MultiSeg
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0
monai
numpy
scipy
fvcore
```

---

## Usage

### Inference on a single volume

```python
from MultiSeg_Model2 import MultiModalSegNet
import torch

model = MultiModalSegNet(
    base_ch=64,
    num_heads=8,
    num_classes=4,
    deep_supervision=False
)

ckpt = torch.load("BEST_MODEL.pt", map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
model.load_state_dict(sd, strict=False)
model.eval()

# Input: (B, 4, D, H, W) — 4 MRI modalities
x = torch.randn(1, 4, 160, 160, 160)
with torch.no_grad():
    output = model(x)
    logits = output["logits"]  # (B, 4, D, H, W)
```

### Sliding Window Inference (recommended for full volumes)

```python
from monai.inferers import sliding_window_inference

pred = sliding_window_inference(
    inputs=x,
    roi_size=(160, 160, 160),
    sw_batch_size=4,
    predictor=lambda z: model(z)["logits"],
    overlap=0.5
)
```

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW + Lookahead |
| Learning rate | Cosine schedule (1e-4 → 5e-6) |
| Loss function | Dice + Weighted CE + Focal Loss |
| Batch size | 2 |
| Patch size | 160×160×160 |
| Epochs | 137 |
| Best epoch | 104 |
| Hardware | Google Colab A100 (40GB) |

**Loss evolution:**
- Epochs 1–108: Dice Loss + Weighted Cross-Entropy
- Epoch 108+: Added Focal Loss (γ=2) to address ET plateau

---

## Key Findings

- MultiSeg with **34% fewer parameters than nnU-Net** achieves higher Dice on WT and TC
- Training stability: Score difference between best epoch (104) and last epoch (137) is only **0.0054**
- Main limitation: HD95 gap on ET region (5.22 mm vs. 2.97 mm for CKD-TransBTS) — targeted for future work via boundary-aware losses

---

## Citation

If you use this work, please cite:

```
@mastersthesis{rastgoo2026multiseg,
  author  = {Azadeh Rastgoo},
  title   = {Brain Tumor Segmentation in MRI Images Using a Combination of
             Lightweight 3D Axial Transformer and Modality Correlation Mechanisms},
  school  = {Islamic Azad University, Science and Research Branch},
  year    = {2026},
  advisor = {Dr. Mehdi Mazinani},
}
```

---

## License

This project is released for academic use only.
