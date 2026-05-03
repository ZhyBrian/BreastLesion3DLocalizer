We have already uploaded the **ultrasound data capture and reconstruction module**.

We are still organizing and verifying the code for the ultrasound sequence segmentation & diagnosis module, as well as the 3D nipple-centric localization module, and it will be made available soon.

# Full pipeline Demo

https://youtu.be/wBAuzPZo6To



# HLST — Hybrid Lesion-informed Spatiotemporal Transformer

Sequence-level breast ultrasound video classification (benign vs. malignant) via a dual-stream spatial encoder + Longformer-based temporal encoder.

## Pretrained Weights

https://github.com/ZhyBrian/BreastLesion3DLocalizer/releases/tag/HLSTPretrainedWeights

| File                                     | Description                      | Source                                                       |
| ---------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `spatial_encoder_pretrained_weights.pth` | Spatial encoder (ConvFormer_MTL) | Pretrained on **Breast Ultrasound Image Dataset-4K** (4086 images from 4 public/private cohorts) for segmentation + classification |
| `hlst_temporal_transformer.pth`          | Longformer temporal encoder      | Pretrained on **large-scale natural video data** (NOT ultrasound) |
| `hlst_mlp_head.pth`                      | MLP classification head          | Same natural video pretraining                               |
| `hlst_cls_token.pth`                     | Learnable [CLS] token            | Same natural video pretraining                               |

> **⚠️ Important:** The temporal encoder and MLP head weights are **not** pretrained on ultrasound data due to the lack of large-scale ultrasound video datasets. To use HLST on your own ultrasound video data, you **must** fine-tune at least the temporal encoder + MLP head (and ideally train end-to-end) on your target dataset.

------

## Quick Start

### Dependencies

```bash
pip install torch torchvision transformers pyyaml
```

The spatial encoder module (`spatial_encoder/`) must be available in your Python path.

### Configuration

All model hyperparameters are specified in `hlst.yaml`:

```yaml
num_classes: 2
spatial_frozen: True          # True for inference, False for end-to-end training

spatial_args:
  img_size: 256
  in_chans: 3
  attn_drop_rate: 0.0
  drop_rate: 0.0
  pretrained_weights: ./weights/spatial_encoder_pretrained_weights.pth

temporal_type: longformer
temporal_pretrained_weights: ./weights/hlst_temporal_transformer.pth
mlp_head_weights: ./weights/hlst_mlp_head.pth
cls_token_weights: ./weights/hlst_cls_token.pth
temporal_args:
  EMBED_DIM: 768
  MLP_DIM: 768
  HIDDEN_DIM: 768
  MAX_POSITION_EMBEDDINGS: 288
  NUM_ATTENTION_HEADS: 12
  NUM_HIDDEN_LAYERS: 3
  ATTENTION_MODE: 'sliding_chunks'
  PAD_TOKEN_ID: -1
  ATTENTION_WINDOW: [18, 18, 18]
  INTERMEDIATE_SIZE: 3072
  ATTENTION_PROBS_DROPOUT_PROB: 0.1
  HIDDEN_DROPOUT_PROB: 0.1
  DROPOUT_RATE: 0.5
```

### Loading the Model

```python
import yaml
import torch
from argparse import Namespace
from model import HLST

def load_yaml(file):
    with open(file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return Namespace(**cfg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load config & build model
cfg = load_yaml("./hlst.yaml")
model = HLST(**vars(cfg)).to(device)

# Load fine-tuned weights (after training on your ultrasound video dataset)
model.load_state_dict(torch.load("your_finetuned_weights.pth", map_location=device))
model.eval()
```

------

## Inference

The model expects input tensors in the shape `(B, C, F, H, W)` where:

- `B` — batch size
- `C` — channels (3 for RGB)
- `F` — number of sampled frames (full sequence supported; or sub-sampled for quicker inference)
- `H, W` — spatial resolution (256 × 256)

```python
import torch

# video_tensor: (B, 3, F, 256, 256), float32, normalized to [0, 1]
# position_ids: (B, F), int64, frame indices in the original sequence

with torch.no_grad():
    logits = model(video_tensor.to(device), position_ids=pos_ids.to(device))
    probs = torch.softmax(logits, dim=1)          # (B, 2)
    p_malignant = probs[:, 1]                      # malignancy probability
    pred_label = (p_malignant > 0.5).long()        # 0=benign, 1=malignant
```

**With BUS-SAM-2 lesion masks** (recommended, for HLST's tumoral guidance):

```python
# mask_tensor: (B, 1, F, 256, 256), binary lesion mask from BUS-SAM-2
logits = model(video_tensor.to(device),
               masks=mask_tensor.to(device),
               position_ids=pos_ids.to(device))
```

> When `spatial_frozen=True` (inference mode), `forward()` returns only the classification logits `(B, num_classes)`.

**Frame sampling for inference:** uniformly sample `F` frames from the full video sequence by taking every `D // F`-th frame (where `D` is the total number of frames):

```python
D = total_frames
F = 24
step = D // F
selected = [i * step for i in range(F)]
```

------

## Training

To fine-tune HLST on your own breast ultrasound video dataset, set `spatial_frozen: False` in the YAML (or override at runtime) to enable end-to-end training with multi-task supervision.

### Training Configuration (Reference)

| Parameter       | Value                                                        |
| --------------- | ------------------------------------------------------------ |
| Optimizer       | Adam                                                         |
| Learning rate   | 1e-5                                                         |
| Batch size      | 2                                                            |
| Epochs          | 45                                                           |
| Scheduler       | CosineAnnealingLR                                            |
| Training frames | 12 (randomly sampled)                                        |
| Test frames     | full sequence supported; or sub-sampled for quicker inference, e.g. 24 (uniformly sampled) |

### Loss for end-to-end Training

When `spatial_frozen=False`, the model returns `(logits, out_f_mask, out_f_cls)`:

```python
# forward returns 3 outputs in training mode
pred, out_f_mask, out_f_cls = model(video, masks, position_ids)

# pred:        (B, 2)       — sequence-level classification logits
# out_f_mask:  (B*F, 1, H, W) — per-frame segmentation prediction
# out_f_cls:   (B*F, 2)     — per-frame classification logits
```

You can design your own loss combination.

The simplest loss only supervises the sequence-level classification logits:

```python
import loss.focal_loss as focal_loss

loss_cls   = focal_loss.BinaryFocalLoss(alpha_neg=..., gamma=...)

# Compute losses
loss = loss_cls(pred, label)             # sequence-level focal loss

loss.backward()
```

### Minimal Training Loop (an example)

```python
import yaml
from argparse import Namespace
from torch.utils.data import DataLoader
from model import HLST

def load_yaml(file):
    with open(file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return Namespace(**cfg)

train_loader = DataLoader(
    ...,
    batch_size=2, shuffle=True
)
test_loader = DataLoader(
    ...,
    batch_size=1, shuffle=False
)

# Override spatial_frozen for training
cfg = load_yaml("./hlst.yaml")
cfg_dict = vars(cfg)
cfg_dict['spatial_frozen'] = False    # enable end-to-end fine-tuning
model = HLST(**cfg_dict).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=45//3, eta_min=1e-5/1000
)

for epoch in range(1, 46):
    model.train()
    for video, masks, label, pos_ids in train_loader:
        video  = video.to(device)
        masks  = masks.to(device)
        label  = label.to(device)
        pos_ids = pos_ids.to(device)

        pred, out_f_mask, out_f_cls = model(video, masks, pos_ids)

        # ... compute multi-task loss (see above) ...
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
```

### Temporal-Encoder-Only Fine-Tuning

If you want to freeze the spatial encoder and only train the temporal components (faster, less GPU memory):

```python
cfg_dict['spatial_frozen'] = True       # freeze spatial encoder
model = HLST(**cfg_dict).to(device)

# Only optimize temporal encoder + MLP head
trainable_params = list(model.temporal_transformer.parameters()) \
                 + list(model.mlp_head.parameters()) \
                 + [model.cls_token]
optimizer = torch.optim.Adam(trainable_params, lr=1e-5)
```

Note: in this mode, `forward()` returns only `logits`.

------

## Temporal Attention Visualization

Use `forward_att()` to extract the Longformer attention maps for interpretability analysis:

```python
model.eval()
with torch.no_grad():
    attentions, global_attentions, num_frames = model.forward_att(
        video_tensor.to(device), position_ids=pos_ids.to(device)
    )
# attentions:        tuple of (B, num_heads, seq_len, window_size) per layer
# global_attentions: tuple of (B, num_heads, seq_len, num_global_tokens) per layer
# num_frames:        int, number of input frames (F)
```

The global attention on the `[CLS]` token reveals which frames the model considers most diagnostically significant — analogous to a sonographer's key-frame selection.

## File Structure

```
├── model.py                  # HLST model definition
├── hlst_helper.py            # HLSTLongformerModel & padding utilities
├── hlst.yaml                 # Model configuration
├── spatial_encoder/          # Spatial encoder modules (ConvFormer_MTL)
├── weights/
│   ├── spatial_encoder_pretrained_weights.pth
│   ├── hlst_temporal_transformer.pth
│   ├── hlst_mlp_head.pth
│   └── hlst_cls_token.pth
└── loss/
    ├── focal_loss.py
    └── ....py
```

------

# Citation

```bibtex
@article{zhang2026navigation,
  title={A navigation-guided 3D breast ultrasound scanning and reconstruction system for automated multi-lesion spatial localization and diagnosis},
  author={Zhang, Yi and Yan, Yulin and Wang, Kun and Cai, Muyu and Xiang, Yifei and Guo, Yan and Tu, Puxun and Ying, Tao and Chen, Xiaojun},
  journal={Medical Image Analysis},
  volume={110},
  pages={103965},
  year={2026},
  publisher={Elsevier}
}
```
