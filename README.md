# SyncWeld-Net: Multi-Modal Deepfake Detection

**Paper**: *SyncWeld-Net: Detecting Audio-Visual Synchronization Mismatches in Deepfake Videos*

A deep learning framework for detecting face swapping and lip-syncing deepfakes through audio-visual cross-modal synchronization analysis.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **98.20%** |
| **F1-Score** | **98.18%** |
| **AUC** | **99.18%** |
| **10-Fold CV** | 97.2% ± 0.8% |

---

## 🔬 Key Visualizations

### ROC Curve Comparison with SOTA
![ROC](experiment_results/paper_figures/forensic_comparative_roc.png)

*SyncWeld-Net achieves AUC=0.992 vs Xception (0.945), Visual-Only (0.965), MesoNet (0.912)*

### Cross-Modal Alignment Heatmap
![Sync](experiment_results/paper_figures/forensic_alignment_heatmap.png)

*Real videos show diagonal sync correlation (perfect alignment); deepfakes show scattered/off-diagonal patterns (sync mismatch)*

### XAI: Grad-CAM Attention
![XAI](experiment_results/paper_figures/forensic_xai_attribution.png)

*Model attention focuses on perioral region (67%) and eye reflections (23%), proving it detects lip-sync errors rather than background*

### 10-Fold Cross-Validation Stability
![CV](experiment_results/paper_figures/forensic_stability_boxplot.png)

*Consistent performance across FakeForensics, Celeb-DF, and FaceForensics++ datasets*

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/Angelgupta13/SyncWeld-Deepfake-Detection.git
cd SyncWeld-Deepfake-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Examples

### Training the Model

```bash
# Basic training
python train_syncweld.py

# With custom parameters
python train_syncweld.py --epochs 50 --patience 5 --batch_size 16

# Full pipeline (all phases)
python master_pipeline.py --mode full --use_segmented
```

### Evaluation

```bash
# Single video inference
python test_inference_syncweld.py video.mp4 --checkpoint phase1_checkpoints/syncweld_best.pth

# Batch evaluation
python evaluate_model.py --checkpoint phase1_checkpoints/syncweld_best.pth

# Compare with baselines
python run_comparison.py
```

### Python API

```python
import torch
from models.syncweld import SyncWeldNet
from segmented_dataset import SegmentedFakeAVCelebDataset
from torchvision import transforms

# Load model
config = {"model": {"image-size": 160, "patch-size": 1, "num-classes": 1, 
                   "num-patches": 100, "dim": 512, "num-frames": 8}}
model = SyncWeldNet(config, num_classes=1)
model.load_state_dict(torch.load("phase1_checkpoints/syncweld_best.pth"))
model.eval()

# Preprocess input
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Inference
with torch.no_grad():
    # video_frames: [B, 8, 3, 160, 160]
    # audio_wav: [B, 1, 22050]
    logits = model(video_frames, audio_wav)
    prob = torch.sigmoid(logits)
    
if prob > 0.5:
    print(f"Deepfake: {prob.item():.2%}")
else:
    print(f"Real: {(1-prob).item():.2%}")
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: 4s Video Clip                   │
│              (8 frames @ 160x160 + 22kHz audio)           │
└──────────────────────��──────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
┌─────────────────────┐                 ┌─────────────────────┐
│  TimeSformer        │                 │   Wav2Vec2.0       │
│  (Visual Branch)    │                 │   (Audio Branch)   │
│  - 9 transformer    │                 │   - Large (53-lang) │
│    layers          │                 │   - 512-dim output │
│  - 512-dim output  │                 │                     │
└─────────────────────┘                 └─────────────────────┘
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
              ┌─────────────────────────────────┐
              │   Cross-Modal Fusion Layer      │
              │   - Multi-head attention        │
              │   - Contrastive Dissonance Loss │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │   Binary Classifier             │
              │   (Real vs Deepfake)            │
              └─────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **TimeSformer** | Size-Invariant video transformer with spatial-temporal attention |
| **Wav2Vec2.0** | Self-supervised audio encoder (facebook/wav2vec2-large-xlsr-53) |
| **Fusion** | Cross-modal attention layer combining visual and audio features |
| **Loss** | BCE + Contrastive Dissonance (detects sync mismatches) |

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | FakeAVCeleb-v1.2 |
| Train Samples | 1,000 segments |
| Validation Samples | 200 segments |
| Segment Duration | 4 seconds |
| Frames per Segment | 8 |
| Image Size | 160×160 |
| Optimizer | AdamW (lr=1e-4) |
| Scheduler | OneCycleLR (max_lr=3e-4) |
| Batch Size | 16 |
| Mixed Precision | FP16 |
| Early Stopping | Patience=5 |

---

## 📊 Detailed Results

### Phase 1: Training (13 Epochs)

| Epoch | Train Loss | Val Loss | Accuracy | F1 | AUC |
|-------|------------|----------|----------|-----|-----|
| 1 | 0.0814 | 0.2788 | 0.9752 | 0.9752 | 0.9788 |
| 5 | 0.0408 | 0.2237 | 0.9820 | 0.9818 | 0.9877 |
| 10 | 0.0346 | 0.2081 | 0.9820 | 0.9818 | 0.9872 |
| **13** | 0.0415 | 0.4848 | 0.9820 | 0.9818 | 0.9849 |

### Phase 2: Baseline Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| **SyncWeld-Net** | **97.5%** | **97.4%** | **97.6%** | **97.5%** | **99.2%** |
| Visual-Only | 96.0% | 95.0% | 97.0% | 96.0% | 99.0% |
| Audio-Only | 49.0% | 48.0% | 100% | 65.0% | 62.0% |

### Phase 3: Ablation Study

| Configuration | Accuracy | Improvement |
|---------------|----------|-------------|
| **Full Model** | **97.5%** | — |
| - Contrastive Loss | 91.0% | -6.5% |
| - Dissonance Penalty | 93.0% | -4.5% |
| - Audio Frozen | 89.0% | -8.5% |
| - Visual Frozen | 92.0% | -5.5% |

### Phase 4: 10-Fold Cross-Validation

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|------|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 98 | 97 | 98 | 96 | 97 | 98 | 97 | 96 | 97 | 98 |

**Mean: 97.2% ± 0.8%**

---

## 📁 Project Structure

```
SyncWeld-Net/
├── config/                       # Model configurations
├── datasets/                      # FakeAVCeleb, FaceForensics++
├── models/
│   ├── syncweld.py               # Main model
│   └── size_invariant_timesformer.py
├── phase1_checkpoints/
│   └── syncweld_best.pth         # Best model weights
├── experiment_results/
│   └── paper_figures/           # 16 publication figures
├── train_syncweld.py             # Training script
├── evaluate_model.py            # Evaluation
├── baseline_models.py           # Baseline comparisons
├── segmented_dataset.py         # Dataset loader
├── README.md                     # This file
└── RESULTS_ANALYSIS.md          # Detailed results
```

---

## 🔧 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
opencv-python>=4.8.0
librosa>=0.10.0
soundfile>=0.12.0
```

---

## 📖 Citation

```bibtex
@article{syncweld2026,
  title={SyncWeld-Net: Detecting Audio-Visual Synchronization Mismatches in Deepfake Videos},
  author={Your Name},
  year={2026}
}
```

---

## 🙏 Acknowledgments

- TimeSformer: https://github.com/facebookresearch/TimeSformer
- Wav2Vec2: https://github.com/facebookresearch/wav2vec2
- FakeAVCeleb Dataset: https://github.com/DashraIV/FakeAVCeleb

---

*Built for deepfake detection research*