# SyncWeld-Net: Multi-Modal Deepfake Detection

**[Paper]: SyncWeld-Net: Detecting Audio-Visual Synchronization Mismatches in Deepfake Videos**

A state-of-the-art multi-modal deepfake detection framework combining Size-Invariant TimeSformer and Wav2Vec2.0 for detecting face swapping and lip-syncing forgeries through audio-visual synchronization analysis.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **98.20%** |
| **F1-Score** | **98.18%** |
| **AUC** | **99.18%** |
| **10-Fold CV** | 97.2% ± 0.8% |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python master_pipeline.py --mode full --use_segmented

# Evaluate
python evaluate_model.py --checkpoint phase1_checkpoints/syncweld_best.pth
```

---

## 🏗️ Architecture

```
Input Video (4s, 8 frames)
    │
    ├─► TimeSformer (Visual, 512D) ──┐
    │                               │
    └─► Wav2Vec2.0 (Audio, 512D) ───┼─► Cross-Modal Fusion ──► Classifier
                                   │
                                   └─► Contrastive Dissonance Loss
```

### Key Features
- **Size-Invariant Attention**: Handles varying video resolutions
- **Contrastive Dissonance**: Detects audio-visual sync mismatches
- **End-to-End Training**: Joint audio-visual optimization

---

## 📁 Project Structure

```
SyncWeld-Net/
├── README.md                    # Main documentation
├── RESULTS_ANALYSIS.md         # Detailed results
├── requirements.txt            # Dependencies
├── master_pipeline.py          # Complete pipeline
├── train_syncweld.py          # Main training
├── evaluate_model.py          # Evaluation
├── baseline_models.py        # Baselines
├── ablation_study.py         # Ablation experiments
├── extended_training.py     # Training utilities
├── segmented_dataset.py     # Dataset loader
├── fakeavceleb_dataset.py    # Dataset
├── models/                   # Model code
│   ├── syncweld.py
│   └── size_invariant_timesformer.py
├── phase1_checkpoints/       # Training history
├── experiment_results/       # Results
│   └── paper_figures/      # Publication figures
└── datasets/                # Data (not in repo)
```

---

## 📊 Results Summary

### Phase 1: Training
| Metric | Best Value |
|--------|----------|
| Accuracy | 98.20% |
| F1-Score | 98.18% |
| AUC | 99.18% |

### Phase 2: Comparison
| Model | Accuracy | AUC |
|-------|----------|-----|
| **SyncWeld-Net** | **97.5%** | **99.2%** |
| Visual-Only | 96.0% | 99.0% |
| Audio-Only | 49.0% | 62.0% |

### Phase 3: Ablation
| Configuration | Accuracy |
|---------------|----------|
| Full Model | 97.5% |
| - Contrastive | 91.0% |
| - Dissonance | 93.0% |

### Phase 4: 10-Fold CV
- **Mean Accuracy**: 97.2% ± 0.8%

---

## 🛠️ Usage

### Training
```bash
python train_syncweld.py --epochs 50 --patience 5
python master_pipeline.py --mode full --use_segmented
```

### Evaluation
```bash
python evaluate_model.py --checkpoint phase1_checkpoints/syncweld_best.pth
```

### Inference
```python
from models.syncweld import SyncWeldNet
import torch

model = SyncWeldNet(config, num_classes=1)
checkpoint = torch.load("phase1_checkpoints/syncweld_best.pth")
model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    logits = model(video_frames, audio_wav)
    prob = torch.sigmoid(logits)
    print(f"Deepfake: {prob.item():.2%}")
```

---

## 📦 Requirements

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

## 📊 Dataset

- **Training**: 1,000 video segments (FakeAVCeleb-v1.2)
- **Validation**: 200 video segments
- **Class Balance**: 50% Real, 50% Deepfake
- **Segment Duration**: 4 seconds
- **Frames per Segment**: 8 frames

---

## 📊 Publication Figures

All figures are saved in `experiment_results/paper_figures/`:

### Core Forensics (6 figures)
- `forensic_alignment_heatmap.png` - Cross-modal sync analysis
- `forensic_spectrogram.png` - Audio GAN artifacts
- `forensic_xai_attribution.png` - Grad-CAM attention
- `forensic_comparative_roc.png` - ROC vs SOTA
- `forensic_stability_boxplot.png` - 10-fold CV
- `forensic_efficiency_scatter.png` - Accuracy vs Latency

### Training Curves (4 figures)
- `paper_fig1_loss.png`, `paper_fig2_accuracy.png`, `paper_fig5_f1_score.png`, `paper_fig6_auc.png`

### Evaluation (3 figures)
- `paper_fig9_confusion_matrix_heatmap.png`, `paper_fig10_roc_curve.png`, `paper_fig11_pr_curve.png`

### Advanced (3 figures)
- `paper_fig12_tsne_features.png`, `paper_fig13_modality_breakdown.png`, `paper_fig8_dataset_distribution.png`

---

## 📖 Citation

```bibtex
@article{syncweld2026,
  title={SyncWeld-Net: Detecting Audio-Visual Synchronization Mismatches in Deepfake Videos},
  author={Your Name},
  year={2026},
  journal={arXiv preprint}
}
```

---

## ⚖️ License

MIT License

---

## 👤 Author

- **Name**: [Your Name]
- **Email**: your.email@example.com

*Built with ❤️ for deepfake detection research*