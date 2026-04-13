# SyncWeld-Net: Multi-Modal Deepfake Detection

A multi-modal deepfake detection framework using audio-visual synchronization analysis.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **98.20%** |
| **F1-Score** | **98.18%** |
| **AUC** | **99.18%** |
| **10-Fold CV** | 97.2% ± 0.8% |

---

## Visual Results

### ROC Comparison
![ROC](experiment_results/paper_figures/forensic_comparative_roc.png)

### Cross-Modal Sync Analysis
![Sync](experiment_results/paper_figures/forensic_alignment_heatmap.png)

### Model Attention (Grad-CAM)
![XAI](experiment_results/paper_figures/forensic_xai_attribution.png)

### 10-Fold Stability
![CV](experiment_results/paper_figures/forensic_stability_boxplot.png)

---

## Quick Start

```bash
# Training
python train_syncweld.py

# Evaluation
python evaluate_model.py --checkpoint phase1_checkpoints/syncweld_best.pth
```

---

## Architecture

- **Visual**: Size-Invariant TimeSformer (512D)
- **Audio**: Wav2Vec2.0 (512D)  
- **Fusion**: Cross-modal attention + Contrastive Dissonance Loss

---

## Dataset

- **Training**: 1,000 segments (FakeAVCeleb-v1.2)
- **Validation**: 200 segments
- **Balance**: 50% Real / 50% Deepfake
- **Segment Duration**: 4 seconds

---

## Citation

```bibtex
@article{syncweld2026,
  title={SyncWeld-Net: Detecting Audio-Visual Synchronization Mismatches in Deepfake Videos},
  author={Your Name},
  year={2026}
}
```