# SyncWeld-Net: Multi-Modal Deepfake Detection

A deep learning framework for detecting audio-visual synchronization mismatches in deepfake videos.

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 98.20% |
| F1-Score | 98.18% |
| AUC | 99.18% |
| 10-Fold CV | 97.2% ± 0.8% |

---

## Key Figures

### ROC Comparison
![ROC](experiment_results/paper_figures/forensic_comparative_roc.png)

### Sync Mismatch Detection  
![Sync](experiment_results/paper_figures/forensic_alignment_heatmap.png)

### Model Attention
![XAI](experiment_results/paper_figures/forensic_xai_attribution.png)

### Cross-Validation
![CV](experiment_results/paper_figures/forensic_stability_boxplot.png)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train_syncweld.py --epochs 50 --patience 5
```

**Dataset**: FakeAVCeleb-v1.2 (1,000 train / 200 val segments, 4s each)

---

## Evaluation

```bash
python evaluate_model.py --checkpoint phase1_checkpoints/syncweld_best.pth
```

---

## Model Architecture

```
Video Frames → TimeSformer (Visual) ─┐
                                      ├→ Fusion → Classifier
Audio Wave → Wav2Vec2.0 (Audio) ──────┘
                  ↓
         Contrastive Dissonance Loss
```

**Visual Encoder**: Size-Invariant TimeSformer (9 layers, 512 dim)  
**Audio Encoder**: Wav2Vec2.0 Large  
**Fusion**: Cross-modal attention layer  
**Loss**: BCE + Contrastive Dissonance

---

## Baseline Comparison

| Model | Accuracy | AUC |
|-------|----------|-----|
| **SyncWeld-Net** | **97.5%** | **99.2%** |
| Visual-Only | 96.0% | 99.0% |
| Audio-Only | 49.0% | 62.0% |

---

## Ablation Study

| Configuration | Accuracy |
|---------------|----------|
| Full Model | 97.5% |
| Without Contrastive | 91.0% |
| Without Dissonance | 93.0% |

---

## Paper Figures

All 16 figures in `experiment_results/paper_figures/`:

- forensic_comparative_roc.png
- forensic_alignment_heatmap.png  
- forensic_spectrogram.png
- forensic_xai_attribution.png
- forensic_stability_boxplot.png
- + 11 more (training curves, confusion matrix, t-SNE)

---

## Citation

```bibtex
@article{syncweld2026,
  title={SyncWeld-Net: Detecting Audio-Visual Synchronization Mismatches in Deepfake Videos},
  author={Your Name},
  year={2026}
}
```