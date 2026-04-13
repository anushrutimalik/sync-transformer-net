# SyncWeld-Net: Multi-Modal Deepfake Detection

A state-of-the-art multi-modal deepfake detection framework combining Size-Invariant TimeSformer (visual) and Wav2Vec2.0 (audio) with Contrastive Dissonance Loss for detecting face swapping and lip-syncing forgeries.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.20% |
| F1-Score | 98.18% |
| AUC | 99.18% |
| 10-Fold CV | 97.2% ± 0.8% |

### Model Comparison

| Model | Accuracy | AUC |
|-------|----------|-----|
| **SyncWeld-Net** | **97.5%** | **99.2%** |
| Visual-Only (TimeSformer) | 96.0% | 99.0% |
| Audio-Only (Wav2Vec2) | 49.0% | 62.0% |

---

## Architecture

```
Input Video → TimeSformer (Visual) ─┐
                                 ├──→ Cross-Modal Fusion → Classifier
Input Audio → Wav2Vec2.0 (Audio) ──┘
                                 │
                                 └─→ Contrastive Dissonance Loss
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Training
```bash
python master_pipeline.py --mode full --use_segmented
```

### Evaluation
```bash
python evaluate_model.py --checkpoint phase1_checkpoints/syncweld_best.pth
```

---

## Dataset

- Training: 1,000 video segments (4-second clips)
- Validation: 200 segments
- Class balance: 50% Real, 50% Deepfake

---

## Citation

```bibtex
@article{syncweld2026,
  title={SyncWeld-Net: Detecting Audio-Visual Synchronization Mismatches in Deepfake Videos},
  author={Gupta, A.},
  year={2026}
}
```

---

## License

MIT License