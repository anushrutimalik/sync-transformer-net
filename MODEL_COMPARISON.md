# Deepfake Detection Models - Complete Comparison

## Our Model: MINTIME (SyncWeldNet)
- **Architecture**: Multi-Identity-size-iNvariant TIMEsformer
- **Input**: Video (8 frames) + Audio (wav2vec)
- **Features**: Audio-visual cross-modal fusion, contrastive dissonance loss
- **Dataset**: FakeAVCeleb_v1.2 (~21k videos)

---

## Market Models Comparison

| Model | Architecture | Input Type | Key Features | Reported AUC | Year | Notes |
|-------|-------------|------------|---------------|--------------|------|-------|
| **XceptionNet** | Separable CNN | Visual only | Pretrained ImageNet, transfer learning | 85-95% | 2018 | Baseline for deepfake detection |
| **EfficientNet-B0/B3** | Compound scaling CNN | Visual only | EfficientNet architecture | 88-97% | 2019 | Good accuracy-efficiency trade-off |
| **EfficientNet-B7** | Large CNN | Visual only | High capacity, 66M params | 90-98% | 2019 | Best accuracy among CNNs |
| **ResNet50** | 50-layer CNN | Visual only | Skip connections, deep architecture | 85-95% | 2015 | Classic baseline |
| **ResNet101** | 101-layer CNN | Visual only | Deeper than ResNet50 | 86-96% | 2015 | Better than ResNet50 |
| **MTCNN + VGG16** | Face detection + CNN | Visual only | Two-stage detection | 82-92% | 2019 | Traditional approach |
| **CNN+LSTM** | CNN + Temporal | Visual | Temporal modeling | 87-94% | 2019 | Captures temporal artifacts |
| **3D-CNN** | Volumetric CNN | Video | Spatiotemporal features | 88-95% | 2018 | Native video understanding |
| **I3D** | Inflated 3D | Video | Kinetics pretrained | 90-96% | 2018 | Two-stream capabilities |
| **SlowFast** | Two-pathway CNN | Video | Slow + Fast pathways | 91-97% | 2019 | Facebook AI research |
| **Vision Transformer (ViT)** | Transformer | Visual | Self-attention on patches | 86-95% | 2020 | Needs large datasets |
| **TimesFormer** | Transformer | Video | Space-time attention | 90-97% | 2021 | State-of-art video transformer |
| **ViViT** | ViT 3D | Video | Variants of ViT for video | 91-97% | 2022 | Google's approach |
| **Audio-Visual Fusion** | Multi-modal | Video + Audio | Cross-modal consistency | 93-99% | 2020-2024 | Best for audio-fake detection |
| **FCA3D** | 3D CNN | Video | Frequency domain features | 92-97% | 2023 | Uses frequency artifacts |
| **DeepForensics** | EfficientNet | Visual | Domain adaptation | 88-96% | 2020 | FaceForensics baseline |
| **Celeb-DF Specific** | EfficientNet-B0 | Visual | Celeb-DF optimized | 95-99% | 2021 | Dataset-specific |
| **Multi-task Learning** | CNN + Keypoints | Visual | Face boundaries + content | 91-97% | 2021 | Uses facial landmarks |
| **Self-Supervised** | Contrastive | Visual | No labels needed | 75-85% | 2022 | Emerging approach |
| **LSTM+Attention** | RNN + Attn | Video | Temporal attention | 89-95% | 2020 | Lightweight option |

---

## Our MINTIME vs Others

| Aspect | MINTIME (Ours) | XceptionNet | EfficientNet-B3 | TimesFormer | Audio-Visual Fusion |
|--------|---------------|-------------|-----------------|-------------|---------------------|
| **Modality** | Audio + Video | Video only | Video only | Video only | Audio + Video |
| **Architecture** | TIMEsformer | CNN | CNN | Transformer | Fusion |
| **Parameters** | ~50M | ~22M | ~12M | ~120M | ~60M |
| **Input** | 8 frames + wav | Single frame | Single frame | 8-16 frames | 8 frames + audio |
| **Temporal** | Yes | No | No | Yes | Yes |
| **Audio Fake Detection** | Yes | No | No | No | Yes |
| **Our Best AUC** | 98.97% | 85-95% | 88-97% | 90-97% | 93-99% |

---

## Datasets for Benchmarking

| Dataset | Videos | Types | Year |
|---------|--------|-------|------|
| FaceForensics++ | 1,000 | 7 methods | 2019 |
| FakeAVCeleb | 21,566 | 4 types | 2021 |
| Celeb-DF | 590 | High-quality | 2019 |
| DFDC | 5,000 | 8 methods | 2020 |
| WildDeepfake | 6,000 | In-wild | 2020 |

---

## Recommendation

For fair comparison, train these models on FakeAVCeleb:

1. **XceptionNet** - Available in codebase
2. **EfficientNet-B3** - Add to comparison
3. **TimesFormer** - Available (size_invariant_timesformer.py)
4. **ResNet50** - Easy to add via timm

Run: `python run_comparison.py`
