# SyncWeld-Net: Cross-Modal Attention for Deepfake Detection

SyncWeld-Net is a state-of-the-art multimodal deepfake detection framework that identifies forged videos by analyzing the structural dissonance between lip-motion (Visual) and voice energy (Audio). 

By leveraging the **MINTIME Size-Invariant TimeSformer** for spatial-temporal visual tokenization, and **Wav2Vec2.0** for generalized audio representations, the model utilizes a Cross-Modal attention layer to seamlessly "weld" audio and video modalities, revealing deepfakes based on synchronization abnormalities.

## Repository Contents
- **`analyze_dataset.py`:** Standalone tool to visualize dataset balance and health.
- **`fakeavceleb_dataset.py`:** PyTorch DataLoader structured for paired AV ingestion across balanced datasets.
- **`test_inference_syncweld.py`:** An out-of-the-box inference script. Detects deepfakes from *any* `.mp4` video via saved weights.
- **`train_syncweld.py`:** The primary training loop, utilizing AMP and Custom Contrastive Dissonance Loss for multimodal learning.
- **`models/`:** Directory holding the core `syncweld.py` fusion layers alongside the MINTIME TimeSformer architecture.

---

## ⚙️ Environment Setup

1. Clone the repository and navigate to the local directory.
2. Initialize and activate your Python environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
*(Optionally use `conda env create -f environment.yml` if utilizing Anaconda.)*

## 📚 Dataset Structure
This framework is built parallel to the **FakeAVCeleb** and **FaceForensics++** formats.
Your dataset should ideally have separate structural categories for Real and Fake variants if utilizing the default loader.

## 🚀 Usage 

### 1. Training the Model
You can start training directly from the command line. The model will automatically build balanced batches.
```bash
python train_syncweld.py
```
*Note: Due to large Transformer footprints, training requires a CUDA-enabled GPU with >12GB VRAM. Mixed Precision relies on `torch.amp` bindings.*

### 2. Inferring / Predicting a Video
Use the robust inference script to analyze an individual video without loading entire datasets.

```bash
python test_inference_syncweld.py "path/to/any/video.mp4" --weights "syncweld_epoch_8.pth"
```
**Example output:**
```
--------------------------------------------------
Prediction Result for sample_video.mp4:
  Result: FAKE
  Confidence: 99.32%
--------------------------------------------------
```
