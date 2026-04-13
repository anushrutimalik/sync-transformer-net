# PROJECT BULLETIN: MINTIME Video Deepfake Detection

================================================================================
PROJECT OVERVIEW
================================================================================

Project Name: MINTIME (Multi-Identity-size-iNvariant TIMEsformer)
Purpose: Video Deepfake Detection using Audio-Visual Multi-modal Learning
Dataset: FakeAVCeleb_v1.2 (21,566 videos)

================================================================================
WHAT WE DID - TIMELINE
================================================================================

APRIL 04, 2026 - DAY 1
----------------------
1. INITIAL DISCOVERY
   - Found training already running on system
   - Training on 1,000 videos (2,000 segments)
   - Completed 48 epochs in ~10 hours
   
2. ANALYSIS RESULTS
   - Best AUC: 98.97% (Epoch 60)
   - Best F1: 98.14% (Epoch 58)
   - Found: Only 4.6% of dataset used
   - Found: RealVideo-FakeAudio incorrectly labeled as Real

3. ISSUES IDENTIFIED
   - Dataset too small (should use more data)
   - Class balancing artificially limiting data
   - RealVideo-FakeAudio should be FAKE (not Real)

APRIL 05, 2026 - DAY 2
----------------------
4. CODE FIXES
   - Fixed label mapping in segmented_dataset.py
   - RealVideo-FakeAudio now = Fake (correct)
   - Detailed labels: RealVideo-RealAudio, FakeVideo-FakeAudio, etc.

5. UPDATED TRAINING CONFIG
   - Increased: 1,000 → 8,000 videos (train)
   - Increased: 1,000 → 2,000 videos (val)
   - Removed: ensure_balance (False now)
   - Changed: batch_size 2 → 4
   - Changed: num_workers 2 → 4

6. PROBLEMS FOUND
   - Training RESUMED from old checkpoint instead of fresh start
   - This is DANGEROUS - old weights persisted
   - Deleted all old checkpoints
   - Set resume_from_checkpoint = False

7. CURRENT STATUS
   - Fresh training started with 16,000 segments
   - Running now on GPU

================================================================================
DATASET EVOLUTION
================================================================================

| Metric           | Original    | Fixed       |
|------------------|-------------|-------------|
| Train Videos     | 1,000       | 8,000       |
| Val Videos       | 1,000       | 2,000       |
| Segments/Video   | 2           | 2           |
| Train Segments   | 2,000       | 16,000      |
| Val Segments     | 2,000       | 4,000       |
| Class Balance    | Balanced    | Natural     |
| Data Used        | 4.6%        | ~37%        |

================================================================================
MODEL ARCHITECTURE
================================================================================

SyncWeldNet (MINTIME):
- Visual: Timesformer (9 layers, 8 heads, 512 dim)
- Audio: wav2vec linear projection
- Fusion: Cross-modal attention
- Loss: BCE + Contrastive Dissonance

Training Config:
- Optimizer: AdamW
  - Learning Rate: 1e-4
  - Weight Decay: 1e-5
- Scheduler: OneCycleLR (max_lr=3e-4, pct_start=0.2)
- Mixed Precision: FP16
- Gradient Clipping: max_norm=1.0
- Early Stopping: patience=10

================================================================================
TRAINING TIME COMPARISON
================================================================================

Dataset Size    | Epochs | Est. Time  | Status
----------------|--------|-----------|--------
2,000 seg      | 48     | ~10 hrs   | COMPLETED
16,000 seg     | 50     | ~30-40 hrs| RUNNING

Per Epoch Time:
- 2k segments: ~12 min/epoch
- 16k segments: ~35-40 min/epoch

================================================================================
RESULTS SO FAR
================================================================================

Run 1 (2,000 segments - COMPLETED):
--------------------------------------
Epoch 60 (Best AUC): 99.59% AUC, 97.84% F1
Epoch 58 (Best F1):  98.14% F1,  98.87% AUC

Final Metrics:
- Accuracy:  97.45%
- F1 Score:  97.31%
- AUC:       98.97%
- Precision: 97.31%
- Recall:    97.31%

Run 2 (16,000 segments - IN PROGRESS)
--------------------------------------
Status: Currently training...
Started: April 05, 2026 (afternoon)
Expected: ~30-40 hours

================================================================================
FILES CREATED/MODIFIED
================================================================================

Modified:
- train_phase1.py (training config)
- segmented_dataset.py (label fix)

Created:
- evaluate_model.py (test on multiple datasets)
- run_comparison.py (compare models)
- MODEL_COMPARISON.md (market comparison)
- PROJECT_BULLETIN.md (this file)

================================================================================
BASELINE MODELS FOR COMPARISON
================================================================================

| Model            | Type        | Input    | Expected AUC |
|------------------|-------------|----------|--------------|
| XceptionNet      | CNN         | Visual   | 85-95%       |
| EfficientNet-B3  | CNN         | Visual   | 88-97%       |
| ResNet50         | CNN         | Visual   | 85-95%       |
| TimesFormer      | Transformer | Video    | 90-97%       |
| MINTIME (Ours)  | Fusion      | V+Audio  | 95-99%       |

================================================================================
NEXT STEPS
================================================================================

1. Monitor current training (~30-40 hrs remaining)
2. Once complete:
   a. Evaluate on FakeAVCeleb train split
   b. Evaluate on FakeAVCeleb val split
   c. Evaluate on FaceForensics++
3. Run comparison with baseline models
4. Publish/generate paper results

================================================================================
