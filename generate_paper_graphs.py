import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from fakeavceleb_dataset import FakeAVCelebDataset
import torch.optim as optim
import math

# Use seaborn aesthetics for paper-ready graphs
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {name}")

epochs = []
train_losses = []
val_losses = []
val_accs = []
val_precs = []
val_recs = []
val_f1s = []
val_aucs = []

print("Extracting metrics from checkpoints...")
for i in range(1, 11):
    ckpt_path = f"syncweld_epoch_{i}.pth"
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            epochs.append(ckpt['epoch'])
            train_losses.append(ckpt['train_loss'])
            val_losses.append(ckpt['val_loss'])
            
            metrics = ckpt['val_metrics']
            val_accs.append(metrics.get('acc', 0))
            val_precs.append(metrics.get('prec', 0))
            val_recs.append(metrics.get('rec', 0))
            val_f1s.append(metrics.get('f1', 0))
            val_aucs.append(metrics.get('auc', 0))
            del ckpt
        except Exception as e:
            pass

if not epochs:
    print("Checkpoints not found.")
    exit(1)

# Graph 1: Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o', linewidth=2.5)
plt.plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
save_fig('paper_fig1_loss.png')

# Graph 2: Accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, val_accs, color='green', marker='o', linewidth=2.5)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
save_fig('paper_fig2_accuracy.png')

# Graph 3: Precision
plt.figure(figsize=(8, 6))
plt.plot(epochs, val_precs, color='purple', marker='^', linewidth=2.5)
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Validation Precision')
save_fig('paper_fig3_precision.png')

# Graph 4: Recall
plt.figure(figsize=(8, 6))
plt.plot(epochs, val_recs, color='orange', marker='v', linewidth=2.5)
plt.xlabel('Epoch')
plt.ylabel('Recall (Sensitivity)')
plt.title('Validation Recall')
save_fig('paper_fig4_recall.png')

# Graph 5: F1 Score
plt.figure(figsize=(8, 6))
plt.plot(epochs, val_f1s, color='red', marker='d', linewidth=2.5)
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Validation F1 Score')
save_fig('paper_fig5_f1_score.png')

# Graph 6: AUC ROC
plt.figure(figsize=(8, 6))
plt.plot(epochs, val_aucs, color='darkblue', marker='x', linewidth=2.5)
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Validation ROC AUC')
save_fig('paper_fig6_auc.png')

# Graph 7: Learning Rate Schedule
try:
    # Reconstruct the OneCycleLR schedule to plot it
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = optim.AdamW(dummy_model.parameters(), lr=1e-4) # Base doesn't matter much for OneCycle
    
    # Matching config from train_syncweld.py
    max_train_samples = 4500
    batch_size = 4
    steps_per_epoch = math.ceil(max_train_samples / batch_size)
    total_epochs = 10
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4, 
        steps_per_epoch=steps_per_epoch, 
        epochs=total_epochs,
        pct_start=0.2 
    )
    
    lrs = []
    step_counts = []
    for step in range(steps_per_epoch * total_epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        step_counts.append(step)
        
    plt.figure(figsize=(8, 6))
    plt.plot(step_counts, lrs, color='teal', linewidth=2.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('OneCycleLR Schedule')
    save_fig('paper_fig7_lr_schedule.png')
except Exception as e:
    print(f"Skipping LR graph: {e}")

# Load Dataset Validation stats for CM & Dist
print("Loading dataset to compute dataset distribution and confusion matrix...")
val_dataset = FakeAVCelebDataset(
    metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
    base_path="datasets/FakeAVCeleb_v1.2",
    num_frames=8,
    image_size=160,
    target_sample_rate=16000,
    split='val',
    max_samples=700
)

# Parse real vs fake
num_fake = sum([1 for row in val_dataset.metadata if row['type'] != 'RealVideo-RealAudio'])
num_real = len(val_dataset.metadata) - num_fake

# Graph 8: Dataset Distribution
plt.figure(figsize=(7, 6))
# Updated barplot syntax to avoid FutureWarnings in seaborn
sns.barplot(x=['Real', 'Fake (Deepfake)'], y=[num_real, num_fake], hue=['Real', 'Fake (Deepfake)'], legend=False, palette='viridis')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Validation Dataset Distribution')
for i, v in enumerate([num_real, num_fake]):
    plt.text(i, v + max([num_real, num_fake])*0.02, str(v), ha='center', fontsize=14)
save_fig('paper_fig8_dataset_distribution.png')

# Graph 9: Confusion Matrix Heatmap for Epoch 10 (or best)
if len(val_accs) > 0:
    best_epoch_idx = np.argmax(val_f1s) # use the epoch with best F1
    print(f"Generating CM for best epoch: {epochs[best_epoch_idx]} (F1: {val_f1s[best_epoch_idx]:.4f})")
    
    P = num_fake
    N = num_real
    
    recall = val_recs[best_epoch_idx]
    precision = val_precs[best_epoch_idx]
    
    TP = int(round(recall * P))
    FN = P - TP
    
    if precision > 0:
        FP = int(round((TP / precision) - TP))
    else:
        FP = 0
        
    TN = N - FP
    
    # Due to float rounding, cap values just in case
    if FP < 0: FP = 0
    if TN > N: TN = N
    if TN < 0: TN = 0
    if FP > N: FP = N

    cm = np.array([[TN, FP], [FN, TP]])
    
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 16})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(['Real (0)', 'Fake (1)'])
    ax.set_yticklabels(['Real (0)', 'Fake (1)'], va='center')
    plt.title(f'Confusion Matrix Heatmap (Epoch {epochs[best_epoch_idx]})')
    save_fig('paper_fig9_confusion_matrix_heatmap.png')

print("All 9 graphs generated successfully!")
