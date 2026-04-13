import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd

from models.syncweld import SyncWeldNet
from fakeavceleb_dataset import FakeAVCelebDataset
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {name}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Config
config = {
    'model': {
        'image-size': 160, 
        'patch-size': 1,
        'num-classes': 1,
        'num-patches': 100, 
        'dim': 512,
        'num-frames': 8, 
        'max-identities': 2,
        'depth': 9,
        'dim-head': 64,
        'heads': 8,
        'channels': 112, 
        'attn-dropout': 0.1,
        'ff-dropout': 0.1,
        'shift-tokens': False,
        'enable-size-emb': False, 
        'enable-pos-emb': True,
        'enable-identity-attention': False,
        'efficient-net-block': 7
    }
}

weights_path = "syncweld_epoch_8.pth"
if not os.path.exists(weights_path):
    print(f"Error: {weights_path} not found.")
    exit(1)

print("Initializing SyncWeldNet...")
model = SyncWeldNet(timesformer_config=config, audio_model_name='facebook/wav2vec2-large-xlsr-53', num_classes=1)
checkpoint = torch.load(weights_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 2. Dataset
print("Loading Validation Dataset...")
# Use 400 samples to keep inference time brief (~2 mins) but statistically significant for t-SNE and ROC
val_dataset = FakeAVCelebDataset(
    metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
    base_path="datasets/FakeAVCeleb_v1.2",
    num_frames=config['model']['num-frames'],
    image_size=config['model']['image-size'],
    target_sample_rate=16000,
    split='val',
    max_samples=400 
)
dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# 3. Inference
all_preds = []
all_labels = []
all_features = []
all_types = []

print("Running inference to extract features and probabilities...")
with torch.no_grad():
    for idx, (visual_x, audio_wav, labels) in enumerate(tqdm(dataloader)):
        visual_x = visual_x.to(device)
        audio_wav = audio_wav.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, audio_latents, visual_features = model(visual_x, audio_wav)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        # visual_features is [Batch, Seq, Dim] or [Batch, Dim]
        if len(visual_features.shape) == 3:
            vf = visual_features.mean(dim=1).cpu().numpy()
        else:
            vf = visual_features.cpu().numpy()
            
        all_preds.extend(probs)
        all_labels.extend(labels.numpy())
        all_features.append(vf)
        
        batch_start = idx * 4
        batch_end = min(batch_start + 4, len(val_dataset))
        for i in range(batch_start, batch_end):
            all_types.append(val_dataset.metadata[i]['type'])

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_features = np.vstack(all_features)
all_types = np.array(all_types)

# 4. Graph 10: ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
save_fig('paper_fig10_roc_curve.png')

# 5. Graph 11: PR Curve
precision, recall, _ = precision_recall_curve(all_labels, all_preds)
ap = average_precision_score(all_labels, all_preds)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision, color='purple', lw=2.5, label=f'PR curve (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
save_fig('paper_fig11_pr_curve.png')

# 6. Graph 12: t-SNE Feature Splitting
print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(all_features)

df_tsne = pd.DataFrame({
    'tsne-2d-one': tsne_results[:, 0],
    'tsne-2d-two': tsne_results[:, 1],
    'Label': ['Fake' if l == 1 else 'Real' for l in all_labels]
})

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Label",
    palette={'Real': 'blue', 'Fake': 'red'},
    data=df_tsne,
    legend="full",
    alpha=0.7,
    s=80
)
plt.title('t-SNE of SyncWeld Visual Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
save_fig('paper_fig12_tsne_features.png')

# 7. Graph 13: Modality-Specific Performance Breakdown
print("Computing modality breakdown...")
unique_types = np.unique(all_types)
type_accs = {}
for t in unique_types:
    idxs = np.where(all_types == t)[0]
    if len(idxs) > 0:
        t_preds = (all_preds[idxs] >= 0.5).astype(int)
        t_labels = all_labels[idxs]
        acc = (t_preds == t_labels).mean() * 100
        type_accs[t] = acc

df_types = pd.DataFrame(list(type_accs.items()), columns=['Deepfake Modality', 'Accuracy (%)'])
df_types = df_types.sort_values('Accuracy (%)', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy (%)', y='Deepfake Modality', data=df_types, hue='Deepfake Modality', legend=False, palette='coolwarm')
plt.xlim(0, 100)
plt.title('Accuracy by Deepfake Modality')
for index, value in enumerate(df_types['Accuracy (%)']):
    plt.text(value - 5, index, f'{value:.1f}%', va='center', color='white', fontweight='bold')
save_fig('paper_fig13_modality_breakdown.png')

print("Advanced graphs generated successfully!")
