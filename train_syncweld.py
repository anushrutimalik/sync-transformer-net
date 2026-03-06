import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from models.syncweld import SyncWeldNet
from fakeavceleb_dataset import FakeAVCelebDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Maximum GPU acceleration
torch.backends.cudnn.benchmark = True

class ContrastiveDissonanceLoss(nn.Module):
    """
    Penalizes mismatch between audio energy and lip-motion velocity (visual tokens).
    L_CD = 1/N * sum( |E_a - V_v| ) + standard BCE for fake/real classification.
    """
    def __init__(self, alpha=0.5, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels, audio_latents, visual_features):
        """
        logits: Output prediction for deepfake detection [Batch, 1]
        labels: Ground truth (1 for Fake, 0 for Real) [Batch, 1]
        audio_latents: [Batch, SeqLen, AudioDim]
        visual_features: [Batch, SeqLen, VisualDim]
        """
        # Baseline Classification Loss
        cls_loss = self.bce(logits, labels.float())
        
        # Simulated Contrastive Dissonance (Audio Energy vs Visual Velocity)
        # Here we approximate Energy by the L2 norm of the audio latents
        # and Velocity by the L2 norm of the difference between adjacent visual frames
        
        audio_energy = torch.norm(audio_latents, dim=-1).mean(dim=1) # [Batch]
        
        # If visual_features has a temporal dimension: diff between consecutive frames
        if len(visual_features.shape) == 3 and visual_features.shape[1] > 1:
            visual_velocity = torch.norm(visual_features[:, 1:, :] - visual_features[:, :-1, :], dim=-1).mean(dim=1)
        else:
            # Fallback if there's no sequence dimension (e.g. only CLS token)
            visual_velocity = torch.norm(visual_features, dim=-1).mean(dim=1)
            
        # Normalize both to similar scales for stable loss
        norm_audio = audio_energy / (audio_energy.max() + 1e-8)
        norm_visual = visual_velocity / (visual_velocity.max() + 1e-8)
        
        # We assume real videos have synchronized energy/velocity. Fake videos often have dissonance.
        # This is strictly a self-supervised auxiliary regularization.
        dissonance_loss = torch.mean(torch.abs(norm_audio - norm_visual))
        
        total_loss = cls_loss + self.alpha * dissonance_loss
        return total_loss, cls_loss, dissonance_loss

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (visual_x, audio_wav, labels) in enumerate(pbar):
        visual_x = visual_x.to(device, non_blocking=True)
        audio_wav = audio_wav.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True) # Faster than zero_grad()
        
        # Forward pass with AMP
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, audio_latents, visual_features = model(visual_x, audio_wav)
            loss, cls_loss, cd_loss = criterion(logits, labels, audio_latents, visual_features)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Step and reset gradients
        scaler.step(optimizer)
        scaler.update()
            
        if scheduler is not None:
            scheduler.step()
            
        # Accumulate metrics
        running_loss += loss.item()
        total_loss += loss.item()
        
        # Update progress bar every step
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}", 
            'CLS': f"{cls_loss.item():.4f}", 
            'CD': f"{cd_loss.item():.4f}"
        })
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for visual_x, audio_wav, labels in pbar:
            visual_x = visual_x.to(device)
            audio_wav = audio_wav.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, audio_latents, visual_features = model(visual_x, audio_wav)
                loss, cls_loss, cd_loss = criterion(logits, labels, audio_latents, visual_features)
                
            total_loss += loss.item()
            
            # Prevent OOM during evaluation loop by aggressively freeing references
            del visual_x, audio_wav, audio_latents, visual_features, loss, cls_loss, cd_loss
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = (all_preds >= 0.5).astype(int)
    
    try:
        acc = accuracy_score(all_labels, pred_classes)
        prec = precision_score(all_labels, pred_classes, zero_division=0)
        rec = recall_score(all_labels, pred_classes, zero_division=0)
        f1 = f1_score(all_labels, pred_classes, zero_division=0)
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        # Failsafe if only one class exists in the batch/split during testing
        acc, prec, rec, f1, auc = 0.0, 0.0, 0.0, 0.0, 0.0
        
    return avg_loss, acc, prec, rec, f1, auc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Configuration for SyncWeldNet and TimeSformer (Optimized for Speed)
    config = {
        'model': {
            'image-size': 160, # Reduced from 224 to massively boost batch speed
            'patch-size': 1,
            'num-classes': 1,
            'num-patches': 100, # 10x10 patches for 160 resolution
            'dim': 512,
            'num-frames': 8, # Cut completely in half to double execution speed
            'max-identities': 2,
            'depth': 9,
            'dim-head': 64,
            'heads': 8,
            'channels': 112, # Output channels of efficientnet block 7
            'attn-dropout': 0.1,
            'ff-dropout': 0.1,
            'shift-tokens': False,
            'enable-size-emb': False, # Avoid tracking sizes for generic training
            'enable-pos-emb': True,
            'enable-identity-attention': False,
            'efficient-net-block': 7
        },
        'training': {
            # Target 6.5 hours total, 10 epochs = 39 mins/epoch.
            # Based on local benchmarking (~0.44s total time per sample), we can fit ~5,200 samples per epoch.
            # We set max_samples to 4500 for train and 700 for val to safely stay under 6.5hrs.
            'max-train-samples': 4500,
            'max-val-samples': 700
        }
    }
    # 1. Initialize Dataset & Dataloader
    print(f"Loading FakeAVCeleb Dataset (capping at {config['training']['max-train-samples']} samples to ensure < 20hr training)...")
    train_dataset = FakeAVCelebDataset(
        metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
        base_path="datasets/FakeAVCeleb_v1.2",
        num_frames=config['model']['num-frames'],
        image_size=config['model']['image-size'],
        target_sample_rate=16000,
        split='train',
        max_samples=config['training']['max-train-samples']
    )
    
    val_dataset = FakeAVCelebDataset(
        metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
        base_path="datasets/FakeAVCeleb_v1.2",
        num_frames=config['model']['num-frames'],
        image_size=config['model']['image-size'],
        target_sample_rate=16000,
        split='val',
        max_samples=config['training']['max-val-samples']
    )
    
    # Aggressive Optimization: Multi-threading and Pinned Memory for fast PCIe transfers
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, # Resolution and frames halved, so physical batch size can quadruple in VRAM!
        shuffle=True, 
        num_workers=4, 
        pin_memory=True # Safely enabled again since RAM usage is cut down
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Initialize Model
    model = SyncWeldNet(timesformer_config=config, num_classes=1).to(device)
    
    # 3. Setup Loss, Optimizer, and Scheduler
    # Calculate pos_weight for BCE loss to handle the severe Fake vs Real class imbalance
    num_fake = sum([1 for row in train_dataset.metadata if row['type'] != 'RealVideo-RealAudio'])
    num_real = len(train_dataset.metadata) - num_fake
    print(f"Class Distribution in Training: {num_real} Real, {num_fake} Fake")
    # BCE pos_weight is num_negative / num_positive
    pos_weight = torch.tensor([num_real / max(1, num_fake)], dtype=torch.float32).to(device)
    
    criterion = ContrastiveDissonanceLoss(alpha=0.5, pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Use standard torch.amp instead of deprecated torch.cuda.amp
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = torch.cuda.amp.GradScaler()
    
    epochs = 10
    steps_per_epoch = len(train_dataloader)
    
    # Implementing OneCycleLR with Warmup to avoid mode collapse
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4, # Lowered from 1e-3 to prevent catastrophic forgetting
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs,
        pct_start=0.2 # 20% of training time spent warming up the learning rate
    )
    
    print(f"SyncWeld-Net training setup is ready.")
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples...")
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, criterion, device, scaler)
        
        # Validation Phase
        val_loss, acc, prec, rec, f1, auc = evaluate(model, val_dataloader, criterion, device)
        
        print(f"Epoch [{epoch}/{epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Acc   : {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Proactive Checkpoint Saving for Laptop Safety
        checkpoint_path = f"syncweld_epoch_{epoch}.pth"
        print(f"Saving checkpoint to {checkpoint_path}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}
        }, checkpoint_path)
        print("Checkpoint saved successfully.")

if __name__ == "__main__":
    main()
