"""
Comparison Runner: Train baseline models and test on multiple datasets
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

from models.syncweld import SyncWeldNet
from models.xception import xception
from models.size_invariant_timesformer import SizeInvariantTIMEsformer
from segmented_dataset import SegmentedFakeAVCelebDataset
from deepfakes_dataset import DeepFakesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_fakeavceleb_dataset(split="train", max_samples=2000, ensure_balance=False):
    """Get FakeAVCeleb dataset"""
    return SegmentedFakeAVCelebDataset(
        metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
        base_path="datasets/FakeAVCeleb_v1.2",
        num_frames=8,
        image_size=160,
        segment_duration=4.0,
        max_segments_per_video=2,
        split=split,
        max_samples=max_samples,
        ensure_balance=ensure_balance,
    )


def get_faceforensics_dataset(split="train", max_samples=2000):
    """Get FaceForensics++ dataset"""
    return DeepFakesDataset(
        base_path="datasets/FakeForensics++_C23",
        split=split,
        max_samples=max_samples,
        num_frames=8,
        image_size=160,
    )


def create_model(model_name, config):
    """Create model by name"""
    if model_name == "syncweld":
        return SyncWeldNet(timesformer_config=config, num_classes=1).to(device)
    elif model_name == "xception":
        return xception(num_classes=1).to(device)
    elif model_name == "timesformer":
        return SizeInvariantTIMEsformer(config).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model, train_loader, val_loader, epochs=30, model_name="model"):
    """Train a model and return best metrics"""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=epochs
    )

    best_f1 = 0
    best_metrics = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for visual_x, audio_wav, labels in tqdm(
            train_loader, desc=f"{model_name} Epoch {epoch}"
        ):
            visual_x = visual_x.to(device)
            audio_wav = audio_wav.to(device) if audio_wav is not None else None
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if model_name == "syncweld":
                    logits, _, _ = model(visual_x, audio_wav)
                else:
                    logits = model(visual_x).squeeze(1)
                loss = criterion(logits, labels.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for visual_x, audio_wav, labels in val_loader:
                visual_x = visual_x.to(device)
                audio_wav = audio_wav.to(device) if audio_wav is not None else None

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if model_name == "syncweld":
                        logits, _, _ = model(visual_x, audio_wav)
                    else:
                        logits = model(visual_x).squeeze(1)

                preds = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_classes = (all_preds >= 0.5).astype(int)

        acc = accuracy_score(all_labels, pred_classes)
        prec = precision_score(all_labels, pred_classes, zero_division=0)
        rec = recall_score(all_labels, pred_classes, zero_division=0)
        f1 = f1_score(all_labels, pred_classes, zero_division=0)
        auc = roc_auc_score(all_labels, all_preds)

        print(f"Epoch {epoch}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

    return best_metrics


def run_comparison():
    """Run full comparison"""

    print("=" * 70)
    print("MODEL & DATASET COMPARISON")
    print("=" * 70)

    results = {}

    # Config for models
    config = {
        "model": {
            "image-size": 160,
            "patch-size": 1,
            "num-classes": 1,
            "num-patches": 100,
            "dim": 512,
            "num-frames": 8,
            "max-identities": 2,
            "depth": 9,
            "dim-head": 64,
            "heads": 8,
            "channels": 112,
            "attn-dropout": 0.1,
            "ff-dropout": 0.1,
        }
    }

    # =========================================================================
    # PART 1: Train different models on FakeAVCeleb
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Different Models on FakeAVCeleb")
    print("=" * 70)

    # Get datasets
    train_ds = get_fakeavceleb_dataset("train", max_samples=2000, ensure_balance=False)
    val_ds = get_fakeavceleb_dataset("val", max_samples=500, ensure_balance=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    print(f"Dataset: FakeAVCeleb - Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Train SyncWeldNet (your model)
    print("\n--- Training SyncWeldNet (MINTIME) ---")
    model = create_model("syncweld", config)
    results["SyncWeldNet_FakeAVCeleb"] = train_model(
        model, train_loader, val_loader, epochs=20, model_name="SyncWeldNet"
    )

    # Train Xception
    print("\n--- Training Xception ---")
    model = create_model("xception", config)
    results["Xception_FakeAVCeleb"] = train_model(
        model, train_loader, val_loader, epochs=20, model_name="Xception"
    )

    # =========================================================================
    # PART 2: Train SyncWeldNet on different datasets
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: SyncWeldNet on Different Datasets")
    print("=" * 70)

    # FaceForensics++
    print("\n--- Loading FaceForensics++ ---")
    try:
        train_ff = get_faceforensics_dataset("train", max_samples=1000)
        val_ff = get_faceforensics_dataset("val", max_samples=300)

        train_loader_ff = DataLoader(
            train_ff, batch_size=4, shuffle=True, num_workers=2
        )
        val_loader_ff = DataLoader(val_ff, batch_size=4, shuffle=False, num_workers=2)

        print(f"Dataset: FaceForensics++ - Train: {len(train_ff)}, Val: {len(val_ff)}")

        print("\n--- Training SyncWeldNet on FaceForensics++ ---")
        model = create_model("syncweld", config)
        results["SyncWeldNet_FaceForensics"] = train_model(
            model,
            train_loader_ff,
            val_loader_ff,
            epochs=20,
            model_name="SyncWeldNet_FF",
        )
    except Exception as e:
        print(f"FaceForensics++ not available: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    for key, metrics in results.items():
        print(f"\n{key}:")
        print(f"  Accuracy:  {metrics['acc']:.4f}")
        print(f"  Precision: {metrics['prec']:.4f}")
        print(f"  Recall:    {metrics['rec']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

    # Save results
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to comparison_results.json")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
