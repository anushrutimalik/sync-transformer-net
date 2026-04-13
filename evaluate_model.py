"""
Evaluate trained model on different datasets for comparison
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from models.syncweld import SyncWeldNet
from segmented_dataset import SegmentedFakeAVCelebDataset
from deepfakes_dataset import DeepFakesDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_fakeavceleb_dataset(split="val", max_samples=2000):
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
        ensure_balance=False,
    )


def evaluate_model(model, loader, model_name="model"):
    """Evaluate model on dataset"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for visual_x, audio_wav, labels in tqdm(
            loader, desc=f"Evaluating {model_name}"
        ):
            visual_x = visual_x.to(device)
            audio_wav = audio_wav.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _, _ = model(visual_x, audio_wav)

            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_classes = (all_preds >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, pred_classes),
        "precision": precision_score(all_labels, pred_classes, zero_division=0),
        "recall": recall_score(all_labels, pred_classes, zero_division=0),
        "f1": f1_score(all_labels, pred_classes, zero_division=0),
        "auc": roc_auc_score(all_labels, all_preds),
    }

    cm = confusion_matrix(all_labels, pred_classes)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


def run_evaluation():
    """Run evaluation on multiple datasets"""

    print("=" * 70)
    print("MODEL EVALUATION ON MULTIPLE DATASETS")
    print("=" * 70)

    results = {}

    # Load trained model
    checkpoint_path = "phase1_checkpoints/best_model.pth"

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
            "shift-tokens": False,
            "enable-size-emb": False,
            "enable-pos-emb": True,
            "enable-identity-attention": False,
            "efficient-net-block": 7,
        }
    }

    print("\nLoading model...")
    model = SyncWeldNet(timesformer_config=config, num_classes=1).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"WARNING: {checkpoint_path} not found!")
        return

    # =========================================================================
    # Evaluate on FakeAVCeleb (train split)
    # =========================================================================
    print("\n" + "=" * 70)
    print("FakeAVCeleb Dataset (Train Split)")
    print("=" * 70)

    train_ds = get_fakeavceleb_dataset("train", max_samples=1000)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=2)

    print(f"Dataset size: {len(train_ds)}")
    metrics = evaluate_model(model, train_loader, "FakeAVCeleb_Train")
    results["FakeAVCeleb_Train"] = metrics

    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={metrics['tn']}, FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}, TP={metrics['tp']}")

    # =========================================================================
    # Evaluate on FakeAVCeleb (val split)
    # =========================================================================
    print("\n" + "=" * 70)
    print("FakeAVCeleb Dataset (Validation Split)")
    print("=" * 70)

    val_ds = get_fakeavceleb_dataset("val", max_samples=1000)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    print(f"Dataset size: {len(val_ds)}")
    metrics = evaluate_model(model, val_loader, "FakeAVCeleb_Val")
    results["FakeAVCeleb_Val"] = metrics

    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")

    # =========================================================================
    # Evaluate on FaceForensics++
    # =========================================================================
    print("\n" + "=" * 70)
    print("FaceForensics++ Dataset")
    print("=" * 70)

    try:
        ff_ds = DeepFakesDataset(
            base_path="datasets/FakeForensics++_C23",
            split="val",
            max_samples=500,
            num_frames=8,
            image_size=160,
        )
        ff_loader = DataLoader(ff_ds, batch_size=4, shuffle=False, num_workers=2)

        print(f"Dataset size: {len(ff_ds)}")
        metrics = evaluate_model(model, ff_loader, "FaceForensics++")
        results["FaceForensics++"] = metrics

        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
    except Exception as e:
        print(f"FaceForensics++ evaluation failed: {e}")

    # =========================================================================
    # Save results
    # =========================================================================
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to evaluation_results.json")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
