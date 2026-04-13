"""
Master Training Pipeline: Complete Phase 1-4 Implementation
Generates comprehensive comparison matrix for journal paper.

Usage:
    python master_pipeline.py --mode full
    python master_pipeline.py --mode ablation
    python master_pipeline.py --mode kfold
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from models.syncweld import SyncWeldNet
from fakeavceleb_dataset import FakeAVCelebDataset
from segmented_dataset import SegmentedFakeAVCelebDataset
from extended_training import (
    train_with_early_stopping,
    train_with_kfold_cv,
    EarlyStopping,
    MetricsTracker,
)
from baseline_models import (
    VisualOnlyModel,
    AudioOnlyModel,
    FusionHeadEvaluator,
    extract_multimodal_features,
)
from ablation_study import (
    AblationStudy,
    MixedSignalAnalyzer,
    ContrastiveDissonanceLoss,
    StandardBCELoss,
)


def generate_paper_comparison_matrix(results: dict, output_dir: str):
    """Generate the final comparison matrix for the paper."""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    header_color = "#2E4057"
    alt_row_color = "#F5F5F5"

    table_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Train Time (s)"],
        [
            "Audio-Only (Wav2Vec2)",
            f"{results.get('audio_only', {}).get('accuracy', 0):.4f}",
            f"{results.get('audio_only', {}).get('precision', 0):.4f}",
            f"{results.get('audio_only', {}).get('recall', 0):.4f}",
            f"{results.get('audio_only', {}).get('f1', 0):.4f}",
            f"{results.get('audio_only', {}).get('train_time', 0):.1f}",
        ],
        [
            "Visual-Only (TimeSformer)",
            f"{results.get('visual_only', {}).get('accuracy', 0):.4f}",
            f"{results.get('visual_only', {}).get('precision', 0):.4f}",
            f"{results.get('visual_only', {}).get('recall', 0):.4f}",
            f"{results.get('visual_only', {}).get('f1', 0):.4f}",
            f"{results.get('visual_only', {}).get('train_time', 0):.1f}",
        ],
        [
            "SyncWeld + SVM Head",
            f"{results.get('syncweld_svm', {}).get('accuracy', 0):.4f}",
            f"{results.get('syncweld_svm', {}).get('precision', 0):.4f}",
            f"{results.get('syncweld_svm', {}).get('recall', 0):.4f}",
            f"{results.get('syncweld_svm', {}).get('f1', 0):.4f}",
            "~0 (frozen)",
        ],
        [
            "SyncWeld + ELM Head",
            f"{results.get('syncweld_elm', {}).get('accuracy', 0):.4f}",
            f"{results.get('syncweld_elm', {}).get('precision', 0):.4f}",
            f"{results.get('syncweld_elm', {}).get('recall', 0):.4f}",
            f"{results.get('syncweld_elm', {}).get('f1', 0):.4f}",
            "~0 (frozen)",
        ],
        [
            "SyncWeld-Net (Full)",
            f"{results.get('syncweld_full', {}).get('accuracy', 0):.4f}",
            f"{results.get('syncweld_full', {}).get('precision', 0):.4f}",
            f"{results.get('syncweld_full', {}).get('recall', 0):.4f}",
            f"{results.get('syncweld_full', {}).get('f1', 0):.4f}",
            f"{results.get('syncweld_full', {}).get('train_time', 0):.1f}",
        ],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.5)

    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor(header_color)
        table[(0, j)].set_text_props(color="white", weight="bold")

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(alt_row_color)

    for j in range(len(table_data[0])):
        if j == 0:
            table[(5, j)].set_facecolor("#90EE90")

    plt.title(
        "Table 5: Comprehensive Model Comparison\nSyncWeld-Net vs Baseline Methods",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "paper_table5_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    latex_table = """
\\begin{table}[h]
\\centering
\\caption{Comprehensive Model Comparison}
\\label{tab:comparison}
\\begin{tabular}{|l|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Train Time (s)} \\\\ \\hline
Audio-Only (Wav2Vec2) & ${:.4f}$ & ${:.4f}$ & ${:.4f}$ & ${:.4f}$ & ${:.1f}$ \\\\ \\hline
Visual-Only (TimeSformer) & ${:.4f}$ & ${:.4f}$ & ${:.4f}$ & ${:.4f}$ & ${:.1f}$ \\\\ \\hline
SyncWeld + SVM Head & ${:.4f}$ & ${:.4f}$ & ${:.4f}$ & ${:.4f}$ & frozen \\\\ \\hline
SyncWeld + ELM Head & ${:.4f}$ & ${:.4f}$ & ${:.4f}$ & ${:.4f}$ & frozen \\\\ \\hline
\\textbf{SyncWeld-Net (Full)} & \\textbf{${:.4f}$} & \\textbf{${:.4f}$} & \\textbf{${:.4f}$} & \\textbf{${:.4f}$} & ${:.1f}$ \\\\ \\hline
\\end{tabular}
\\end{table}
""".format(
        results.get("audio_only", {}).get("accuracy", 0),
        results.get("audio_only", {}).get("precision", 0),
        results.get("audio_only", {}).get("recall", 0),
        results.get("audio_only", {}).get("f1", 0),
        results.get("audio_only", {}).get("train_time", 0),
        results.get("visual_only", {}).get("accuracy", 0),
        results.get("visual_only", {}).get("precision", 0),
        results.get("visual_only", {}).get("recall", 0),
        results.get("visual_only", {}).get("f1", 0),
        results.get("visual_only", {}).get("train_time", 0),
        results.get("syncweld_svm", {}).get("accuracy", 0),
        results.get("syncweld_svm", {}).get("precision", 0),
        results.get("syncweld_svm", {}).get("recall", 0),
        results.get("syncweld_svm", {}).get("f1", 0),
        results.get("syncweld_elm", {}).get("accuracy", 0),
        results.get("syncweld_elm", {}).get("precision", 0),
        results.get("syncweld_elm", {}).get("recall", 0),
        results.get("syncweld_elm", {}).get("f1", 0),
        results.get("syncweld_full", {}).get("accuracy", 0),
        results.get("syncweld_full", {}).get("precision", 0),
        results.get("syncweld_full", {}).get("recall", 0),
        results.get("syncweld_full", {}).get("f1", 0),
        results.get("syncweld_full", {}).get("train_time", 0),
    )

    with open(os.path.join(output_dir, "paper_table5_latex.txt"), "w") as f:
        f.write(latex_table)

    return table_data


def run_full_pipeline(args):
    """Run complete Phase 1-4 pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "phase2_checkpoint.json")
    completed_models = set()
    saved_results = {}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                completed_models = set(checkpoint_data.get("completed_models", []))
                if "audio_results" in checkpoint_data:
                    saved_results["audio_only"] = checkpoint_data["audio_results"]
                print(
                    f"\n>>> RESUMING: Found checkpoint. Already completed: {completed_models}"
                )
        except:
            pass

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

    print("\n" + "=" * 70)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 70)

    if args.use_segmented:
        print("Using SegmentedDataset (3-5s clips)...")
        train_dataset = SegmentedFakeAVCelebDataset(
            metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
            base_path="datasets/FakeAVCeleb_v1.2",
            num_frames=8,
            image_size=160,
            segment_duration=4.0,
            split="train",
            max_samples=args.max_train_samples,
            ensure_balance=True,
        )
        val_dataset = SegmentedFakeAVCelebDataset(
            metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
            base_path="datasets/FakeAVCeleb_v1.2",
            num_frames=8,
            image_size=160,
            segment_duration=4.0,
            split="val",
            max_samples=args.max_val_samples,
            ensure_balance=True,
        )
    else:
        print("Using standard FakeAVCelebDataset...")
        train_dataset = FakeAVCelebDataset(
            metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
            base_path="datasets/FakeAVCeleb_v1.2",
            num_frames=8,
            image_size=160,
            split="train",
            max_samples=args.max_train_samples,
        )
        val_dataset = FakeAVCelebDataset(
            metadata_csv_path="datasets/FakeAVCeleb_v1.2/meta_data.csv",
            base_path="datasets/FakeAVCeleb_v1.2",
            num_frames=8,
            image_size=160,
            split="val",
            max_samples=args.max_val_samples,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    all_results = {}

    print("\n" + "=" * 70)
    print("PHASE 3: BASELINE MODELS")
    print("=" * 70)

    print("\n--- Training Audio-Only Baseline ---")
    audio_model = AudioOnlyModel(num_classes=1).to(device)
    audio_optimizer = optim.AdamW(audio_model.parameters(), lr=1e-4)
    audio_scheduler = optim.lr_scheduler.OneCycleLR(
        audio_optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=8,
        pct_start=0.2,
    )
    audio_criterion = nn.BCEWithLogitsLoss()

    if "audio_only" in completed_models:
        print(">>> Skipping Audio-Only (already completed)")
        if "audio_only" in saved_results:
            all_results["audio_only"] = saved_results["audio_only"]
        else:
            raise ValueError("Audio results missing from checkpoint")
    else:
        audio_model, audio_tracker = train_baseline_model(
            audio_model,
            train_loader,
            val_loader,
            audio_criterion,
            audio_optimizer,
            audio_scheduler,
            device,
            is_audio_only=True,
        )
        completed_models.add("audio_only")
        all_results["audio_only"] = {
            "accuracy": audio_tracker.history["val_accuracy"][-1],
            "precision": audio_tracker.history["val_precision"][-1],
            "recall": audio_tracker.history["val_recall"][-1],
            "f1": audio_tracker.history["val_f1"][-1],
            "auc": audio_tracker.history["val_auc"][-1],
            "train_time": sum(audio_tracker.history.get("epoch_times", [])),
        }
    with open(checkpoint_path, "w") as f:
        json.dump(
            {
                "completed_models": list(completed_models),
                "audio_results": all_results["audio_only"],
            },
            f,
        )
    print(f">>> Saved checkpoint: audio_only done")

    print("\n--- Training Visual-Only Baseline ---")
    visual_model = VisualOnlyModel(timesformer_config=config, num_classes=1).to(device)
    visual_optimizer = optim.AdamW(visual_model.parameters(), lr=1e-4)
    visual_scheduler = optim.lr_scheduler.OneCycleLR(
        visual_optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=8,
        pct_start=0.2,
    )
    visual_criterion = nn.BCEWithLogitsLoss()

    visual_model, visual_tracker = train_baseline_model(
        visual_model,
        train_loader,
        val_loader,
        visual_criterion,
        visual_optimizer,
        visual_scheduler,
        device,
        is_audio_only=False,
    )

    all_results["visual_only"] = {
        "accuracy": visual_tracker.history["val_accuracy"][-1],
        "precision": visual_tracker.history["val_precision"][-1],
        "recall": visual_tracker.history["val_recall"][-1],
        "f1": visual_tracker.history["val_f1"][-1],
        "auc": visual_tracker.history["val_auc"][-1],
        "train_time": sum(visual_tracker.history.get("epoch_times", [])),
    }

    print("\n" + "=" * 70)
    print("PHASE 2: SYNCWELD-NET TRAINING (Extended + Early Stopping)")
    print("=" * 70)

    full_model = SyncWeldNet(timesformer_config=config, num_classes=1).to(device)

    num_fake = sum(
        1 for i in range(len(train_dataset)) if train_dataset[i][2].item() == 1
    )
    num_real = len(train_dataset) - num_fake
    pos_weight = torch.tensor([num_real / max(1, num_fake)], dtype=torch.float32).to(
        device
    )

    full_criterion = ContrastiveDissonanceLoss(alpha=0.5, pos_weight=pos_weight)
    full_optimizer = optim.AdamW(full_model.parameters(), lr=1e-4, weight_decay=1e-5)

    full_model, full_tracker = train_with_early_stopping(
        full_model,
        train_loader,
        val_loader,
        full_criterion,
        full_optimizer,
        None,
        device,
        max_epochs=50,
        patience=5,
        checkpoint_dir=output_dir,
        experiment_name="syncweld_full",
    )

    all_results["syncweld_full"] = {
        "accuracy": full_tracker.history["val_accuracy"][-1],
        "precision": full_tracker.history["val_precision"][-1],
        "recall": full_tracker.history["val_recall"][-1],
        "f1": full_tracker.history["val_f1"][-1],
        "auc": full_tracker.history["val_auc"][-1],
        "train_time": sum(full_tracker.history.get("epoch_times", [])),
    }

    print("\n" + "=" * 70)
    print("ALTERNATIVE FUSION HEADS (SVM, ELM)")
    print("=" * 70)

    print("Extracting multimodal features...")
    features, labels = extract_multimodal_features(full_model, train_loader, device)

    evaluator = FusionHeadEvaluator(feature_dim=features.shape[1])

    print("Training SVM Head...")
    svm_metrics = evaluator.train_svm(features, labels)
    all_results["syncweld_svm"] = {**svm_metrics, "train_time": 0}

    print("Training ELM Head...")
    elm_metrics = evaluator.train_elm(
        torch.from_numpy(features).float(), torch.from_numpy(labels).float()
    )
    all_results["syncweld_elm"] = {**elm_metrics, "train_time": 0}

    print("\n" + "=" * 70)
    print("GENERATING PAPER COMPARISON MATRIX")
    print("=" * 70)

    generate_paper_comparison_matrix(all_results, output_dir)

    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to:", output_dir)
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for model_name, metrics in all_results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")


def train_baseline_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    is_audio_only=False,
    epochs=8,
):
    """Train a baseline model."""
    tracker = MetricsTracker()
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for visual_x, audio_wav, labels in pbar:
            visual_x = visual_x.to(device, non_blocking=True)
            audio_wav = audio_wav.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if is_audio_only:
                    logits, _ = model(audio_wav)
                else:
                    logits, _ = model(visual_x)
                loss = criterion(logits, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for visual_x, audio_wav, labels in val_loader:
                if is_audio_only:
                    vx, lb = audio_wav.to(device), labels.to(device).unsqueeze(1)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        lgts, _ = model(vx)
                else:
                    vx, lb = visual_x.to(device), labels.to(device).unsqueeze(1)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        lgts, _ = model(vx)

                loss = criterion(lgts, lb)
                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(lgts).cpu().numpy())
                all_labels.extend(lb.cpu().numpy())

        val_loss = total_loss / len(val_loader)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_classes = (all_preds >= 0.5).astype(int)

        acc = accuracy_score(all_labels, pred_classes)
        prec = precision_score(all_labels, pred_classes, zero_division=0)
        rec = recall_score(all_labels, pred_classes, zero_division=0)
        f1 = f1_score(all_labels, pred_classes, zero_division=0)
        auc = roc_auc_score(all_labels, all_preds)

        epoch_time = time.time() - epoch_start

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_auc": auc,
            "epoch_times": epoch_time,
        }
        tracker.update(metrics, epoch)

        print(
            f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}"
        )

    return model, tracker


if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser(description="Master Training Pipeline")
    parser.add_argument(
        "--mode", type=str, default="full", choices=["full", "ablation", "kfold"]
    )
    parser.add_argument("--output_dir", type=str, default="./experiment_results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--max_val_samples", type=int, default=200)
    parser.add_argument(
        "--use_segmented", action="store_true", help="Use segmented dataset"
    )

    args = parser.parse_args()

    run_full_pipeline(args)
