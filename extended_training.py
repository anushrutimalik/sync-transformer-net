"""
Extended Training Framework for SyncWeld-Net - Phase 2
- Extended epochs (up to 50) with Early Stopping
- 10-Fold Stratified Cross-Validation
- Comprehensive metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, List, Tuple, Optional
import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class EarlyStopping:
    """
    Early stopping callback to terminate training when validation loss plateaus.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class CrossValidationFramework:
    """
    10-Fold Stratified Cross-Validation for robust model evaluation.
    Reports Mean ± Standard Deviation across all folds.
    """

    def __init__(
        self, n_splits: int = 10, shuffle: bool = True, random_state: int = 42
    ):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        self.fold_results = []

    def create_folds(
        self, features: np.ndarray, labels: np.ndarray
    ) -> List[Tuple[Subset, Subset]]:
        """Create train/val splits for each fold."""
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.skf.split(features, labels)
        ):
            train_subset = Subset(range(len(labels)), train_idx)
            val_subset = Subset(range(len(labels)), val_idx)
            splits.append((train_subset, val_subset))
        return splits


class MetricsTracker:
    """Tracks and aggregates training metrics for comprehensive reporting."""

    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_auc": [],
            "learning_rates": [],
            "epoch_times": [],
        }

    def update(self, metrics: Dict[str, float], epoch: int):
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Calculate mean and std for all metrics."""
        summary = {}
        for key, values in self.history.items():
            if len(values) > 0 and key != "learning_rates" and key != "epoch_times":
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
        return summary

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 5,
    checkpoint_dir: str = "./checkpoints",
    experiment_name: str = "syncweld",
) -> Tuple[nn.Module, MetricsTracker]:
    """
    Train model with early stopping based on validation loss.
    Returns best model and training history.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    tracker = MetricsTracker()
    early_stopping = EarlyStopping(patience=patience, mode="min")
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    best_model_state = None
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
        for visual_x, audio_wav, labels in pbar:
            visual_x = visual_x.to(device, non_blocking=True)
            audio_wav = audio_wav.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, audio_latents, visual_features = model(visual_x, audio_wav)
                loss, cls_loss, cd_loss = criterion(
                    logits, labels, audio_latents, visual_features
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if scheduler:
                scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "CLS": f"{cls_loss.item():.4f}",
                    "CD": f"{cd_loss.item():.4f}",
                }
            )

        train_loss /= len(train_loader)

        val_loss, acc, prec, rec, f1, auc = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_auc": auc,
            "learning_rates": optimizer.param_groups[0]["lr"],
            "epoch_times": epoch_time,
        }
        tracker.update(metrics, epoch)

        print(f"\nEpoch {epoch}/{max_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"  Val Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
        )
        print(
            f"  Epoch Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "val_loss": val_loss,
                    "val_metrics": {
                        "acc": acc,
                        "prec": prec,
                        "rec": rec,
                        "f1": f1,
                        "auc": auc,
                    },
                },
                os.path.join(checkpoint_dir, f"{experiment_name}_best.pth"),
            )
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        if early_stopping(val_loss, epoch):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(
                f"Best epoch was {early_stopping.best_epoch} with val_loss={early_stopping.best_score:.4f}"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, tracker


def train_with_kfold_cv(
    model_class: type,
    model_config: Dict,
    dataset: torch.utils.data.Dataset,
    criterion: nn.Module,
    device: torch.device,
    n_folds: int = 10,
    max_epochs: int = 50,
    patience: int = 5,
    checkpoint_dir: str = "./kfold_checkpoints",
    experiment_name: str = "syncweld_kfold",
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation training.
    Returns per-fold and aggregated metrics.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    cv_framework = CrossValidationFramework(n_splits=n_folds)

    all_labels = np.array([dataset[i][2].item() for i in range(len(dataset))])
    all_indices = np.arange(len(dataset))

    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": [],
        "val_loss": [],
        "train_time": [],
    }

    for fold, (train_idx, val_idx) in enumerate(
        cv_framework.skf.split(all_indices, all_labels)
    ):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True
        )

        model = model_class(**model_config).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            steps_per_epoch=len(train_loader),
            epochs=max_epochs,
            pct_start=0.2,
        )

        fold_criterion = type(criterion)(**criterion.__dict__)

        trained_model, tracker = train_with_early_stopping(
            model,
            train_loader,
            val_loader,
            fold_criterion,
            optimizer,
            scheduler,
            device,
            max_epochs=max_epochs,
            patience=patience,
            checkpoint_dir=os.path.join(checkpoint_dir, f"fold_{fold}"),
            experiment_name=f"{experiment_name}_fold{fold}",
        )

        _, acc, prec, rec, f1, auc = evaluate(
            trained_model, val_loader, fold_criterion, device
        )

        summary = tracker.get_summary()

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)
        fold_metrics["auc"].append(auc)
        fold_metrics["val_loss"].append(summary["val_loss"]["mean"])

        total_time = sum(tracker.history.get("epoch_times", []))
        fold_metrics["train_time"].append(total_time)

        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Train Time: {total_time:.1f}s")

    aggregated = {}
    for key, values in fold_metrics.items():
        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)

    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"Accuracy:  {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}"
    )
    print(
        f"Precision: {aggregated['precision_mean']:.4f} ± {aggregated['precision_std']:.4f}"
    )
    print(
        f"Recall:    {aggregated['recall_mean']:.4f} ± {aggregated['recall_std']:.4f}"
    )
    print(f"F1-Score:  {aggregated['f1_mean']:.4f} ± {aggregated['f1_std']:.4f}")
    print(f"AUC:       {aggregated['auc_mean']:.4f} ± {aggregated['auc_std']:.4f}")
    print(
        f"Train Time: {aggregated['train_time_mean']:.1f}s ± {aggregated['train_time_std']:.1f}s"
    )

    with open(os.path.join(checkpoint_dir, "kfold_results.json"), "w") as f:
        json.dump({"fold_metrics": fold_metrics, "aggregated": aggregated}, f, indent=2)

    return fold_metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for visual_x, audio_wav, labels in dataloader:
            visual_x = visual_x.to(device)
            audio_wav = audio_wav.to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, audio_latents, visual_features = model(visual_x, audio_wav)
                loss, _, _ = criterion(logits, labels, audio_latents, visual_features)

            total_loss += loss.item()

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
        acc, prec, rec, f1, auc = 0.0, 0.0, 0.0, 0.0, 0.0

    return avg_loss, acc, prec, rec, f1, auc


if __name__ == "__main__":
    print("Extended Training Framework loaded successfully")
    print("Usage:")
    print(
        "  from extended_training import train_with_early_stopping, train_with_kfold_cv"
    )
