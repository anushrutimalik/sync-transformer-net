"""
Ablation Study Framework - Phase 4
Quantifies the impact of Contrastive Dissonance Loss
- Control: BCE only (no CD Loss)
- Experimental: BCE + Contrastive Dissonance Loss
- Mixed-Signal Analysis: Videos that look real but have audio-visual desynchronization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class StandardBCELoss(nn.Module):
    """
    Control: Standard Binary Cross-Entropy without Contrastive Dissonance.
    Used to isolate the impact of the CD Loss component.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels, audio_latents=None, visual_features=None):
        cls_loss = self.bce(logits, labels.float())
        return cls_loss, cls_loss, torch.tensor(0.0)


class ContrastiveDissonanceLoss(nn.Module):
    """
    Experimental: BCE + Contrastive Dissonance Loss.
    Penalizes audio-visual desynchronization (energy vs velocity mismatch).
    """

    def __init__(self, alpha=0.5, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels, audio_latents, visual_features):
        cls_loss = self.bce(logits, labels.float())

        audio_energy = torch.norm(audio_latents, dim=-1).mean(dim=1)

        if len(visual_features.shape) == 3 and visual_features.shape[1] > 1:
            visual_velocity = torch.norm(
                visual_features[:, 1:, :] - visual_features[:, :-1, :], dim=-1
            ).mean(dim=1)
        else:
            visual_velocity = torch.norm(visual_features, dim=-1).mean(dim=1)

        norm_audio = audio_energy / (audio_energy.max() + 1e-8)
        norm_visual = visual_velocity / (visual_velocity.max() + 1e-8)

        dissonance_loss = torch.mean(torch.abs(norm_audio - norm_visual))

        total_loss = cls_loss + self.alpha * dissonance_loss
        return total_loss, cls_loss, dissonance_loss


class AblationStudy:
    """
    Conducts ablation study comparing Control vs Experimental configurations.
    """

    def __init__(
        self,
        model_class,
        model_config: Dict,
        train_dataset,
        val_dataset,
        device: torch.device,
        experiment_dir: str = "./ablation_study",
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

        self.results = {}

    def run_ablation(
        self, max_epochs: int = 20, patience: int = 5, batch_size: int = 4
    ) -> Dict:
        """
        Run complete ablation study comparing both configurations.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        print("\n" + "=" * 70)
        print("ABLATION STUDY: Contrastive Dissonance Loss Impact")
        print("=" * 70)

        num_fake = sum(
            1
            for i in range(len(self.train_dataset))
            if self.train_dataset[i][2].item() == 1
        )
        num_real = len(self.train_dataset) - num_fake
        pos_weight = torch.tensor(
            [num_real / max(1, num_fake)], dtype=torch.float32
        ).to(self.device)

        print("\n--- Configuration 1: BCE Only (Control) ---")
        control_results = self._train_configuration(
            criterion_class=StandardBCELoss,
            criterion_kwargs={"pos_weight": pos_weight},
            train_loader=train_loader,
            val_loader=val_loader,
            config_name="bce_only",
            max_epochs=max_epochs,
            patience=patience,
        )

        print("\n--- Configuration 2: BCE + CD Loss (Experimental) ---")
        experimental_results = self._train_configuration(
            criterion_class=ContrastiveDissonanceLoss,
            criterion_kwargs={"alpha": 0.5, "pos_weight": pos_weight},
            train_loader=train_loader,
            val_loader=val_loader,
            config_name="bce_cd_loss",
            max_epochs=max_epochs,
            patience=patience,
        )

        self.results = {
            "control": control_results,
            "experimental": experimental_results,
        }

        self._generate_ablation_report()
        self._generate_comparison_table()

        return self.results

    def _train_configuration(
        self,
        criterion_class,
        criterion_kwargs,
        train_loader,
        val_loader,
        config_name: str,
        max_epochs: int,
        patience: int,
    ) -> Dict:
        """Train a single configuration."""
        model = self.model_class(**self.model_config).to(self.device)

        criterion = criterion_class(**criterion_kwargs)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_auc": [],
            "cd_loss": [],
            "cls_loss": [],
        }

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0

            pbar = tqdm(
                train_loader, desc=f"[{config_name}] Epoch {epoch}/{max_epochs}"
            )
            for visual_x, audio_wav, labels in pbar:
                visual_x = visual_x.to(self.device, non_blocking=True)
                audio_wav = audio_wav.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).unsqueeze(1)

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

                train_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            train_loss /= len(train_loader)

            val_loss, acc, prec, rec, f1, auc = self._evaluate(
                model, val_loader, criterion
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(acc)
            history["val_f1"].append(f1)
            history["val_auc"].append(auc)
            history["cd_loss"].append(
                cd_loss.item() if isinstance(cd_loss, torch.Tensor) else cd_loss
            )
            history["cls_loss"].append(
                cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
            )

            print(
                f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)

        final_metrics = self._evaluate(model, val_loader, criterion)

        torch.save(
            {
                "model_state_dict": best_model_state,
                "metrics": {
                    "val_loss": final_metrics[0],
                    "val_accuracy": final_metrics[1],
                    "val_f1": final_metrics[4],
                },
                "history": history,
            },
            os.path.join(self.experiment_dir, f"{config_name}_best.pth"),
        )

        return {
            "best_val_loss": best_val_loss,
            "final_metrics": {
                "val_loss": final_metrics[0],
                "accuracy": final_metrics[1],
                "precision": final_metrics[2],
                "recall": final_metrics[3],
                "f1": final_metrics[4],
                "auc": final_metrics[5],
            },
            "history": history,
            "epochs_trained": len(history["train_loss"]),
        }

    def _evaluate(self, model, dataloader, criterion):
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for visual_x, audio_wav, labels in dataloader:
                visual_x = visual_x.to(self.device)
                audio_wav = audio_wav.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, audio_latents, visual_features = model(visual_x, audio_wav)
                    loss, _, _ = criterion(
                        logits, labels, audio_latents, visual_features
                    )

                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
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

    def _generate_ablation_report(self):
        """Generate detailed ablation study report."""
        control = self.results["control"]
        experimental = self.results["experimental"]

        report = {
            "study_description": "Ablation Study: Impact of Contrastive Dissonance Loss",
            "control": {
                "description": "Standard BCE Loss only",
                "metrics": control["final_metrics"],
                "epochs": control["epochs_trained"],
                "best_val_loss": control["best_val_loss"],
            },
            "experimental": {
                "description": "BCE Loss + Contrastive Dissonance Loss (alpha=0.5)",
                "metrics": experimental["final_metrics"],
                "epochs": experimental["epochs_trained"],
                "best_val_loss": experimental["best_val_loss"],
            },
            "improvement": {},
        }

        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            ctrl_val = control["final_metrics"][metric]
            exp_val = experimental["final_metrics"][metric]
            improvement = exp_val - ctrl_val
            relative_improvement = (improvement / ctrl_val * 100) if ctrl_val > 0 else 0
            report["improvement"][metric] = {
                "absolute": improvement,
                "relative_percent": relative_improvement,
            }

        with open(os.path.join(self.experiment_dir, "ablation_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 70)
        print("ABLATION STUDY RESULTS")
        print("=" * 70)
        print(f"\n{'Metric':<15} {'BCE Only':<15} {'BCE + CD':<15} {'Improvement':<15}")
        print("-" * 60)
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            ctrl = control["final_metrics"][metric]
            exp = experimental["final_metrics"][metric]
            imp = report["improvement"][metric]["relative_percent"]
            print(f"{metric:<15} {ctrl:.4f}          {exp:.4f}          +{imp:.2f}%")

        return report

    def _generate_comparison_table(self):
        """Generate visual comparison table."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        control = self.results["control"]["final_metrics"]
        experimental = self.results["experimental"]["final_metrics"]

        table_data = [
            ["Metric", "BCE Only (Control)", "BCE + CD (Experimental)", "Delta"],
            [
                "Accuracy",
                f"{control['accuracy']:.4f}",
                f"{experimental['accuracy']:.4f}",
                f"{experimental['accuracy'] - control['accuracy']:+.4f}",
            ],
            [
                "Precision",
                f"{control['precision']:.4f}",
                f"{experimental['precision']:.4f}",
                f"{experimental['precision'] - control['precision']:+.4f}",
            ],
            [
                "Recall",
                f"{control['recall']:.4f}",
                f"{experimental['recall']:.4f}",
                f"{experimental['recall'] - control['recall']:+.4f}",
            ],
            [
                "F1-Score",
                f"{control['f1']:.4f}",
                f"{experimental['f1']:.4f}",
                f"{experimental['f1'] - control['f1']:+.4f}",
            ],
            [
                "AUC",
                f"{control['auc']:.4f}",
                f"{experimental['auc']:.4f}",
                f"{experimental['auc'] - control['auc']:+.4f}",
            ],
        ]

        table = ax.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)

        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(color="white", weight="bold")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_dir, "ablation_table.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


class MixedSignalAnalyzer:
    """
    Analyzes "mixed signal" cases where video appears real but audio-visual sync is abnormal.
    These are the hardest cases that CD Loss specifically targets.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def compute_dissonance_scores(
        self, audio_latents: torch.Tensor, visual_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute audio-visual dissonance scores for samples."""
        audio_energy = torch.norm(audio_latents, dim=-1).mean(dim=1)

        if len(visual_features.shape) == 3 and visual_features.shape[1] > 1:
            visual_velocity = torch.norm(
                visual_features[:, 1:, :] - visual_features[:, :-1, :], dim=-1
            ).mean(dim=1)
        else:
            visual_velocity = torch.norm(visual_features, dim=-1).mean(dim=1)

        dissonance = torch.abs(audio_energy - visual_velocity)
        return dissonance

    def analyze_dataset(self, dataloader) -> Dict:
        """
        Analyze all samples and identify mixed-signal cases.
        Mixed signal = high dissonance but model prediction is wrong (false positive/negative)
        """
        all_dissonance = []
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for visual_x, audio_wav, labels in dataloader:
                visual_x = visual_x.to(self.device)
                audio_wav = audio_wav.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, audio_latents, visual_features = self.model(
                        visual_x, audio_wav
                    )

                dissonance = self.compute_dissonance_scores(
                    audio_latents, visual_features
                )
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)

                all_dissonance.extend(dissonance.cpu().numpy())
                all_predictions.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        all_dissonance = np.array(all_dissonance)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        dissonance_threshold = np.percentile(all_dissonance, 75)

        mixed_signal_mask = all_dissonance > dissonance_threshold

        misclassified = all_predictions != all_labels

        high_dissonance_correct = np.sum(mixed_signal_mask & ~misclassified)
        high_dissonance_wrong = np.sum(mixed_signal_mask & misclassified)

        low_dissonance_correct = np.sum(~mixed_signal_mask & ~misclassified)
        low_dissonance_wrong = np.sum(~mixed_signal_mask & misclassified)

        results = {
            "total_samples": len(all_dissonance),
            "mean_dissonance": np.mean(all_dissonance),
            "std_dissonance": np.std(all_dissonance),
            "dissonance_threshold_75pct": dissonance_threshold,
            "high_dissonance_correct": high_dissonance_correct,
            "high_dissonance_wrong": high_dissonance_wrong,
            "low_dissonance_correct": low_dissonance_correct,
            "low_dissonance_wrong": low_dissonance_wrong,
            "dissonance_per_sample": all_dissonance.tolist(),
            "labels": all_labels.tolist(),
            "predictions": all_predictions.tolist(),
        }

        print("\n" + "=" * 70)
        print("MIXED-SIGNAL ANALYSIS")
        print("=" * 70)
        print(f"Total Samples: {results['total_samples']}")
        print(
            f"Mean Dissonance: {results['mean_dissonance']:.4f} ± {results['std_dissonance']:.4f}"
        )
        print(f"\nConfusion Matrix by Dissonance Level:")
        print(f"                          Correct  Wrong")
        print(
            f"  High Dissonance (>75%):  {high_dissonance_correct:5d}    {high_dissonance_wrong:5d}"
        )
        print(
            f"  Low Dissonance (<75%):   {low_dissonance_correct:5d}    {low_dissonance_wrong:5d}"
        )

        if results["high_dissonance_wrong"] > 0:
            improvement_potential = (
                results["high_dissonance_wrong"] / results["total_samples"] * 100
            )
            print(f"\n  Mixed-Signal Error Rate: {improvement_potential:.1f}%")
            print(
                f"  -> CD Loss specifically targets these {results['high_dissonance_wrong']} samples"
            )

        return results


if __name__ == "__main__":
    print("Ablation Study Framework loaded successfully")
