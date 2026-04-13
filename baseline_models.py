"""
Baseline Models for Phase 3: Multi-Model Comparative Study
- Visual-Only: TimeSformer with linear head
- Audio-Only: Wav2Vec2 with linear head
- Alternative Fusion Heads: SVM, Random Forest, ELM
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings

warnings.filterwarnings("ignore")


class VisualOnlyModel(nn.Module):
    """
    Visual-Only baseline: TimeSformer with linear classification head.
    Proves that "individual modalities often miss emotional/structural depth."
    """

    def __init__(self, timesformer_config: Dict, num_classes: int = 1):
        super().__init__()
        from models.size_invariant_timesformer import SizeInvariantTimeSformer
        from models.efficientnet.efficientnet_pytorch import EfficientNet

        self.efficient_net = EfficientNet.from_pretrained("efficientnet-b0")
        efficient_net_block = timesformer_config["model"].get("efficient-net-block", 7)
        for m in self.efficient_net.modules():
            m.requires_grad = False
        self.efficient_net.eval()

        self.visual_engine = SizeInvariantTimeSformer(
            config=timesformer_config, require_attention=False
        )
        self.visual_dim = timesformer_config["model"]["dim"]

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.visual_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, visual_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, f, c, h, w = visual_x.shape

        visual_frames = visual_x.view(b * f, c, h, w)
        features = self.efficient_net.extract_features_at_block(visual_frames, 7)
        _, feat_c, feat_h, feat_w = features.shape
        visual_x = features.view(b, f, feat_c, feat_h, feat_w)

        pos_seq = torch.arange(1, f * feat_h * feat_w + 1, device=visual_x.device)
        cls_pos = torch.zeros(1, dtype=torch.long, device=visual_x.device)
        positions = torch.cat([cls_pos, pos_seq]).unsqueeze(0).repeat(b, 1)

        _, visual_features = self.visual_engine.forward(visual_x, positions=positions)
        cls_token_raw = visual_features[:, 0]

        logits = self.classifier(cls_token_raw)
        return logits, cls_token_raw


class AudioOnlyModel(nn.Module):
    """
    Audio-Only baseline: Wav2Vec2 with linear classification head.
    Demonstrates audio-only deepfake detection limitations.
    """

    def __init__(
        self,
        audio_model_name: str = "facebook/wav2vec2-large-xlsr-53",
        num_classes: int = 1,
    ):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.audio_engine = Wav2Vec2Model.from_pretrained(audio_model_name)
        for param in self.audio_engine.parameters():
            param.requires_grad = False
        self.audio_engine.eval()

        self.audio_dim = self.audio_engine.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.audio_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, audio_waveforms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            audio_outputs = self.audio_engine(audio_waveforms)
            audio_latents = audio_outputs.last_hidden_state

        pooled = audio_latents.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, pooled


class ExtremeLearningMachine(nn.Module):
    """
    ELM: Fast and effective for high-dimensional multimodal features.
    Single hidden-layer feedforward network with random weights.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_weights = nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.output_weights = None

        nn.init.uniform_(self.input_weights.weight, -1, 1)

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        H = torch.tanh(self.input_weights(x) + self.hidden_bias)

        if train:
            return H
        else:
            if self.output_weights is None:
                raise RuntimeError("ELM not fitted. Call fit() first.")
            return torch.mm(H, self.output_weights)

    def fit(self, H: torch.Tensor, targets: torch.Tensor):
        """Solve for output weights using Moore-Penrose pseudoinverse."""
        H_pinv = torch.linalg.pinv(H)
        self.output_weights = torch.mm(H_pinv, targets)
        return self

    def get_output_weights(self) -> Optional[torch.Tensor]:
        return self.output_weights


class FusionHeadEvaluator:
    """
    Evaluates different fusion heads on multimodal features.
    Supports: Neural Network, SVM, Random Forest, ELM.
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.models = {}

    def train_svm(
        self, features: np.ndarray, labels: np.ndarray, C: float = 1.0
    ) -> Dict:
        """Train SVM classifier on extracted features."""
        features_scaled = self.scaler.fit_transform(features)

        svm_model = SVC(kernel="rbf", C=C, probability=True, random_state=42)
        svm_model.fit(features_scaled, labels)

        self.models["svm"] = {"model": svm_model, "scaler": self.scaler}

        predictions = svm_model.predict(features_scaled)
        probs = svm_model.predict_proba(features_scaled)[:, 1]

        return self._compute_metrics(labels, predictions, probs)

    def train_random_forest(
        self, features: np.ndarray, labels: np.ndarray, n_estimators: int = 100
    ) -> Dict:
        """Train Random Forest classifier."""
        features_scaled = self.scaler.fit_transform(features)

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=10, random_state=42, n_jobs=-1
        )
        rf_model.fit(features_scaled, labels)

        self.models["rf"] = {"model": rf_model, "scaler": self.scaler}

        predictions = rf_model.predict(features_scaled)
        probs = rf_model.predict_proba(features_scaled)[:, 1]

        return self._compute_metrics(labels, predictions, probs)

    def train_elm(self, features: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Train Extreme Learning Machine."""
        actual_dim = features.shape[1]
        hidden_dim = min(512, actual_dim)

        # Create ELM with correct dimensions
        elm = ExtremeLearningMachine(input_dim=actual_dim, hidden_dim=hidden_dim)
        H_train = elm(features, train=True)

        targets = labels.unsqueeze(1) if len(labels.shape) == 1 else labels

        # Compute output weights directly and assign to elm
        H_pinv = torch.linalg.pinv(H_train)
        output_weights = H_pinv @ targets
        elm.output_weights = output_weights

        # Get predictions
        logits = elm(features, train=False)
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        predictions = (probs >= 0.5).astype(int)

        self.models["elm"] = elm

        return self._compute_metrics(labels.cpu().numpy(), predictions, probs)

    def _compute_metrics(
        self, labels: np.ndarray, predictions: np.ndarray, probs: np.ndarray
    ) -> Dict:
        """Compute evaluation metrics."""
        pred_classes = (probs >= 0.5).astype(int)

        return {
            "accuracy": accuracy_score(labels, pred_classes),
            "precision": precision_score(labels, pred_classes, zero_division=0),
            "recall": recall_score(labels, pred_classes, zero_division=0),
            "f1": f1_score(labels, pred_classes, zero_division=0),
            "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        }

    def predict(
        self, features: np.ndarray, model_name: str = "svm"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        model_info = self.models[model_name]

        if model_name in ["svm", "rf"]:
            features_scaled = model_info["scaler"].transform(features)
            predictions = model_info["model"].predict(features_scaled)
            probs = model_info["model"].predict_proba(features_scaled)[:, 1]
        else:
            features_tensor = torch.from_numpy(features).float()
            H = self.models[model_name](features_tensor, train=False)
            logits = self.models[model_name](H, train=False)
            probs = torch.sigmoid(logits).numpy().flatten()
            predictions = (probs >= 0.5).astype(int)

        return predictions, probs


def extract_multimodal_features(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fused multimodal features from trained SyncWeldNet for downstream classifiers.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for visual_x, audio_wav, labels in dataloader:
            visual_x = visual_x.to(device)
            audio_wav = audio_wav.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, audio_latents, visual_features = model(visual_x, audio_wav)

            fused = torch.cat(
                [audio_latents.mean(dim=1), visual_features.mean(dim=1)], dim=1
            )
            all_features.append(fused.cpu().numpy())
            all_labels.extend(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)

    return features, labels


def train_and_evaluate_all_baselines(
    visual_only_model: VisualOnlyModel,
    audio_only_model: AudioOnlyModel,
    full_model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    Train and evaluate all baseline models for the comparison matrix.
    Returns metrics for each model.
    """
    from train_syncweld import ContrastiveDissonanceLoss
    import torch.optim as optim

    results = {}

    print("\n" + "=" * 70)
    print("TRAINING VISUAL-ONLY BASELINE")
    print("=" * 70)

    visual_criterion = nn.BCEWithLogitsLoss()
    visual_optimizer = optim.AdamW(visual_only_model.parameters(), lr=1e-4)
    visual_scheduler = optim.lr_scheduler.OneCycleLR(
        visual_optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=8,
        pct_start=0.2,
    )

    visual_only_model, visual_tracker = train_baseline(
        visual_only_model,
        train_loader,
        val_loader,
        visual_criterion,
        visual_optimizer,
        visual_scheduler,
        device,
    )

    results["visual_only"] = {
        "accuracy": visual_tracker.history["val_accuracy"][-1],
        "precision": visual_tracker.history["val_precision"][-1],
        "recall": visual_tracker.history["val_recall"][-1],
        "f1": visual_tracker.history["val_f1"][-1],
        "auc": visual_tracker.history["val_auc"][-1],
        "train_time": sum(visual_tracker.history.get("epoch_times", [])),
    }

    print("\n" + "=" * 70)
    print("TRAINING AUDIO-ONLY BASELINE")
    print("=" * 70)

    audio_criterion = nn.BCEWithLogitsLoss()
    audio_optimizer = optim.AdamW(audio_only_model.parameters(), lr=1e-4)
    audio_scheduler = optim.lr_scheduler.OneCycleLR(
        audio_optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=8,
        pct_start=0.2,
    )

    audio_only_model, audio_tracker = train_baseline(
        audio_only_model,
        train_loader,
        val_loader,
        audio_criterion,
        audio_optimizer,
        audio_scheduler,
        device,
        is_audio_only=True,
    )

    results["audio_only"] = {
        "accuracy": audio_tracker.history["val_accuracy"][-1],
        "precision": audio_tracker.history["val_precision"][-1],
        "recall": audio_tracker.history["val_recall"][-1],
        "f1": audio_tracker.history["val_f1"][-1],
        "auc": audio_tracker.history["val_auc"][-1],
        "train_time": sum(audio_tracker.history.get("epoch_times", [])),
    }

    print("\n" + "=" * 70)
    print("EXTRACTING FEATURES FOR ALTERNATIVE FUSION HEADS")
    print("=" * 70)

    train_features, train_labels = extract_multimodal_features(
        full_model, train_loader, device
    )
    val_features, val_labels = extract_multimodal_features(
        full_model, val_loader, device
    )

    evaluator = FusionHeadEvaluator(feature_dim=train_features.shape[1])

    print("\nTraining SVM Head...")
    svm_metrics = evaluator.train_svm(train_features, train_labels)
    results["syncweld_svm"] = {**svm_metrics, "train_time": 0}

    print("\nTraining Random Forest Head...")
    rf_metrics = evaluator.train_random_forest(train_features, train_labels)
    results["syncweld_rf"] = {**rf_metrics, "train_time": 0}

    print("\nTraining ELM Head...")
    elm_metrics = evaluator.train_elm(
        torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).float()
    )
    results["syncweld_elm"] = {**elm_metrics, "train_time": 0}

    print("\n" + "=" * 70)
    print("COMPARISON MATRIX")
    print("=" * 70)
    print(f"{'Model':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Time(s)':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<30} {metrics['accuracy']:.4f}   {metrics['precision']:.4f}   "
            f"{metrics['recall']:.4f}   {metrics['f1']:.4f}   {metrics['train_time']:.1f}"
        )

    return results


def train_baseline(
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
    from extended_training import MetricsTracker, evaluate
    import time

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

        if is_audio_only:

            def eval_model(m, dl, c, d):
                m.eval()
                total_loss = 0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for vx, aw, lbl in dl:
                        aw = aw.to(d)
                        lbl = lbl.to(d).unsqueeze(1)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            lgts, _ = m(aw)
                            ls, _, _ = c(lgts, lbl, torch.zeros(1), torch.zeros(1))
                        total_loss += ls.item()
                        all_preds.extend(torch.sigmoid(lgts).detach().cpu().numpy())
                        all_labels.extend(lbl.detach().cpu().numpy())
                return total_loss / len(dl), *evaluate(m, dl, c, d)

            from train_syncweld import ContrastiveDissonanceLoss

            temp_criterion = ContrastiveDissonanceLoss()
            val_loss, acc, prec, rec, f1, auc = eval_model(
                model, val_loader, temp_criterion, device
            )
        else:
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
            "epoch_times": epoch_time,
        }
        tracker.update(metrics, epoch)

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={acc:.4f}, F1={f1:.4f}"
        )

    return model, tracker


if __name__ == "__main__":
    print("Baseline Models loaded successfully")
    print(
        "Available classes: VisualOnlyModel, AudioOnlyModel, FusionHeadEvaluator, ELM"
    )
