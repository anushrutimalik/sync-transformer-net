import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from models.size_invariant_timesformer import SizeInvariantTimeSformer
from models.efficientnet.efficientnet_pytorch import EfficientNet


class CrossModalAttention(nn.Module):
    def __init__(
        self, visual_dim: int, audio_dim: int, num_heads: int = 8, dropout: float = 0.1
    ):
        """
        Calculates cross attention between the visual sequence and audio sequence.
        Audio acts as the Query, Visual acts as Key and Value.
        """
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim

        # Project audio and visual into a shared embedding space if they differ
        self.audio_proj = (
            nn.Linear(audio_dim, visual_dim)
            if audio_dim != visual_dim
            else nn.Identity()
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=visual_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(visual_dim)

    def forward(self, visual_tokens, audio_latents):
        # audio_latents: [Batch, AudioSeqLen, AudioDim]
        # visual_tokens: [Batch, VisualSeqLen, VisualDim]

        # Both must be projected to the same dimension (visual_dim) before entering standard MultiheadAttention
        queries = self.audio_proj(audio_latents)  # [Batch, AudioSeqLen, VisualDim]
        keys = visual_tokens  # [Batch, VisualSeqLen, VisualDim]
        values = visual_tokens  # [Batch, VisualSeqLen, VisualDim]

        # apply cross attention
        attn_output, attn_weights = self.multihead_attn(queries, keys, values)

        output = self.layer_norm(attn_output + queries)
        return output, attn_weights


class SyncWeldNet(nn.Module):
    def __init__(
        self,
        timesformer_config,
        audio_model_name="facebook/wav2vec2-base",
        num_classes=1,
    ):
        """
        Fuses the MINTIME Size-Invariant TimeSformer with an Audio Engine to perform
        cross-modal deepfake detection.
        """
        super().__init__()

        # --- Visual Engine ---
        # 1. Initialize EfficientNet for spatial feature extraction
        self.efficient_net = EfficientNet.from_pretrained("efficientnet-b0")
        self.efficient_net_block = timesformer_config["model"].get(
            "efficient-net-block", 7
        )
        for m in self.efficient_net.modules():
            m.requires_grad = False
        self.efficient_net.eval()

        # 2. Initialize MINTIME visual backbone
        self.visual_engine = SizeInvariantTimeSformer(
            config=timesformer_config, require_attention=False
        )
        self.visual_dim = timesformer_config["model"]["dim"]

        # --- Audio Engine ---
        self.audio_engine = Wav2Vec2Model.from_pretrained(audio_model_name)
        # We generally freeze the large Wav2Vec2 model to prevent overfitting on smaller AV datasets
        for param in self.audio_engine.parameters():
            param.requires_grad = False
        self.audio_dim = (
            self.audio_engine.config.hidden_size
        )  # usually 1024 for Large-XLSR

        # --- Cross-Modal Welding Layer ---
        self.cross_modal_fusion = CrossModalAttention(
            visual_dim=self.visual_dim, audio_dim=self.audio_dim
        )

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.visual_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        visual_x,
        audio_waveforms,
        mask=None,
        identities_mask=None,
        size_embedding=None,
        positions=None,
    ):
        """
        visual_x: Input visual sequence extracted from EfficientNet patches [Batch, Frames, 3, 224, 224]
        audio_waveforms: Raw audio waveforms aligned with the videos
        """

        # 1. Process Visual Modality
        b, f, c, h, w = visual_x.shape
        # Extract features using EfficientNet block 7
        visual_frames = visual_x.view(b * f, c, h, w)
        features = self.efficient_net.extract_features_at_block(
            visual_frames, self.efficient_net_block
        )  # [B*F, Channels, P_H, P_W]
        _, feat_c, feat_h, feat_w = features.shape
        visual_x = features.view(b, f, feat_c, feat_h, feat_w)

        # MINTIME SizeInvariantTimeSformer requires explicit positions tracking due to faces identity logic
        if positions is None:
            # We construct basic sequential positions unless provided (accounting for +1 CLS token)
            pos_seq = torch.arange(1, f * feat_h * feat_w + 1, device=visual_x.device)
            cls_pos = torch.zeros(1, dtype=torch.long, device=visual_x.device)
            positions = torch.cat([cls_pos, pos_seq]).unsqueeze(0).repeat(b, 1)

        # Extract visual tokens using MINTIME's Transformer. We modified it to return the CLS and the full sequence.
        cls_token, visual_features = self.visual_engine.forward(
            visual_x,
            mask=mask,
            identities_mask=identities_mask,
            size_embedding=size_embedding,
            positions=positions,
        )

        # 2. Process Audio Modality
        # Extract audio latents from Wav2Vec2
        with torch.no_grad():
            audio_outputs = self.audio_engine(audio_waveforms)
            audio_latents = audio_outputs.last_hidden_state

        # 3. Weld Modalities (Cross-Modal Attention)
        # We now use the full visual_features [Batch, SeqLen, Dim]
        if len(visual_features.shape) == 2:
            visual_features = visual_features.unsqueeze(1)

        # Audio is usually [B, SeqLength, Dim]. If it is 2D, unsqueeze.
        if len(audio_latents.shape) == 2:
            audio_latents = audio_latents.unsqueeze(1)

        fused_latents, _ = self.cross_modal_fusion(
            visual_tokens=visual_features, audio_latents=audio_latents
        )

        # 4. Global Pooling & Classification
        # Average the fused sequence to get a single vector per video
        pooled_fused = fused_latents.mean(dim=1)  # [Batch, VisualDim]

        logits = self.classifier(pooled_fused)
        return logits, audio_latents, visual_features
