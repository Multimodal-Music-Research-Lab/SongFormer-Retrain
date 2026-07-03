from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .labels import IGNORE_INDEX

try:
    from models.songgen_conformer_encoder import ConformerEncoder
except Exception:  # pragma: no cover - allows package-style imports in tests
    from SongFormer.models.songgen_conformer_encoder import ConformerEncoder


@dataclass
class AlmaV9ModelConfig:
    audio_dim: int
    vocab_size: int
    num_classes: int = 8
    d_model: int = 384
    lyrics_layers: int = 6
    lyrics_heads: int = 8
    lyrics_ffn_dim: int = 1536
    fusion_layers: int = 4
    fusion_block_type: str = "mamba"
    dropout: float = 0.1
    frame_hz: float = 75.0
    loss_weight_boundary: float = 0.2
    loss_weight_function: float = 0.8
    focal_weight: float = 0.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


class SongGenStyleLyricsEncoder(nn.Module):
    """Trainable SongGen-style lyrics encoder.

    ALMA-Chor states that its lyrics encoder follows SongGen. The local
    repository already contains a Conformer block adapted from that family, so
    v9 uses a token embedding followed by that Conformer and gathers hidden
    states at explicit ``<LINE>`` token positions.
    """

    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.encoder = ConformerEncoder(
            input_size=dim,
            output_size=dim,
            attention_heads=heads,
            linear_units=ffn_dim,
            num_blocks=layers,
            dropout_rate=dropout,
            positional_dropout_rate=dropout,
            attention_dropout_rate=dropout,
            input_layer="linear",
            static_chunk_size=0,
            use_dynamic_chunk=False,
            use_dynamic_left_chunk=False,
            macaron_style=True,
            use_cnn_module=True,
            cnn_module_kernel=15,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids, attention_mask, line_positions, line_valid):
        token_states = self.embedding(input_ids)
        encoded, _ = self.encoder(token_states, attention_mask.bool())
        encoded = self.norm(encoded)

        safe_positions = line_positions.clamp_min(0)
        gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, encoded.size(-1))
        line_states = torch.gather(encoded, dim=1, index=gather_index)
        line_states = line_states * line_valid.unsqueeze(-1).to(line_states.dtype)
        return line_states


class MambaFusionBlock(nn.Module):
    def __init__(self, dim: int, layers: int, dropout: float, block_type: str):
        super().__init__()
        block_type = str(block_type).lower()
        self.block_type = block_type
        self.blocks = nn.ModuleList()
        if block_type == "mamba2":
            try:
                from mamba_ssm import Mamba2
            except Exception as exc:
                raise ImportError("fusion_block_type=mamba2 requires mamba_ssm") from exc
            for _ in range(layers):
                self.blocks.append(
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(dim),
                            "ssm": Mamba2(d_model=dim, d_state=64, d_conv=4, expand=2),
                            "drop": nn.Dropout(dropout),
                        }
                    )
                )
        elif block_type in {"mamba", "mamba1"}:
            try:
                from mamba_ssm import Mamba
            except Exception as exc:
                raise ImportError("fusion_block_type=mamba requires mamba_ssm") from exc
            for _ in range(layers):
                self.blocks.append(
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(dim),
                            "ssm": Mamba(d_model=dim, d_state=64, d_conv=4, expand=2),
                            "drop": nn.Dropout(dropout),
                        }
                    )
                )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x, seq_valid):
        if hasattr(self, "transformer"):
            return self.transformer(x, src_key_padding_mask=~seq_valid)
        for block in self.blocks:
            x = x + block["drop"](block["ssm"](block["norm"](x)))
        return x


class AlmaV9Model(nn.Module):
    def __init__(self, cfg: AlmaV9ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(cfg.audio_dim),
            nn.Linear(cfg.audio_dim, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.lyrics_encoder = SongGenStyleLyricsEncoder(
            vocab_size=cfg.vocab_size,
            dim=d,
            layers=cfg.lyrics_layers,
            heads=cfg.lyrics_heads,
            ffn_dim=cfg.lyrics_ffn_dim,
            dropout=cfg.dropout,
        )
        self.fusion_in = nn.Sequential(
            nn.Linear(d * 2 + 1, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(d),
        )
        self.fusion = MambaFusionBlock(
            dim=d,
            layers=cfg.fusion_layers,
            dropout=cfg.dropout,
            block_type=cfg.fusion_block_type,
        )

        # Boundary remains audio-only to avoid letting vocal/lyric onset timing
        # perturb structural boundary timing. Function prediction uses ALMA-style
        # audio-lyric joint sequence modeling.
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d // 2, 1),
        )
        self.function_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d // 2, cfg.num_classes),
        )

    def _align_lines_to_frames(
        self,
        line_states,
        line_starts,
        line_ends,
        line_valid,
        crop_start,
        num_frames: int,
        device,
    ):
        batch, _, dim = line_states.shape
        aligned = line_states.new_zeros((batch, num_frames, dim))
        counts = line_states.new_zeros((batch, num_frames, 1))
        for b in range(batch):
            valid_lines = torch.nonzero(line_valid[b], as_tuple=False).flatten().tolist()
            for l in valid_lines:
                start = float(line_starts[b, l].detach().cpu())
                end = float(line_ends[b, l].detach().cpu())
                if end <= start:
                    continue
                left = int(torch.floor((line_starts[b, l] - crop_start[b]) * self.cfg.frame_hz).item())
                right = int(torch.ceil((line_ends[b, l] - crop_start[b]) * self.cfg.frame_hz).item())
                left = max(0, left)
                right = min(num_frames, right)
                if right <= left:
                    continue
                aligned[b, left:right] += line_states[b, l].view(1, -1)
                counts[b, left:right] += 1.0
        active = counts > 0
        aligned = aligned / counts.clamp_min(1.0)
        return aligned.to(device), active.to(device).to(line_states.dtype)

    def forward(self, batch):
        audio = batch["audio_features"]
        seq_valid = batch["seq_valid"]
        audio_states = self.audio_proj(audio)
        line_states = self.lyrics_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            line_positions=batch["line_positions"],
            line_valid=batch["line_valid"],
        )
        lyric_states, lyric_active = self._align_lines_to_frames(
            line_states=line_states,
            line_starts=batch["line_starts"],
            line_ends=batch["line_ends"],
            line_valid=batch["line_valid"],
            crop_start=batch["crop_start"],
            num_frames=audio_states.size(1),
            device=audio_states.device,
        )
        joint = torch.cat([audio_states, lyric_states, lyric_active], dim=-1)
        joint = self.fusion_in(joint)
        joint = self.fusion(joint, seq_valid=seq_valid)
        boundary_logits = self.boundary_head(audio_states).squeeze(-1)
        function_logits = self.function_head(joint)
        return {
            "boundary_logits": boundary_logits,
            "function_logits": function_logits,
            "lyric_frame_coverage": lyric_active.squeeze(-1).mean(),
        }

    def _focal_loss(self, logits, targets):
        ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none", ignore_index=IGNORE_INDEX)
        valid_targets = targets.clamp_min(0)
        pt = torch.gather(F.softmax(logits, dim=-1), -1, valid_targets.unsqueeze(-1)).squeeze(-1)
        mod = (1.0 - pt).pow(self.cfg.focal_gamma)
        return self.cfg.focal_alpha * mod * ce

    def compute_loss(self, outputs, batch):
        seq_valid = batch["seq_valid"]
        boundary_valid = batch["boundary_valid"] & seq_valid
        function_valid = batch["function_valid"] & seq_valid

        if boundary_valid.any():
            bce = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"],
                batch["boundary_target"],
                reduction="none",
            )
            loss_boundary = bce[boundary_valid].mean()
        else:
            loss_boundary = outputs["boundary_logits"].sum() * 0.0

        function_target = batch["function_target"].clone()
        function_target[~function_valid] = IGNORE_INDEX
        ce = F.cross_entropy(
            outputs["function_logits"].transpose(1, 2),
            function_target,
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )
        if self.cfg.focal_weight > 0:
            ce = ce + self.cfg.focal_weight * self._focal_loss(outputs["function_logits"], function_target)
        if function_valid.any():
            loss_function = ce[function_valid].mean()
        else:
            loss_function = outputs["function_logits"].sum() * 0.0

        loss = self.cfg.loss_weight_boundary * loss_boundary + self.cfg.loss_weight_function * loss_function
        return {
            "loss": loss,
            "loss_boundary": loss_boundary.detach(),
            "loss_function": loss_function.detach(),
            "lyric_frame_coverage": outputs["lyric_frame_coverage"].detach(),
        }
