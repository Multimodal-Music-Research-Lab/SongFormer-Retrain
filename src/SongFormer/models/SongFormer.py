import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataset.custom_types import MsaInfo
from msaf.eval import compute_results
from postprocessing.functional import postprocess_functional_structure
from x_transformers import Encoder
import bisect


class Head(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, activation="silu"):
        super().__init__()
        hidden_dims = hidden_dims or []
        act_layers = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
        act_layer = act_layers.get(activation.lower())
        if not act_layer:
            raise ValueError(f"Unsupported activation: {activation}")

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_layer())
        self.net = nn.Sequential(*layers)

    def reset_parameters(self, confidence):
        bias_value = -torch.log(torch.tensor((1 - confidence) / confidence))
        self.net[-1].bias.data.fill_(bias_value.item())

    def forward(self, x):
        batch, T, C = x.shape
        x = x.reshape(-1, C)
        x = self.net(x)
        return x.reshape(batch, T, -1)


class WrapedTransformerEncoder(nn.Module):
    def __init__(
        self, input_dim, transformer_input_dim, num_layers=1, nhead=8, dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.transformer_input_dim = transformer_input_dim

        if input_dim != transformer_input_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, transformer_input_dim),
                nn.LayerNorm(transformer_input_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(transformer_input_dim, transformer_input_dim),
            )
        else:
            self.input_proj = nn.Identity()

        self.transformer = Encoder(
            dim=transformer_input_dim,
            depth=num_layers,
            heads=nhead,
            layer_dropout=dropout,
            attn_dropout=dropout,
            ff_dropout=dropout,
            attn_flash=True,
            rotary_pos_emb=True,
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        The input src_key_padding_mask is a B x T boolean mask, where True indicates masked positions.
        However, in x-transformers, False indicates masked positions.
        Therefore, it needs to be converted so that False represents masked positions.
        """
        x = self.input_proj(x)
        mask = (
            ~torch.tensor(src_key_padding_mask, dtype=torch.bool, device=x.device)
            if src_key_padding_mask is not None
            else None
        )
        return self.transformer(x, mask=mask)


def prefix_dict(d, prefix: str):
    if prefix:
        return d
    return {prefix + key: value for key, value in d.items()}


class TimeDownsample(nn.Module):
    def __init__(
        self, dim_in, dim_out=None, kernel_size=5, stride=5, padding=0, dropout=0.1
    ):
        super().__init__()
        self.dim_out = dim_out or dim_in
        assert self.dim_out % 2 == 0

        self.depthwise_conv = nn.Conv1d(
            in_channels=dim_in,
            out_channels=dim_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim_in,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=dim_in,
            out_channels=self.dim_out,
            kernel_size=1,
            bias=False,
        )
        self.pool = nn.AvgPool1d(kernel_size, stride, padding=padding)
        self.norm1 = nn.LayerNorm(self.dim_out)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        if dim_in != self.dim_out:
            self.residual_conv = nn.Conv1d(
                dim_in, self.dim_out, kernel_size=1, bias=False
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x  # [B, T, D_in]
        # Convolutional module
        x_c = x.transpose(1, 2)  # [B, D_in, T]
        x_c = self.depthwise_conv(x_c)  # [B, D_in, T_down]
        x_c = self.pointwise_conv(x_c)  # [B, D_out, T_down]

        # Residual module
        res = self.pool(residual.transpose(1, 2))  # [B, D_in, T]
        if self.residual_conv:
            res = self.residual_conv(res)  # [B, D_out, T_down]
        x_c = x_c + res  # [B, D_out, T_down]
        x_c = x_c.transpose(1, 2)  # [B, T_down, D_out]
        x_c = self.norm1(x_c)
        x_c = self.act1(x_c)
        x_c = self.dropout1(x_c)
        return x_c


class AddFuse(nn.Module):
    def __init__(self):
        super(AddFuse, self).__init__()

    def forward(self, x, cond):
        return x + cond


class TVLoss1D(nn.Module):
    def __init__(
        self, beta=1.0, lambda_tv=0.4, boundary_threshold=0.01, reduction_weight=0.1
    ):
        """
        Args:
            beta: Exponential parameter for TV loss (recommended 0.5~1.0)
            lambda_tv: Overall weight for TV loss
            boundary_threshold: Label threshold to determine if a region is a "boundary area" (e.g., 0.01)
            reduction_weight: Scaling factor for TV penalty within boundary regions (e.g., 0.1, meaning only 10% penalty)
        """
        super().__init__()
        self.beta = beta
        self.lambda_tv = lambda_tv
        self.boundary_threshold = boundary_threshold
        self.reduction_weight = reduction_weight

    def forward(self, pred, target=None):
        """
        Args:
            pred: (B, T) or (B, T, 1), float boundary scores output by the model
            target: (B, T) or (B, T, 1), ground truth labels (optional, used for spatial weighting if provided)

        Returns:
            scalar: weighted TV loss
        """
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if target is not None and target.dim() == 3:
            target = target.squeeze(-1)

        diff = pred[:, 1:] - pred[:, :-1]
        tv_base = torch.pow(torch.abs(diff) + 1e-8, self.beta)

        if target is None:
            return self.lambda_tv * tv_base.mean()

        left_in_boundary = target[:, :-1] > self.boundary_threshold
        right_in_boundary = target[:, 1:] > self.boundary_threshold
        near_boundary = left_in_boundary | right_in_boundary
        weight_mask = torch.where(
            near_boundary,
            self.reduction_weight * torch.ones_like(tv_base),
            torch.ones_like(tv_base),
        )
        tv_weighted = (tv_base * weight_mask).mean()
        return self.lambda_tv * tv_weighted


class SoftmaxFocalLoss(nn.Module):
    """
    Softmax Focal Loss for single-label multi-class classification.
    Suitable for mutually exclusive classes.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, T, C], raw logits
            targets: [B, T, C] (soft) or [B, T] (hard, dtype=long)
        Returns:
            loss: scalar or [B, T] depending on reduction
        """
        log_probs = F.log_softmax(pred, dim=-1)
        probs = torch.exp(log_probs)

        if targets.dtype == torch.long:
            targets_onehot = F.one_hot(targets, num_classes=pred.size(-1)).float()
        else:
            targets_onehot = targets

        p_t = (probs * targets_onehot).sum(dim=-1)
        p_t = p_t.clamp(min=1e-8, max=1.0 - 1e-8)

        if self.alpha > 0:
            alpha_t = self.alpha * targets_onehot + (1 - self.alpha) * (
                1 - targets_onehot
            )
            alpha_weight = (alpha_t * targets_onehot).sum(dim=-1)
        else:
            alpha_weight = 1.0

        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -log_probs * targets_onehot
        ce_loss = ce_loss.sum(dim=-1)

        loss = alpha_weight * focal_weight * ce_loss
        return loss



class LyricsEncoder(nn.Module):
    def __init__(
        self,
        lyrics_input_dim,
        time_feat_dim,
        hidden_dim,
        num_heads,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(lyrics_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, line_embeddings, line_time_features, line_masks=None):
        memory = self.text_proj(line_embeddings) + self.time_proj(line_time_features)
        if line_masks is not None:
            all_pad = line_masks.all(dim=1)
            if all_pad.any():
                line_masks = line_masks.clone()
                line_masks[all_pad, 0] = False
        return self.encoder(memory, src_key_padding_mask=line_masks)


class LyricsHeadAdapter(nn.Module):
    def __init__(
        self,
        audio_dim,
        lyrics_input_dim,
        time_feat_dim,
        frame_time_feat_dim,
        num_classes,
        hidden_dim,
        adapter_hidden_dim,
        num_heads,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()
        self.lyrics_encoder = LyricsEncoder(
            lyrics_input_dim=lyrics_input_dim,
            time_feat_dim=time_feat_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.frame_time_proj = nn.Sequential(
            nn.Linear(frame_time_feat_dim, audio_dim),
            nn.LayerNorm(audio_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fused_dim = audio_dim * 4
        self.boundary_delta_head = nn.Sequential(
            nn.Linear(fused_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_dim, 1),
        )
        self.function_delta_head = nn.Sequential(
            nn.Linear(fused_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_dim, num_classes),
        )
        nn.init.zeros_(self.boundary_delta_head[-1].weight)
        nn.init.zeros_(self.boundary_delta_head[-1].bias)
        nn.init.zeros_(self.function_delta_head[-1].weight)
        nn.init.zeros_(self.function_delta_head[-1].bias)

    def _zero_deltas(self, audio_states):
        batch, time_len, _ = audio_states.shape
        boundary_out = self.boundary_delta_head[-1].out_features
        function_out = self.function_delta_head[-1].out_features
        return (
            audio_states.new_zeros(batch, time_len, boundary_out).squeeze(-1),
            audio_states.new_zeros(batch, time_len, function_out),
        )

    def forward(
        self,
        audio_states,
        lyrics_line_embeddings=None,
        lyrics_line_time_features=None,
        lyrics_line_masks=None,
        lyrics_frame_time_features=None,
        has_lyrics=None,
    ):
        if lyrics_line_embeddings is None or lyrics_line_time_features is None:
            return self._zero_deltas(audio_states)

        if not isinstance(lyrics_line_embeddings, torch.Tensor):
            lyrics_line_embeddings = torch.tensor(lyrics_line_embeddings, device=audio_states.device)
        if not isinstance(lyrics_line_time_features, torch.Tensor):
            lyrics_line_time_features = torch.tensor(lyrics_line_time_features, device=audio_states.device)
        if lyrics_line_masks is not None and not isinstance(lyrics_line_masks, torch.Tensor):
            lyrics_line_masks = torch.tensor(lyrics_line_masks, device=audio_states.device)
        if lyrics_frame_time_features is not None and not isinstance(lyrics_frame_time_features, torch.Tensor):
            lyrics_frame_time_features = torch.tensor(lyrics_frame_time_features, device=audio_states.device)

        lyrics_line_embeddings = lyrics_line_embeddings.to(audio_states.device).float()
        lyrics_line_time_features = lyrics_line_time_features.to(audio_states.device).float()
        lyrics_line_masks = (
            lyrics_line_masks.to(audio_states.device).bool()
            if lyrics_line_masks is not None
            else torch.zeros(
                lyrics_line_embeddings.size(0),
                lyrics_line_embeddings.size(1),
                device=audio_states.device,
                dtype=torch.bool,
            )
        )

        batch, time_len, audio_dim = audio_states.shape
        if has_lyrics is None:
            has_lyrics = audio_states.new_zeros(batch)
        elif not isinstance(has_lyrics, torch.Tensor):
            has_lyrics = torch.tensor(has_lyrics, device=audio_states.device)
        has_lyrics = has_lyrics.to(audio_states.device).float().view(batch, 1, 1)

        lyrics_memory = self.lyrics_encoder(
            lyrics_line_embeddings,
            lyrics_line_time_features,
            lyrics_line_masks,
        )
        lyrics_context, _ = self.cross_attn(
            query=audio_states,
            key=lyrics_memory,
            value=lyrics_memory,
            key_padding_mask=lyrics_line_masks,
            need_weights=False,
        )

        valid = (~lyrics_line_masks).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        song_context = (lyrics_memory * valid).sum(dim=1) / denom
        song_context = song_context.unsqueeze(1).expand(-1, time_len, -1)

        if lyrics_frame_time_features is None:
            frame_context = audio_states.new_zeros(batch, time_len, audio_dim)
        else:
            lyrics_frame_time_features = lyrics_frame_time_features.to(audio_states.device).float()
            if lyrics_frame_time_features.size(1) > time_len:
                lyrics_frame_time_features = lyrics_frame_time_features[:, :time_len]
            elif lyrics_frame_time_features.size(1) < time_len:
                pad = lyrics_frame_time_features.new_zeros(
                    batch,
                    time_len - lyrics_frame_time_features.size(1),
                    lyrics_frame_time_features.size(2),
                )
                lyrics_frame_time_features = torch.cat([lyrics_frame_time_features, pad], dim=1)
            frame_context = self.frame_time_proj(lyrics_frame_time_features)

        fused = torch.cat([audio_states, lyrics_context, frame_context, song_context], dim=-1)
        boundary_delta = self.boundary_delta_head(fused).squeeze(-1)
        function_delta = self.function_delta_head(fused)
        return boundary_delta * has_lyrics.squeeze(-1), function_delta * has_lyrics


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_norm = nn.LayerNorm(config.input_dim)
        self.mixed_win_downsample = nn.Linear(config.input_dim_raw, config.input_dim)
        self.dataset_class_prefix = nn.Embedding(
            num_embeddings=config.num_dataset_classes,
            embedding_dim=config.transformer_encoder_input_dim,
        )
        self.down_sample_conv = TimeDownsample(
            dim_in=config.input_dim,
            dim_out=config.transformer_encoder_input_dim,
            kernel_size=config.down_sample_conv_kernel_size,
            stride=config.down_sample_conv_stride,
            dropout=config.down_sample_conv_dropout,
            padding=config.down_sample_conv_padding,
        )
        self.AddFuse = AddFuse()
        self.transformer = WrapedTransformerEncoder(
            input_dim=config.transformer_encoder_input_dim,
            transformer_input_dim=config.transformer_input_dim,
            num_layers=config.num_transformer_layers,
            nhead=config.transformer_nhead,
            dropout=config.transformer_dropout,
        )
        self.boundary_TVLoss1D = TVLoss1D(
            beta=config.boundary_tv_loss_beta,
            lambda_tv=config.boundary_tv_loss_lambda,
            boundary_threshold=config.boundary_tv_loss_boundary_threshold,
            reduction_weight=config.boundary_tv_loss_reduction_weight,
        )
        self.label_focal_loss = SoftmaxFocalLoss(
            alpha=config.label_focal_loss_alpha, gamma=config.label_focal_loss_gamma
        )
        self.boundary_head = Head(config.transformer_input_dim, 1)
        self.function_head = Head(config.transformer_input_dim, config.num_classes)

        # =========================
        # [ADDED FOR LYRICS V2]
        # Lyrics v3: line-token lyrics encoder + residual deltas for both heads.
        # Base SongFormer modules above keep the same initialization order as the
        # no-lyrics baseline; the lyrics branch is initialized afterwards.
        # =========================
        self.use_lyrics = getattr(config, "use_lyrics", False)
        self.lyrics_input_dim = getattr(config, "lyrics_input_dim", 1024)
        self.lyrics_time_feat_dim = getattr(config, "lyrics_time_feat_dim", 8)
        self.lyrics_frame_time_feat_dim = getattr(config, "lyrics_frame_time_feat_dim", 6)
        self.lyrics_dropout = getattr(config, "lyrics_dropout", 0.1)
        self.lyrics_condition_dropout = getattr(config, "lyrics_condition_dropout", 0.0)
        self.lyrics_encoder_hidden_dim = getattr(
            config, "lyrics_encoder_hidden_dim", config.transformer_input_dim
        )
        self.lyrics_encoder_layers = getattr(config, "lyrics_encoder_layers", 2)
        self.lyrics_attn_num_heads = getattr(
            config, "lyrics_attn_num_heads", config.transformer_nhead
        )
        self.lyrics_adapter_hidden_dim = getattr(
            config, "lyrics_adapter_hidden_dim", config.transformer_input_dim
        )

        if self.use_lyrics:
            self.lyrics_head_adapter = LyricsHeadAdapter(
                audio_dim=config.transformer_input_dim,
                lyrics_input_dim=self.lyrics_input_dim,
                time_feat_dim=self.lyrics_time_feat_dim,
                frame_time_feat_dim=self.lyrics_frame_time_feat_dim,
                num_classes=config.num_classes,
                hidden_dim=self.lyrics_encoder_hidden_dim,
                adapter_hidden_dim=self.lyrics_adapter_hidden_dim,
                num_heads=self.lyrics_attn_num_heads,
                num_layers=self.lyrics_encoder_layers,
                dropout=self.lyrics_dropout,
            )
        else:
            self.lyrics_head_adapter = None

    def cal_metrics(self, gt_info: MsaInfo, msa_info: MsaInfo):
        assert gt_info[-1][1] == "end" and msa_info[-1][1] == "end", (
            "gt_info and msa_info should end with 'end'"
        )
        gt_info_labels = [label for time_, label in gt_info][:-1]
        gt_info_inters = [time_ for time_, label in gt_info]
        gt_info_inters = np.column_stack(
            [np.array(gt_info_inters[:-1]), np.array(gt_info_inters[1:])]
        )

        msa_info_labels = [label for time_, label in msa_info][:-1]
        msa_info_inters = [time_ for time_, label in msa_info]
        msa_info_inters = np.column_stack(
            [np.array(msa_info_inters[:-1]), np.array(msa_info_inters[1:])]
        )
        result = compute_results(
            ann_inter=gt_info_inters,
            est_inter=msa_info_inters,
            ann_labels=gt_info_labels,
            est_labels=msa_info_labels,
            bins=11,
            est_file="test.txt",
            weight=0.58,
        )
        return result

    def cal_acc(
        self, ann_info: MsaInfo | str, est_info: MsaInfo | str, post_digit: int = 3
    ):
        ann_info_time = [
            int(round(time_, post_digit) * (10**post_digit))
            for time_, label in ann_info
        ]
        est_info_time = [
            int(round(time_, post_digit) * (10**post_digit))
            for time_, label in est_info
        ]

        common_start_time = max(ann_info_time[0], est_info_time[0])
        common_end_time = min(ann_info_time[-1], est_info_time[-1])

        time_points = {common_start_time, common_end_time}
        time_points.update(
            {
                time_
                for time_ in ann_info_time
                if common_start_time <= time_ <= common_end_time
            }
        )
        time_points.update(
            {
                time_
                for time_ in est_info_time
                if common_start_time <= time_ <= common_end_time
            }
        )

        time_points = sorted(time_points)
        total_duration, total_score = 0, 0

        for idx in range(len(time_points) - 1):
            duration = time_points[idx + 1] - time_points[idx]
            ann_label = ann_info[
                bisect.bisect_right(ann_info_time, time_points[idx]) - 1
            ][1]
            est_label = est_info[
                bisect.bisect_right(est_info_time, time_points[idx]) - 1
            ][1]
            total_duration += duration
            if ann_label == est_label:
                total_score += duration
        return total_score / total_duration

    # =========================
    # [ADDED FOR LYRICS V2]
    # =========================
    def apply_lyrics_head_adapter(
        self,
        audio_states,
        boundary_logits,
        function_logits,
        lyrics_line_embeddings=None,
        lyrics_line_time_features=None,
        lyrics_line_masks=None,
        lyrics_frame_time_features=None,
        has_lyrics=None,
    ):
        if (not self.use_lyrics) or self.lyrics_head_adapter is None:
            return boundary_logits, function_logits

        if has_lyrics is not None and not isinstance(has_lyrics, torch.Tensor):
            has_lyrics = torch.tensor(has_lyrics, device=audio_states.device)
        if has_lyrics is None:
            has_lyrics = audio_states.new_zeros(audio_states.size(0))
        else:
            has_lyrics = has_lyrics.to(audio_states.device).float().view(-1)

        if self.training and self.lyrics_condition_dropout > 0:
            keep = (
                torch.rand_like(has_lyrics, dtype=torch.float32)
                >= float(self.lyrics_condition_dropout)
            ).float()
            has_lyrics = has_lyrics * keep

        boundary_delta, function_delta = self.lyrics_head_adapter(
            audio_states=audio_states,
            lyrics_line_embeddings=lyrics_line_embeddings,
            lyrics_line_time_features=lyrics_line_time_features,
            lyrics_line_masks=lyrics_line_masks,
            lyrics_frame_time_features=lyrics_frame_time_features,
            has_lyrics=has_lyrics,
        )
        return boundary_logits + boundary_delta, function_logits + function_delta

    def infer_with_metrics(self, batch, prefix: str = None):
        with torch.no_grad():
            logits = self.forward_func(batch)

            losses = self.compute_losses(logits, batch, prefix=None)

            expanded_mask = batch["label_id_masks"].expand(
                -1, logits["function_logits"].size(1), -1
            )
            logits["function_logits"] = logits["function_logits"].masked_fill(
                expanded_mask, -float("inf")
            )

            msa_info = postprocess_functional_structure(
                logits=logits, config=self.config
            )
            gt_info = batch["msa_infos"][0]
            results = self.cal_metrics(gt_info=gt_info, msa_info=msa_info)

        ret_results = {
            "loss": losses["loss"].item(),
            "HitRate_3P": results["HitRate_3P"],
            "HitRate_3R": results["HitRate_3R"],
            "HitRate_3F": results["HitRate_3F"],
            "HitRate_0.5P": results["HitRate_0.5P"],
            "HitRate_0.5R": results["HitRate_0.5R"],
            "HitRate_0.5F": results["HitRate_0.5F"],
            "PWF": results["PWF"],
            "PWP": results["PWP"],
            "PWR": results["PWR"],
            "Sf": results["Sf"],
            "So": results["So"],
            "Su": results["Su"],
            "acc": self.cal_acc(ann_info=gt_info, est_info=msa_info),
        }
        if prefix:
            ret_results = prefix_dict(ret_results, prefix)

        return ret_results

    def infer(
        self,
        input_embeddings,
        dataset_ids,
        label_id_masks,
        lyrics_line_embeddings=None,
        lyrics_line_time_features=None,
        lyrics_line_masks=None,
        lyrics_frame_time_features=None,
        has_lyrics=None,
        prefix: str = None,
        with_logits=False,
    ):
        with torch.no_grad():
            input_embeddings = self.mixed_win_downsample(input_embeddings)
            input_embeddings = self.input_norm(input_embeddings)
            x_audio = self.down_sample_conv(input_embeddings)

            dataset_prefix = self.dataset_class_prefix(dataset_ids)
            dataset_prefix_expand = dataset_prefix.unsqueeze(1).expand(
                x_audio.size(0), 1, -1
            )
            x_audio = self.AddFuse(x=x_audio, cond=dataset_prefix_expand)

            x_audio = self.transformer(x=x_audio, src_key_padding_mask=None)

            boundary_logits = self.boundary_head(x_audio).squeeze(-1)
            function_logits = self.function_head(x_audio)
            boundary_logits, function_logits = self.apply_lyrics_head_adapter(
                audio_states=x_audio,
                boundary_logits=boundary_logits,
                function_logits=function_logits,
                lyrics_line_embeddings=lyrics_line_embeddings,
                lyrics_line_time_features=lyrics_line_time_features,
                lyrics_line_masks=lyrics_line_masks,
                lyrics_frame_time_features=lyrics_frame_time_features,
                has_lyrics=has_lyrics,
            )

            logits = {
                "function_logits": function_logits,
                "boundary_logits": boundary_logits,
            }

            expanded_mask = label_id_masks.expand(
                -1, logits["function_logits"].size(1), -1
            )
            logits["function_logits"] = logits["function_logits"].masked_fill(
                expanded_mask, -float("inf")
            )

            msa_info = postprocess_functional_structure(
                logits=logits, config=self.config
            )

        return (msa_info, logits) if with_logits else msa_info

    def compute_losses(self, outputs, batch, prefix: str = None):
        loss = 0.0
        losses = {}

        loss_section = F.binary_cross_entropy_with_logits(
            outputs["boundary_logits"],
            batch["widen_true_boundaries"],
            reduction="none",
        )
        loss_section += self.config.boundary_tvloss_weight * self.boundary_TVLoss1D(
            pred=outputs["boundary_logits"],
            target=batch["widen_true_boundaries"],
        )
        loss_function = F.cross_entropy(
            outputs["function_logits"].transpose(1, 2),
            batch["true_functions"].transpose(1, 2),
            reduction="none",
        )
        # input is [B, T, C]
        ttt = self.config.label_focal_loss_weight * self.label_focal_loss(
            pred=outputs["function_logits"], targets=batch["true_functions"]
        )
        loss_function += ttt

        float_masks = (~batch["masks"]).float()
        boundary_mask = batch.get("boundary_mask", None)
        function_mask = batch.get("function_mask", None)
        if boundary_mask is not None:
            boundary_mask = (~boundary_mask).float()
        else:
            boundary_mask = 1

        if function_mask is not None:
            function_mask = (~function_mask).float()
        else:
            function_mask = 1

        loss_section = torch.mean(boundary_mask * float_masks * loss_section)
        loss_function = torch.mean(function_mask * float_masks * loss_function)

        loss_section *= self.config.loss_weight_section
        loss_function *= self.config.loss_weight_function

        if self.config.learn_label:
            loss += loss_function
        if self.config.learn_segment:
            loss += loss_section

        losses.update(
            loss=loss,
            loss_section=loss_section,
            loss_function=loss_function,
        )
        if prefix:
            losses = prefix_dict(losses, prefix)
        return losses

    def forward_func(self, batch):
        input_embeddings = batch["input_embeddings"]
        input_embeddings = self.mixed_win_downsample(input_embeddings)
        input_embeddings = self.input_norm(input_embeddings)
        x_audio = self.down_sample_conv(input_embeddings)

        dataset_prefix = self.dataset_class_prefix(batch["dataset_ids"])
        x_audio = self.AddFuse(x=x_audio, cond=dataset_prefix.unsqueeze(1))

        src_key_padding_mask = batch["masks"]
        x_audio = self.transformer(x=x_audio, src_key_padding_mask=src_key_padding_mask)

        boundary_logits = self.boundary_head(x_audio).squeeze(-1)
        function_logits = self.function_head(x_audio)
        boundary_logits, function_logits = self.apply_lyrics_head_adapter(
            audio_states=x_audio,
            boundary_logits=boundary_logits,
            function_logits=function_logits,
            lyrics_line_embeddings=batch.get("lyrics_line_embeddings", None),
            lyrics_line_time_features=batch.get("lyrics_line_time_features", None),
            lyrics_line_masks=batch.get("lyrics_line_masks", None),
            lyrics_frame_time_features=batch.get("lyrics_frame_time_features", None),
            has_lyrics=batch.get("has_lyrics", None),
        )

        logits = {
            "function_logits": function_logits,
            "boundary_logits": boundary_logits,
        }
        return logits

    def forward(self, batch):
        logits = self.forward_func(batch)
        losses = self.compute_losses(logits, batch, prefix=None)
        return logits, losses["loss"], losses
