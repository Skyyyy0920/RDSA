"""Inference-time activation integrity monitor.

Detects adversarial inputs by measuring cross-layer safety confidence
variance. When an input's safety assessment differs significantly across
layer groups, it is flagged as potentially adversarial.

Anomaly(x, q) = Var_k[sigma(w_k^T V_s^(k)^T h_{l_k}(x, q))]

CRITICAL: Hook cleanup after every anomaly score computation.
CRITICAL: Subspace projections in fp32.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdsa.config import RDSAConfig
from rdsa.models.hooks import HookManager
from rdsa.models.model_utils import get_layer_accessor
from rdsa.subspace.identifier import SubspaceResult


class ActivationIntegrityMonitor:
    """Inference-time anomaly detection via cross-layer safety confidence variance.

    For each layer group, a lightweight linear classifier maps the safety
    subspace projection to a confidence score. The variance of these scores
    across groups indicates whether the input is adversarial.

    Args:
        model: The defended VLM.
        config: Full RDSA configuration.
        subspace_results: Pre-identified subspace results per group.
        safety_classifiers: Optional pre-trained classifiers. If ``None``,
            must be trained via ``train_safety_classifiers`` or proxy
            confidence (L2 norm) is used.
        device: Inference device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RDSAConfig,
        subspace_results: list[SubspaceResult],
        safety_classifiers: dict[int, nn.Linear] | None = None,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        self.model = model
        self.config = config
        self.device = device
        self.threshold = config.monitor.threshold
        self.conservative_mode = config.monitor.conservative_mode

        # Subspace bases (fp32 on device)
        self.V_s_list = [r.V_s.float().to(device) for r in subspace_results]
        self.rep_layers = [r.representative_layer for r in subspace_results]
        self.layer_accessor = get_layer_accessor(config.model.architecture)

        # Safety classifiers: nn.Linear(d_s, 1) per group
        self.safety_classifiers = safety_classifiers
        if self.safety_classifiers is not None:
            for _k, clf in self.safety_classifiers.items():
                clf.to(device)

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute anomaly scores for a batch of inputs.

        Args:
            input_ids: Token IDs ``[B, seq_len]``.
            attention_mask: Attention mask ``[B, seq_len]``.
            pixel_values: Optional image tensor ``[B, C, H, W]``.

        Returns:
            Per-sample anomaly scores ``[B]``.
        """
        self.model.eval()

        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values.to(self.device)

        with HookManager(
            self.model, self.rep_layers, layer_accessor=self.layer_accessor
        ) as hm:
            self.model(**forward_kwargs)
            raw = hm.get_activations()

        # Aggregate to last token
        attn = forward_kwargs["attention_mask"]
        seq_lengths = attn.sum(dim=1) - 1
        batch_indices = torch.arange(attn.size(0), device=self.device)

        confidences: list[torch.Tensor] = []
        for group_idx, layer_idx in enumerate(self.rep_layers):
            act = raw[layer_idx]  # [B, seq_len, d]
            h = act[batch_indices, seq_lengths].float()  # [B, d]

            V_s = self.V_s_list[group_idx]  # [d, d_s]
            proj = h @ V_s  # [B, d_s]

            if (
                self.safety_classifiers is not None
                and group_idx in self.safety_classifiers
            ):
                logit = self.safety_classifiers[group_idx](proj).squeeze(-1)
                conf = torch.sigmoid(logit)
            else:
                conf = torch.sigmoid(proj.norm(dim=-1))

            confidences.append(conf)

        stacked = torch.stack(confidences, dim=0)  # [G, B]
        return torch.var(stacked, dim=0)  # [B]

    @torch.no_grad()
    def is_anomalous(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if inputs are anomalous.

        Args:
            input_ids: Token IDs ``[B, seq_len]``.
            attention_mask: Attention mask ``[B, seq_len]``.
            pixel_values: Optional image tensor.

        Returns:
            ``(flags, scores)`` where ``flags`` is ``[B]`` bool tensor and
            ``scores`` is ``[B]`` float tensor.
        """
        scores = self.compute_anomaly_score(input_ids, attention_mask, pixel_values)
        flags = scores > self.threshold
        return flags, scores

    def calibrate_threshold(
        self,
        calibration_dataloader: DataLoader,
        percentile: float = 95.0,
    ) -> float:
        """Auto-calibrate the anomaly threshold from clean data.

        Sets the threshold to the given percentile of anomaly scores
        observed on clean (non-adversarial) data.

        Args:
            calibration_dataloader: DataLoader of clean inputs.
            percentile: Percentile for threshold (e.g. 95.0 means 5% FPR).

        Returns:
            The calibrated threshold value.
        """
        all_scores: list[torch.Tensor] = []

        for batch in tqdm(calibration_dataloader, desc="Calibrating threshold"):
            scores = self.compute_anomaly_score(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("pixel_values"),
            )
            all_scores.append(scores.cpu())

        all_scores_cat = torch.cat(all_scores)
        idx = int(len(all_scores_cat) * percentile / 100.0)
        sorted_scores = all_scores_cat.sort().values
        self.threshold = sorted_scores[min(idx, len(sorted_scores) - 1)].item()
        return self.threshold

    def train_safety_classifiers(
        self,
        safe_dataloader: DataLoader,
        unsafe_dataloader: DataLoader,
        num_epochs: int = 10,
        lr: float = 1e-3,
    ) -> dict[int, nn.Linear]:
        """Train linear safety classifiers for each layer group.

        Each classifier maps safety subspace projections to a binary
        safe/unsafe confidence.

        Args:
            safe_dataloader: DataLoader of safe inputs.
            unsafe_dataloader: DataLoader of unsafe inputs.
            num_epochs: Training epochs for the classifiers.
            lr: Learning rate.

        Returns:
            Dict mapping group index to trained ``nn.Linear(d_s, 1)``.
        """
        d_s = self.config.subspace.d_safe

        # Collect projections
        safe_projs = self._collect_projections(safe_dataloader)
        unsafe_projs = self._collect_projections(unsafe_dataloader)

        classifiers: dict[int, nn.Linear] = {}

        for group_idx in range(len(self.V_s_list)):
            safe_p = safe_projs[group_idx]  # [N_safe, d_s]
            unsafe_p = unsafe_projs[group_idx]  # [N_unsafe, d_s]

            X = torch.cat([safe_p, unsafe_p], dim=0).to(self.device)
            y = torch.cat(
                [
                    torch.ones(safe_p.size(0)),
                    torch.zeros(unsafe_p.size(0)),
                ]
            ).to(self.device)

            clf = nn.Linear(d_s, 1).to(self.device)
            optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            for _epoch in range(num_epochs):
                logits = clf(X).squeeze(-1)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            clf.eval()
            classifiers[group_idx] = clf

        self.safety_classifiers = classifiers
        return classifiers

    @torch.no_grad()
    def _collect_projections(
        self, dataloader: DataLoader
    ) -> dict[int, torch.Tensor]:
        """Collect safety subspace projections from a dataloader.

        Returns:
            ``{group_idx: [N, d_s]}`` on CPU.
        """
        self.model.eval()
        group_projs: dict[int, list[torch.Tensor]] = {
            i: [] for i in range(len(self.V_s_list))
        }

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)

            forward_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if pixel_values is not None:
                forward_kwargs["pixel_values"] = pixel_values

            with HookManager(
                self.model, self.rep_layers, layer_accessor=self.layer_accessor
            ) as hm:
                self.model(**forward_kwargs)
                raw = hm.get_activations()

            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(attention_mask.size(0), device=self.device)

            for group_idx, layer_idx in enumerate(self.rep_layers):
                act = raw[layer_idx]
                h = act[batch_indices, seq_lengths].float()
                V_s = self.V_s_list[group_idx]
                proj = h @ V_s
                group_projs[group_idx].append(proj.cpu())

        return {g: torch.cat(ps, dim=0) for g, ps in group_projs.items()}

    @torch.no_grad()
    def generate_with_monitoring(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        max_new_tokens: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a response with anomaly monitoring.

        If the input is flagged as anomalous and ``conservative_mode`` is
        enabled, returns a safety refusal instead of the model's generation.

        Args:
            input_ids: Token IDs ``[B, seq_len]``.
            attention_mask: Attention mask ``[B, seq_len]``.
            pixel_values: Optional image tensor.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            ``(generated_ids, anomaly_scores)`` where ``generated_ids``
            has shape ``[B, generated_seq_len]``.
        """
        flags, scores = self.is_anomalous(input_ids, attention_mask, pixel_values)

        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "max_new_tokens": max_new_tokens,
        }
        if pixel_values is not None:
            generate_kwargs["pixel_values"] = pixel_values.to(self.device)

        generated = self.model.generate(**generate_kwargs)

        if self.conservative_mode and flags.any():
            # Replace flagged samples with safety refusal
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                tokenizer = self.processor
            refusal = "I'm sorry, but I cannot assist with that request."
            refusal_ids = tokenizer.encode(refusal, return_tensors="pt").to(
                self.device
            )

            for i in range(flags.size(0)):
                if flags[i]:
                    # Pad/truncate refusal to match generated length
                    gen_len = generated.size(1)
                    ref_len = refusal_ids.size(1)
                    if ref_len < gen_len:
                        pad = torch.full(
                            (1, gen_len - ref_len),
                            tokenizer.pad_token_id or 0,
                            device=self.device,
                        )
                        refusal_padded = torch.cat([refusal_ids, pad], dim=1)
                    else:
                        refusal_padded = refusal_ids[:, :gen_len]
                    generated[i] = refusal_padded.squeeze(0)

        return generated, scores
