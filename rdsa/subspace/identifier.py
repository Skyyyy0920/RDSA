"""Safety subspace identification via SVD and semantic subspace via PCA.

Implements Step 1 of the RDSA training pipeline:
1. Collect activations from safe/unsafe contrast pairs
2. SVD on activation differences -> V_s (safety subspace) per layer group
3. PCA on normal data activations -> V_t (semantic subspace) per layer group

CRITICAL: SVD numerical stability — use torch.linalg.svd(full_matrices=False).
For large N, compute on CPU then move result to GPU.
CRITICAL: V_s and V_t must always be fp32.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdsa.config import RDSAConfig
from rdsa.models.hooks import HookManager, get_representative_layer_indices


@dataclass
class SubspaceResult:
    """Result of subspace identification for one layer group.

    Attributes:
        V_s: Safety subspace basis, shape ``[d, d_s]``, always fp32.
        V_t: Semantic subspace basis, shape ``[d, d_t]``, always fp32.
        singular_values: Top ``d_s`` singular values from SVD on activation diffs.
        explained_variance: Top ``d_t`` PCA explained variance ratios.
        layer_group_idx: Index of the layer group.
        representative_layer: The middle layer index used for this group.
    """

    V_s: torch.Tensor
    V_t: torch.Tensor
    singular_values: torch.Tensor
    explained_variance: torch.Tensor
    layer_group_idx: int
    representative_layer: int


class SafetySubspaceIdentifier:
    """Identifies safety and semantic subspaces for each layer group.

    The safety subspace ``V_s`` captures directions that differentiate safe
    and unsafe model behaviour (via SVD on activation differences). The
    semantic subspace ``V_t`` captures directions of maximum variance in
    normal model activations (via PCA).

    Args:
        model: The pretrained VLM.
        config: Full RDSA configuration.
        device: Device for intermediate computation. SVD is always on CPU.
        layer_accessor: Dotted path to the model's transformer layers.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RDSAConfig,
        device: torch.device | None = None,
        layer_accessor: str = "model.language_model.layers",
    ) -> None:
        if device is None:
            device = torch.device("cpu")
        self._model = model
        self._config = config
        self._device = device
        self._layer_accessor = layer_accessor
        self._layer_groups = config.model.layer_groups
        self._rep_layers = get_representative_layer_indices(self._layer_groups)
        self._d_s = config.subspace.d_safe
        self._d_t = config.subspace.d_semantic

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_contrast_activations(
        self,
        safe_dataloader: DataLoader,
        unsafe_dataloader: DataLoader,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Collect activations for safe and unsafe inputs per layer group.

        Activations are collected on CPU to avoid GPU OOM.

        Args:
            safe_dataloader: DataLoader yielding safe input batches with keys
                ``"input_ids"``, ``"attention_mask"``, and optionally
                ``"pixel_values"``.
            unsafe_dataloader: DataLoader yielding unsafe input batches.

        Returns:
            Dict mapping group index to ``(safe_activations, unsafe_activations)``
            each of shape ``[N, d]`` on CPU, fp32.
        """
        self._model.eval()

        safe_acts = self._collect_activations(safe_dataloader)
        unsafe_acts = self._collect_activations(unsafe_dataloader)

        result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for group_idx in range(len(self._layer_groups)):
            result[group_idx] = (safe_acts[group_idx], unsafe_acts[group_idx])

        return result

    @torch.no_grad()
    def collect_normal_activations(
        self,
        dataloader: DataLoader,
    ) -> dict[int, torch.Tensor]:
        """Collect activations from normal (benign) data per layer group.

        Args:
            dataloader: DataLoader yielding normal input batches.

        Returns:
            Dict mapping group index to activations ``[N, d]`` on CPU, fp32.
        """
        self._model.eval()
        return self._collect_activations(dataloader)

    def _collect_activations(
        self,
        dataloader: DataLoader,
    ) -> dict[int, torch.Tensor]:
        """Internal helper: collect last-token activations from each group's
        representative layer.

        Returns:
            Dict mapping group index to ``[N, d]`` tensor on CPU, fp32.
        """
        group_acts: dict[int, list[torch.Tensor]] = {
            i: [] for i in range(len(self._layer_groups))
        }

        for batch in tqdm(dataloader, desc="Collecting activations", leave=False):
            input_ids = batch["input_ids"].to(self._device)
            attention_mask = batch["attention_mask"].to(self._device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self._device)

            forward_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if pixel_values is not None:
                forward_kwargs["pixel_values"] = pixel_values

            with HookManager(
                self._model,
                self._rep_layers,
                layer_accessor=self._layer_accessor,
            ) as hm:
                self._model(**forward_kwargs)
                raw = hm.get_activations()

            for group_idx, layer_idx in enumerate(self._rep_layers):
                act = raw[layer_idx]  # [B, seq_len, d]
                # Last-token aggregation
                seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
                batch_indices = torch.arange(act.size(0), device=act.device)
                last_token = act[batch_indices, seq_lengths]  # [B, d]
                group_acts[group_idx].append(last_token.float().cpu())

        return {
            g: torch.cat(acts, dim=0) for g, acts in group_acts.items()
        }

    # ------------------------------------------------------------------
    # Subspace identification
    # ------------------------------------------------------------------

    def identify_safety_subspace(
        self,
        safe_activations: torch.Tensor,
        unsafe_activations: torch.Tensor,
        d_s: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SVD on activation differences to find safety subspace.

        Computes ``ΔH = unsafe - safe``, then performs truncated SVD to extract
        the top ``d_s`` right singular vectors as the safety subspace basis.

        CRITICAL: SVD computed on CPU for numerical stability.

        Args:
            safe_activations: Safe activation matrix ``[N, d]``, fp32.
            unsafe_activations: Unsafe activation matrix ``[N, d]``, fp32.
            d_s: Safety subspace dimension. Defaults to ``config.subspace.d_safe``.

        Returns:
            ``(V_s, singular_values)`` where ``V_s`` has shape ``[d, d_s]``
            and ``singular_values`` has shape ``[d_s]``. Both fp32 on CPU.
        """
        if d_s is None:
            d_s = self._d_s

        # Activation difference matrix
        delta_h = (unsafe_activations - safe_activations).float().cpu()  # [N, d]

        # SVD on CPU for numerical stability
        # delta_h: [N, d] -> U: [N, min(N,d)], S: [min(N,d)], Vh: [min(N,d), d]
        _U, S, Vh = torch.linalg.svd(delta_h, full_matrices=False)

        # Top d_s right singular vectors as safety subspace basis
        V_s = Vh[:d_s, :].T.contiguous()  # [d, d_s]
        singular_values = S[:d_s].contiguous()  # [d_s]

        return V_s.float(), singular_values.float()

    def identify_semantic_subspace(
        self,
        normal_activations: torch.Tensor,
        d_t: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PCA on normal activations to find semantic subspace.

        Centers the data and uses SVD (more numerically stable than explicit
        covariance matrix) to extract the top ``d_t`` principal components.

        CRITICAL: Computed on CPU. Result is always fp32.

        Args:
            normal_activations: Normal activation matrix ``[N, d]``, fp32.
            d_t: Semantic subspace dimension. Defaults to ``config.subspace.d_semantic``.

        Returns:
            ``(V_t, explained_variance_ratios)`` where ``V_t`` has shape
            ``[d, d_t]`` and ratios have shape ``[d_t]``. Both fp32 on CPU.
        """
        if d_t is None:
            d_t = self._d_t

        H = normal_activations.float().cpu()  # [N, d]

        # Center the data
        H_centered = H - H.mean(dim=0, keepdim=True)

        # SVD for PCA (more stable than covariance eigendecomposition)
        _U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)

        # Top d_t principal components
        V_t = Vh[:d_t, :].T.contiguous()  # [d, d_t]

        # Explained variance ratios
        total_var = (S**2).sum()
        explained = (S[:d_t] ** 2) / total_var.clamp(min=1e-10)

        return V_t.float(), explained.float()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def identify_all_groups(
        self,
        safe_dataloader: DataLoader,
        unsafe_dataloader: DataLoader,
        normal_dataloader: DataLoader,
    ) -> list[SubspaceResult]:
        """Run full identification pipeline for all layer groups.

        For each group:
        1. Extract contrast activations and compute V_s via SVD
        2. Extract normal activations and compute V_t via PCA

        Memory note: processes one group at a time, releasing previous
        group's raw data before proceeding.

        Args:
            safe_dataloader: DataLoader for safe contrast inputs.
            unsafe_dataloader: DataLoader for unsafe contrast inputs.
            normal_dataloader: DataLoader for normal (benign) inputs.

        Returns:
            List of ``SubspaceResult``, one per layer group.
        """
        # Collect all activations first (one pass per dataset)
        contrast_acts = self.collect_contrast_activations(
            safe_dataloader, unsafe_dataloader
        )
        normal_acts = self.collect_normal_activations(normal_dataloader)

        results: list[SubspaceResult] = []

        for group_idx in range(len(self._layer_groups)):
            safe_act, unsafe_act = contrast_acts[group_idx]
            norm_act = normal_acts[group_idx]

            V_s, singular_values = self.identify_safety_subspace(safe_act, unsafe_act)
            V_t, explained_var = self.identify_semantic_subspace(norm_act)

            results.append(
                SubspaceResult(
                    V_s=V_s,
                    V_t=V_t,
                    singular_values=singular_values,
                    explained_variance=explained_var,
                    layer_group_idx=group_idx,
                    representative_layer=self._rep_layers[group_idx],
                )
            )

            # Free raw activations for this group to save memory
            del safe_act, unsafe_act, norm_act

        return results

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_subspaces(results: list[SubspaceResult], save_dir: str) -> None:
        """Save subspace bases and metadata to disk.

        Creates one ``.pt`` file per group plus a ``metadata.pt`` summary.

        Args:
            results: List of ``SubspaceResult`` to save.
            save_dir: Directory to save into (created if needed).
        """
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        for r in results:
            group_data = {
                "V_s": r.V_s,
                "V_t": r.V_t,
                "singular_values": r.singular_values,
                "explained_variance": r.explained_variance,
                "layer_group_idx": r.layer_group_idx,
                "representative_layer": r.representative_layer,
            }
            torch.save(group_data, path / f"group_{r.layer_group_idx}.pt")

        metadata = {
            "num_groups": len(results),
            "d_s": results[0].V_s.shape[1] if results else 0,
            "d_t": results[0].V_t.shape[1] if results else 0,
        }
        torch.save(metadata, path / "metadata.pt")

    @staticmethod
    def load_subspaces(save_dir: str) -> list[SubspaceResult]:
        """Load previously saved subspace bases.

        Args:
            save_dir: Directory containing saved subspace files.

        Returns:
            List of ``SubspaceResult``, ordered by group index.
        """
        path = Path(save_dir)
        metadata = torch.load(path / "metadata.pt", weights_only=True)

        results: list[SubspaceResult] = []
        for i in range(metadata["num_groups"]):
            data = torch.load(path / f"group_{i}.pt", weights_only=True)
            results.append(
                SubspaceResult(
                    V_s=data["V_s"].float(),
                    V_t=data["V_t"].float(),
                    singular_values=data["singular_values"].float(),
                    explained_variance=data["explained_variance"].float(),
                    layer_group_idx=data["layer_group_idx"],
                    representative_layer=data["representative_layer"],
                )
            )

        return results
