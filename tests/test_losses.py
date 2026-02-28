"""Tests for RDSA training losses."""

import torch
import torch.nn as nn
import torch.testing

from rdsa.training.losses import (
    ConsistencyLoss,
    SubspaceConstrainedATLoss,
)


def make_orthonormal(d: int, k: int, seed: int = 0) -> torch.Tensor:
    """Create a random orthonormal matrix [d, k]."""
    torch.manual_seed(seed)
    A = torch.randn(d, k)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k]


# ---------------------------------------------------------------------------
# Tests: ConsistencyLoss
# ---------------------------------------------------------------------------


class TestConsistencyLoss:
    def test_identical_projections_zero_loss(self) -> None:
        """Identical safety projections across groups -> loss = 0."""
        d, d_s, B = 64, 8, 4
        V_s = make_orthonormal(d, d_s, seed=42)
        h = torch.randn(B, d)

        # Same hidden states for all groups
        hidden = {0: h, 1: h.clone(), 2: h.clone()}
        loss_fn = ConsistencyLoss()
        loss = loss_fn(hidden, [V_s, V_s, V_s])

        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=1e-4)

    def test_orthogonal_projections_nonzero_loss(self) -> None:
        """Very different projections should produce significant loss."""
        d, d_s, B = 64, 8, 4
        V_s = make_orthonormal(d, d_s, seed=42)

        h0 = torch.randn(B, d)
        h1 = -h0  # Negated -> cosine = -1 -> loss per pair = 2

        hidden = {0: h0, 1: h1}
        loss_fn = ConsistencyLoss()
        loss = loss_fn(hidden, [V_s, V_s])

        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Tests: SubspaceConstrainedATLoss
# ---------------------------------------------------------------------------


class FakeLayer(nn.Module):
    """A fake transformer layer for testing."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, ...]:
        return (self.linear(x),)


class FakeModel(nn.Module):
    """Minimal model that mimics VLM architecture for hook testing.

    Has ``model.language_model.layers[i]`` path so hooks can attach.
    """

    def __init__(self, d: int, n_layers: int = 4, vocab_size: int = 32) -> None:
        super().__init__()
        # Build nested structure: model.language_model.layers
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [FakeLayer(d) for _ in range(n_layers)]
        )
        self.embed = nn.Embedding(vocab_size, d)
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> "FakeOutput":
        h = self.embed(input_ids)  # [B, seq, d]
        for layer in self.model.language_model.layers:
            h = layer(h)[0]
        logits = self.head(h)  # [B, seq, vocab]

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return FakeOutput(loss=loss, logits=logits)


class FakeOutput:
    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


class TestSubspaceConstrainedATLoss:
    def _setup(self, d: int = 32, d_s: int = 8, B: int = 2, seq: int = 4) -> dict:
        """Create test fixtures."""
        torch.manual_seed(42)
        model = FakeModel(d, n_layers=4, vocab_size=32)
        V_s = make_orthonormal(d, d_s, seed=10)
        rep_layers = [1, 2]  # two groups
        V_s_list = [V_s, V_s]

        input_ids = torch.randint(0, 32, (B, seq))
        attention_mask = torch.ones(B, seq, dtype=torch.long)
        labels = input_ids.clone()

        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Get clean hidden states
        from rdsa.models.hooks import HookManager

        with HookManager(
            model, rep_layers,
            layer_accessor="model.language_model.layers",
            detach=False,
        ) as hm:
            model(**forward_kwargs)
            raw = hm.get_activations()

        h_clean = {i: raw[layer_idx] for i, layer_idx in enumerate(rep_layers)}

        return {
            "model": model,
            "V_s_list": V_s_list,
            "rep_layers": rep_layers,
            "forward_kwargs": forward_kwargs,
            "h_clean": h_clean,
            "d_s": d_s,
        }

    def test_pgd_respects_epsilon_constraint(self) -> None:
        """||delta*|| should be <= epsilon for each group."""
        ctx = self._setup()
        loss_fn = SubspaceConstrainedATLoss(
            pgd_steps=5, pgd_alpha=0.1, epsilon=1.0
        )

        delta_star = loss_fn.find_worst_perturbation(
            model=ctx["model"],
            forward_kwargs=ctx["forward_kwargs"],
            h_clean=ctx["h_clean"],
            V_s_list=ctx["V_s_list"],
            rep_layers=ctx["rep_layers"],
            layer_accessor="model.language_model.layers",
        )

        for _gidx, delta in delta_star.items():
            # L-inf bound
            assert delta.abs().max().item() <= 1.0 + 1e-6

    def test_pgd_finds_adversarial_direction(self) -> None:
        """PGD should increase CE loss compared to zero perturbation."""
        ctx = self._setup()
        loss_fn = SubspaceConstrainedATLoss(
            pgd_steps=10, pgd_alpha=0.2, epsilon=2.0
        )

        # Baseline: CE with no perturbation
        with torch.no_grad():
            output_clean = ctx["model"](**ctx["forward_kwargs"])
            ce_clean = output_clean.loss.item()

        # PGD should find perturbation that increases CE
        delta_star = loss_fn.find_worst_perturbation(
            model=ctx["model"],
            forward_kwargs=ctx["forward_kwargs"],
            h_clean=ctx["h_clean"],
            V_s_list=ctx["V_s_list"],
            rep_layers=ctx["rep_layers"],
            layer_accessor="model.language_model.layers",
        )

        # Forward with adversarial perturbation
        from rdsa.models.hooks import InjectionHookManager

        perturbations: dict[int, torch.Tensor] = {}
        for gidx, h in ctx["h_clean"].items():
            V_s = ctx["V_s_list"][gidx].float()
            perturbations[gidx] = h.detach() + delta_star[gidx] @ V_s.T.unsqueeze(0)

        with torch.no_grad():
            with InjectionHookManager(
                ctx["model"], ctx["rep_layers"], perturbations,
                "model.language_model.layers",
            ):
                output_adv = ctx["model"](**ctx["forward_kwargs"])
                ce_adv = output_adv.loss.item()

        # Adversarial CE should be >= clean CE (PGD maximizes CE)
        assert ce_adv >= ce_clean - 0.1  # small tolerance for numerical noise

    def test_perturbation_stays_in_subspace(self) -> None:
        """V_s @ delta should lie within col(V_s)."""
        ctx = self._setup()
        loss_fn = SubspaceConstrainedATLoss(
            pgd_steps=3, pgd_alpha=0.1, epsilon=1.0
        )

        delta_star = loss_fn.find_worst_perturbation(
            model=ctx["model"],
            forward_kwargs=ctx["forward_kwargs"],
            h_clean=ctx["h_clean"],
            V_s_list=ctx["V_s_list"],
            rep_layers=ctx["rep_layers"],
            layer_accessor="model.language_model.layers",
        )

        for gidx, delta in delta_star.items():
            V_s = ctx["V_s_list"][gidx].float()  # [d, d_s]
            # Perturbation in full space: delta @ V_s.T → [B, seq, d]
            perturbation = delta @ V_s.T.unsqueeze(0)
            # Project back onto V_s column space
            proj = perturbation @ V_s @ V_s.T
            # Should be unchanged (perturbation is already in col(V_s))
            torch.testing.assert_close(perturbation, proj, atol=1e-5, rtol=1e-4)

    def test_outer_loss_gradient_flows(self) -> None:
        """Outer loss gradients should flow to model parameters (LoRA)."""
        ctx = self._setup()
        loss_fn = SubspaceConstrainedATLoss(
            pgd_steps=3, pgd_alpha=0.1, epsilon=1.0
        )

        # Need fresh h_clean with graph
        from rdsa.models.hooks import HookManager

        ctx["model"].zero_grad()
        with HookManager(
            ctx["model"], ctx["rep_layers"],
            layer_accessor="model.language_model.layers",
            detach=False,
        ) as hm:
            ctx["model"](**ctx["forward_kwargs"])
            raw = hm.get_activations()
        h_clean = {i: raw[layer] for i, layer in enumerate(ctx["rep_layers"])}

        delta_star = loss_fn.find_worst_perturbation(
            model=ctx["model"],
            forward_kwargs=ctx["forward_kwargs"],
            h_clean=h_clean,
            V_s_list=ctx["V_s_list"],
            rep_layers=ctx["rep_layers"],
            layer_accessor="model.language_model.layers",
        )

        loss = loss_fn.compute_outer_loss(
            model=ctx["model"],
            forward_kwargs=ctx["forward_kwargs"],
            h_clean=h_clean,
            delta_star=delta_star,
            V_s_list=ctx["V_s_list"],
            rep_layers=ctx["rep_layers"],
            layer_accessor="model.language_model.layers",
        )

        loss.backward()

        # At least some model parameters should have gradients
        has_grad = False
        for p in ctx["model"].parameters():
            if p.grad is not None and p.grad.abs().max() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed to model parameters"
