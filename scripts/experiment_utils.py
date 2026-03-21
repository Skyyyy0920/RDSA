"""Shared utilities for all RDSA NeurIPS experiment scripts.

Provides:
- Common config maps and constants
- run_training / run_evaluation helpers (subprocess-based)
- Multi-seed runner for statistical significance
- Results I/O (JSON + LaTeX table generation)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Model shortnames → config paths ──────────────────────────────────

CONFIG_MAP: dict[str, str] = {
    "qwen3vl": "configs/qwen3vl.yaml",
    "gemma3": "configs/gemma3.yaml",
    "llama": "configs/llama32.yaml",
    "internvl2": "configs/internvl2.yaml",
    "minicpm_v": "configs/minicpm_v.yaml",
}

# Default surrogate models (open-access, no login required)
SURROGATE_MODELS = ["qwen3vl", "internvl2", "minicpm_v"]

VICTIM_MODELS: dict[str, str] = {
    "qwen3vl_30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "gemma3_27b": "google/gemma-3-27b-it",
    "llama_11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

COMMERCIAL_MODELS = ["gpt-4o", "gemini-2.5-flash", "claude-sonnet-4-5"]

ALL_ATTACKS = [
    "scia", "umk", "figstep", "mm_safety",
    "adaptive_scia", "adaptive_pgd", "monitor_evasion",
]

BASELINE_DEFENSES = [
    "vanilla", "safety_sft", "circuit_breaker", "lat", "smoothvlm", "vlguard",
]

SEEDS = [42, 123, 456]


# ── Subprocess runners ────────────────────────────────────────────────

def run_training(
    config_path: str,
    overrides: dict[str, Any],
    output_dir: str,
    dry_run: bool = False,
) -> bool:
    """Launch an RDSA training run via subprocess.

    Args:
        config_path: Path to base YAML config.
        overrides: Hydra-style key=value overrides.
        output_dir: Where to save checkpoints.
        dry_run: Print command without executing.

    Returns:
        True if training succeeded (or dry_run).
    """
    cmd = [
        sys.executable, "-m", "rdsa.train",
        "--config", config_path,
    ]
    for key, value in overrides.items():
        cmd.append(f"{key}={value}")
    cmd.append(f"output.save_dir={output_dir}")

    logger.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return True

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Training completed: %s", output_dir)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Training failed: %s", e.stderr[-500:] if e.stderr else "")
        return False


def run_evaluation(
    model: str,
    defense: str,
    attack: str,
    checkpoint_dir: str,
    output_dir: str,
    max_samples: int = 100,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, float]:
    """Launch an RDSA evaluation via subprocess and parse results.

    Args:
        model: Model shortname.
        defense: Defense method name.
        attack: Attack method name.
        checkpoint_dir: Path to trained checkpoint.
        output_dir: Where to save results.
        max_samples: Number of evaluation samples.
        dry_run: Print command without executing.
        extra_args: Additional CLI arguments.

    Returns:
        Metrics dict (empty on failure or dry_run).
    """
    cmd = [
        sys.executable, "-m", "rdsa.evaluate",
        "--model", model,
        "--defense", defense,
        "--attack", attack,
        "--checkpoint-dir", checkpoint_dir,
        "--output-dir", output_dir,
        "--max-samples", str(max_samples),
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return {}

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Try to load results
        results_file = Path(output_dir) / "evaluation_results.json"
        if results_file.exists():
            with open(results_file, encoding="utf-8") as f:
                all_results = json.load(f)
            key = f"{model}/{defense}/{attack}"
            return all_results.get(key, {})
        return {"status": "completed"}
    except subprocess.CalledProcessError as e:
        logger.error("Evaluation failed: %s", e.stderr[-300:] if e.stderr else "")
        return {"error": 1.0}


def run_benchmark(
    model: str,
    defense: str,
    checkpoint_dir: str,
    benchmarks: list[str],
    output_dir: str,
    dry_run: bool = False,
) -> dict[str, float]:
    """Run capability benchmarks on a model.

    Args:
        model: Model shortname.
        defense: Defense method.
        checkpoint_dir: Path to checkpoint.
        benchmarks: List of benchmark names.
        output_dir: Where to save results.
        dry_run: Print command without executing.

    Returns:
        Benchmark metrics dict.
    """
    cmd = [
        sys.executable, "-m", "rdsa.evaluate",
        "--model", model,
        "--defense", defense,
        "--checkpoint-dir", checkpoint_dir,
        "--output-dir", output_dir,
        "--attack", "none",
        "--benchmarks", *benchmarks,
    ]

    logger.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return {}

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {"status": "completed"}
    except subprocess.CalledProcessError as e:
        logger.error("Benchmark failed: %s", e.stderr[-300:] if e.stderr else "")
        return {"error": 1.0}


# ── Multi-seed runner ────────────────────────────────────────────────

def run_with_seeds(
    run_fn: Any,
    seeds: list[int] | None = None,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Run an experiment function with multiple seeds.

    Args:
        run_fn: Callable that accepts a ``seed`` kwarg and returns metrics.
        seeds: List of random seeds. Defaults to SEEDS.
        **kwargs: Additional arguments passed to run_fn.

    Returns:
        List of metrics dicts, one per seed.
    """
    if seeds is None:
        seeds = SEEDS
    results = []
    for seed in seeds:
        logger.info("=== Seed %d ===", seed)
        metrics = run_fn(seed=seed, **kwargs)
        results.append(metrics)
    return results


def aggregate_seeds(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute mean and std across seed runs.

    Args:
        results: List of metrics dicts from different seeds.

    Returns:
        Dict mapping metric_name -> {"mean": ..., "std": ...}.
    """
    import numpy as np

    if not results:
        return {}

    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    aggregated: dict[str, dict[str, float]] = {}
    for key in sorted(all_keys):
        values = [r.get(key, float("nan")) for r in results]
        values = [v for v in values if not (isinstance(v, float) and v != v)]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return aggregated


# ── Results I/O ──────────────────────────────────────────────────────

def save_results(results: dict[str, Any], output_path: str) -> None:
    """Save results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", path)


def load_results(path: str) -> dict[str, Any]:
    """Load results from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def results_to_latex(
    results: dict[str, dict[str, dict[str, float]]],
    metric: str = "asr",
    caption: str = "",
    label: str = "",
) -> str:
    """Convert results dict to a LaTeX table string.

    Args:
        results: Nested dict: {method: {attack: {"mean": x, "std": y}}}.
        metric: Which metric to display.
        caption: LaTeX table caption.
        label: LaTeX table label.

    Returns:
        LaTeX table string.
    """
    attacks = sorted(
        {atk for method_results in results.values() for atk in method_results}
    )
    methods = list(results.keys())

    header = " & ".join(["Method"] + [a.replace("_", r"\_") for a in attacks])

    rows: list[str] = []
    for method in methods:
        cells = [method.replace("_", r"\_")]
        for attack in attacks:
            data = results.get(method, {}).get(attack, {})
            if isinstance(data, dict) and "mean" in data:
                mean = data["mean"] * 100
                std = data["std"] * 100
                cells.append(f"{mean:.1f}$\\pm${std:.1f}")
            elif isinstance(data, (int, float)):
                cells.append(f"{data * 100:.1f}")
            else:
                cells.append("-")
        rows.append(" & ".join(cells) + r" \\")

    table = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{'l' + 'c' * len(attacks)}}}\n"
        r"\toprule" + "\n"
        f"{header} \\\\\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    return table


# ── Logging setup ─────────────────────────────────────────────────────

def setup_logging(log_file: str | None = None) -> None:
    """Configure logging for experiment scripts."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )
