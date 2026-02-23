"""CLI entry point for RDSA evaluation.

Usage:
    python -m rdsa.evaluate --defense rdsa --attack scia --model qwen3vl
    python -m rdsa.evaluate --defense rdsa --attack all --model all

Can also be run directly:
    python src/rdsa/evaluate.py --defense rdsa --attack scia --model qwen3vl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the package is importable when running this file directly
# (e.g. `python src/rdsa/evaluate.py` instead of `python -m rdsa.evaluate`)
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch  # noqa: E402

logger = logging.getLogger(__name__)

ALL_ATTACKS = ["scia", "umk", "figstep", "adaptive_scia", "adaptive_pgd", "monitor_evasion"]
ALL_MODELS = ["qwen3vl", "gemma3", "llama"]
ALL_DEFENSES = ["vanilla", "rdsa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDSA Evaluation Pipeline")
    parser.add_argument(
        "--defense",
        type=str,
        default="rdsa",
        help="Defense method: 'vanilla', 'rdsa', or 'all'",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="scia",
        help="Attack method name or 'all'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3vl",
        help="Model shortname or 'all'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--advbench-path",
        type=str,
        default="data/advbench.csv",
        help="Path to AdvBench harmful prompts",
    )
    parser.add_argument(
        "--subspace-dir",
        type=str,
        default="subspaces/",
        help="Directory containing identified subspaces",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/",
        help="Directory containing RDSA-trained model checkpoints",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Model for safety judging",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Capability benchmarks to run (vqav2, mmbench, mme, orbench)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for evaluation",
    )
    return parser.parse_args()


def _get_attacks(attack_name: str) -> list[str]:
    """Expand 'all' to list of attack names."""
    if attack_name == "all":
        return list(ALL_ATTACKS)
    return [attack_name]


def _get_models(model_name: str) -> list[str]:
    """Expand 'all' to list of model names."""
    if model_name == "all":
        return list(ALL_MODELS)
    return [model_name]


def _get_defenses(defense_name: str) -> list[str]:
    """Expand 'all' to list of defense names."""
    if defense_name == "all":
        return list(ALL_DEFENSES)
    return [defense_name]


def evaluate_single(
    model_name: str,
    defense_name: str,
    attack_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Run a single evaluation: one model x one defense x one attack.

    Args:
        model_name: Model shortname.
        defense_name: Defense method name.
        attack_name: Attack method name.
        args: Parsed CLI arguments.
        device: Computation device.

    Returns:
        Dict of metric names to values.
    """
    from rdsa.identify import MODEL_CONFIGS

    model_dict = MODEL_CONFIGS[model_name]
    from rdsa.config import ModelConfig

    model_cfg = ModelConfig(
        name=str(model_dict["name"]),
        architecture=str(model_dict["architecture"]),
        hidden_dim=int(model_dict["hidden_dim"]),  # type: ignore[arg-type]
        num_layers=int(model_dict["num_layers"]),  # type: ignore[arg-type]
        layer_groups=list(model_dict["layer_groups"]),  # type: ignore[arg-type]
    )

    logger.info(
        "Evaluating: model=%s, defense=%s, attack=%s",
        model_name,
        defense_name,
        attack_name,
    )

    # Load model
    from rdsa.models.model_utils import load_model_and_processor

    model, processor = load_model_and_processor(model_cfg, device=device)

    # Apply defense if RDSA
    if defense_name == "rdsa":
        checkpoint_dir = Path(args.checkpoint_dir) / model_cfg.architecture
        if checkpoint_dir.exists():
            from peft import PeftModel

            lora_dir = _find_latest_lora(checkpoint_dir)
            if lora_dir:
                logger.info("Loading LoRA weights from %s", lora_dir)
                model = PeftModel.from_pretrained(model, str(lora_dir))

    # Load harmful prompts
    from rdsa.training.data import SafetyEvalDataset

    eval_dataset = SafetyEvalDataset(
        data_path=args.advbench_path,
        processor=processor,
    )
    prompts = eval_dataset.instructions[: args.max_samples]

    # Generate responses (with or without attack)
    logger.info("Generating responses for %d prompts...", len(prompts))
    responses = _generate_responses(model, processor, prompts, device)

    # Judge responses
    from rdsa.evaluation.judge import GPT4oSafetyJudge
    from rdsa.evaluation.metrics import compute_all_metrics

    judge = GPT4oSafetyJudge(model_name=args.judge_model)
    judgments = judge.judge_batch(prompts, responses)
    metrics = compute_all_metrics(judgments, responses)

    logger.info("Results: %s", metrics)
    return metrics


def _generate_responses(
    model: torch.nn.Module,
    processor: object,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate model responses for a list of prompts.

    Args:
        model: The VLM.
        processor: HuggingFace processor.
        prompts: List of text prompts.
        device: Computation device.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        List of generated response strings.
    """
    from tqdm import tqdm

    tokenizer = getattr(processor, "tokenizer", processor)
    model.eval()
    responses = []

    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(text)

    return responses


def _find_latest_lora(checkpoint_dir: Path) -> Path | None:
    """Find the latest LoRA checkpoint in the given directory."""
    lora_dirs = sorted(checkpoint_dir.glob("epoch_*/lora_weights"), reverse=True)
    if lora_dirs:
        return lora_dirs[0]
    # Check for direct lora_weights dir
    direct = checkpoint_dir / "lora_weights"
    if direct.exists():
        return direct
    return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _get_models(args.model)
    defenses = _get_defenses(args.defense)
    attacks = _get_attacks(args.attack)

    all_results: dict[str, dict[str, float]] = {}

    for model_name in models:
        for defense_name in defenses:
            for attack_name in attacks:
                key = f"{model_name}/{defense_name}/{attack_name}"
                try:
                    metrics = evaluate_single(
                        model_name, defense_name, attack_name, args, device
                    )
                    all_results[key] = metrics
                except Exception:
                    logger.exception("Failed: %s", key)
                    all_results[key] = {"error": 1.0}

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Results saved to %s", results_file)

    # Print summary table
    logger.info("\n=== Evaluation Summary ===")
    for key, metrics in sorted(all_results.items()):
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("  %s: %s", key, metrics_str)


if __name__ == "__main__":
    main()
