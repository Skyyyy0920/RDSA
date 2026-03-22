"""CLI entry point for RDSA evaluation.

Runs attacks on VLMs (with or without RDSA defense) and measures ASR/RR/OR
using GPT-4o as safety judge.

Supported attacks:
- none: No attack (clean prompts, measures baseline refusal rate)
- scia: Safety Circuit Intervention Attack (neuron suppression via adversarial images)
- umk: Universal Model-Knowledge PGD (white-box, maximize harmful output)
- figstep: Typography-based visual prompt injection (no gradient)
- adaptive_scia: SCIA + anti-entanglement objective (knows V_s, V_t)
- adaptive_pgd: Multi-layer simultaneous PGD (knows all group V_s)
- monitor_evasion: PGD with cross-layer variance constraint (knows monitor)

Usage:
    python -m rdsa.evaluate --defense vanilla --attack none --model qwen3vl
    python -m rdsa.evaluate --defense rdsa --attack scia --model qwen3vl
    python -m rdsa.evaluate --defense rdsa --attack all --model all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

ALL_ATTACKS = [
    "none", "scia", "umk", "figstep",
    "adaptive_scia", "adaptive_pgd", "monitor_evasion",
]
ALL_MODELS = ["qwen3vl", "internvl2", "minicpm_v", "gemma3", "llama"]
ALL_DEFENSES = ["vanilla", "rdsa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDSA Evaluation Pipeline")
    parser.add_argument("--defense", type=str, default="rdsa",
                        help="Defense method: 'vanilla', 'rdsa', or 'all'")
    parser.add_argument("--attack", type=str, default="scia",
                        help="Attack method name or 'all'")
    parser.add_argument("--model", type=str, default="qwen3vl",
                        help="Model shortname or 'all'")
    parser.add_argument("--output-dir", type=str, default="results/",
                        help="Directory for evaluation results")
    parser.add_argument("--advbench-path", type=str,
                        default="data/advbench.csv",
                        help="Path to AdvBench harmful prompts")
    parser.add_argument("--subspace-dir", type=str, default="subspaces/",
                        help="Directory containing identified subspaces")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/",
                        help="Directory containing RDSA-trained model checkpoints")
    parser.add_argument("--safe-data", type=str,
                        default="data/safe_prompts.jsonl",
                        help="Safe prompts for SCIA neuron identification")
    parser.add_argument("--unsafe-data", type=str,
                        default="data/unsafe_prompts.jsonl",
                        help="Unsafe prompts for SCIA neuron identification")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum samples to evaluate")
    parser.add_argument("--judge-model", type=str, default="gpt-4o",
                        help="Model for safety judging")
    parser.add_argument("--benchmarks", nargs="*", default=None,
                        help="Capability benchmarks to run")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for evaluation")
    return parser.parse_args()


def _get_attacks(attack_name: str) -> list[str]:
    if attack_name == "all":
        return list(ALL_ATTACKS)
    return [attack_name]


def _get_models(model_name: str) -> list[str]:
    if model_name == "all":
        return list(ALL_MODELS)
    return [model_name]


def _get_defenses(defense_name: str) -> list[str]:
    if defense_name == "all":
        return list(ALL_DEFENSES)
    return [defense_name]


# ── Attack execution ──────────────────────────────────────────────────


def _create_blank_image(size: int = 448) -> Image.Image:
    """Create a blank white image as a neutral starting point for attacks."""
    return Image.new("RGB", (size, size), (255, 255, 255))


def _run_attack(
    attack_name: str,
    model: nn.Module,
    processor: object,
    prompts: list[str],
    device: torch.device,
    args: argparse.Namespace,
    model_name: str,
) -> list[tuple[str, Image.Image | None]]:
    """Execute an attack and return (prompt, adversarial_image) pairs.

    For text-only attacks or no attack, image is None.
    For visual attacks, image is a PIL Image with adversarial perturbation.

    Args:
        attack_name: Attack identifier.
        model: The target VLM.
        processor: HuggingFace processor.
        prompts: List of harmful prompts.
        device: Computation device.
        args: CLI arguments (for data paths, subspace dirs, etc.)
        model_name: Model shortname (for loading subspaces).

    Returns:
        List of (prompt_text, image_or_None) tuples.
    """
    if attack_name == "none":
        # No attack — plain text prompts
        return [(p, None) for p in prompts]

    elif attack_name == "figstep":
        # FigStep: renders harmful text into images, pairs with benign prompt
        from rdsa.attacks.baselines import FigStepAttack

        attack = FigStepAttack()
        samples = attack.generate_attack_samples(prompts)
        return [(s.prompt, s.image) for s in samples]

    elif attack_name == "scia":
        # SCIA: identify safety neurons, then suppress via adversarial images
        from rdsa.attacks.scia import SCIAAttack

        attack = SCIAAttack(model=model, processor=processor, device=device)

        # Identify safety neurons from contrast data
        safe_dl, unsafe_dl = _load_contrast_dataloaders(processor, args)
        logger.info("Identifying safety neurons for SCIA...")
        attack.identify_safety_neurons(safe_dl, unsafe_dl)

        results = []
        for prompt in prompts:
            clean_image = _create_blank_image()
            adv_image = attack.generate_adversarial_image(clean_image, prompt)
            results.append((prompt, adv_image))
        return results

    elif attack_name == "umk":
        # UMK: white-box PGD on image to maximize harmful response
        from rdsa.attacks.umk import UMKAttack

        attack = UMKAttack(model=model, processor=processor, device=device)

        results = []
        for prompt in prompts:
            clean_image = _create_blank_image()
            adv_image = attack.attack(clean_image, prompt)
            results.append((prompt, adv_image))
        return results

    elif attack_name == "adaptive_scia":
        from rdsa.attacks.adaptive import AdaptiveSCIA

        subspace_results = _load_subspaces(model_name, args, device)
        attack = AdaptiveSCIA(
            model=model, processor=processor,
            subspace_results=subspace_results, device=device,
        )

        safe_dl, unsafe_dl = _load_contrast_dataloaders(processor, args)
        attack.identify_safety_neurons(safe_dl, unsafe_dl)

        results = []
        for prompt in prompts:
            clean_image = _create_blank_image()
            adv_image = attack.generate_adversarial_image(clean_image, prompt)
            results.append((prompt, adv_image))
        return results

    elif attack_name == "adaptive_pgd":
        from rdsa.attacks.adaptive import AdaptivePGD

        subspace_results = _load_subspaces(model_name, args, device)
        attack = AdaptivePGD(
            model=model, processor=processor,
            subspace_results=subspace_results, device=device,
        )

        results = []
        for prompt in prompts:
            clean_image = _create_blank_image()
            adv_image = attack.attack(clean_image, prompt)
            results.append((prompt, adv_image))
        return results

    elif attack_name == "monitor_evasion":
        from rdsa.attacks.adaptive import MonitorEvasion

        subspace_results = _load_subspaces(model_name, args, device)
        attack = MonitorEvasion(
            model=model, processor=processor,
            subspace_results=subspace_results, device=device,
        )

        results = []
        for prompt in prompts:
            clean_image = _create_blank_image()
            adv_image = attack.attack(clean_image, prompt)
            results.append((prompt, adv_image))
        return results

    else:
        logger.warning("Unknown attack '%s', falling back to no attack", attack_name)
        return [(p, None) for p in prompts]


def _load_contrast_dataloaders(
    processor: object,
    args: argparse.Namespace,
) -> tuple:
    """Load safe/unsafe dataloaders for SCIA neuron identification."""
    from rdsa.training.data import create_contrast_dataloaders

    return create_contrast_dataloaders(
        safe_path=args.safe_data,
        unsafe_path=args.unsafe_data,
        processor=processor,
        batch_size=8,
    )


def _load_subspaces(
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
):
    """Load pre-identified subspace results for adaptive attacks."""
    from rdsa.identify import MODEL_CONFIGS
    from rdsa.subspace.identifier import SafetySubspaceIdentifier

    model_dict = MODEL_CONFIGS.get(model_name, {})
    subspace_dir = model_dict.get("subspace_dir", f"subspaces/{model_name}")

    # Override with CLI arg if provided
    if args.subspace_dir != "subspaces/":
        subspace_dir = args.subspace_dir

    subspace_path = Path(subspace_dir)
    if not (subspace_path / "metadata.pt").exists():
        raise FileNotFoundError(
            f"Subspace files not found at {subspace_path}. "
            f"Run `python -m rdsa.identify --model {model_name}` first."
        )

    results = SafetySubspaceIdentifier.load_subspaces(str(subspace_path))
    # Move to device
    for r in results:
        r.V_s = r.V_s.float().to(device)
        r.V_t = r.V_t.float().to(device)
    return results


# ── Response generation ───────────────────────────────────────────────


def _generate_responses(
    model: nn.Module,
    processor: object,
    attack_results: list[tuple[str, Image.Image | None]],
    device: torch.device,
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate model responses for prompts, optionally with adversarial images.

    Args:
        model: The VLM.
        processor: HuggingFace processor (handles both text and images).
        attack_results: List of (prompt, image_or_None) from _run_attack.
        device: Computation device.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        List of generated response strings.
    """
    from tqdm import tqdm

    tokenizer = getattr(processor, "tokenizer", processor)
    model.eval()
    responses = []

    for prompt, image in tqdm(attack_results, desc="Generating"):
        try:
            if image is not None:
                # Multimodal input: text + image
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
            else:
                # Text-only input
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

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

        except Exception as e:
            logger.warning("Generation failed for prompt: %s", str(e)[:100])
            responses.append("")

    return responses


# ── Main evaluation logic ─────────────────────────────────────────────


def evaluate_single(
    model_name: str,
    defense_name: str,
    attack_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Run a single evaluation: one model × one defense × one attack.

    Args:
        model_name: Model shortname.
        defense_name: Defense method name.
        attack_name: Attack method name.
        args: Parsed CLI arguments.
        device: Computation device.

    Returns:
        Dict of metric names to values.
    """
    from rdsa.config import ModelConfig
    from rdsa.identify import MODEL_CONFIGS

    model_dict = MODEL_CONFIGS[model_name]
    model_cfg = ModelConfig(
        name=str(model_dict["name"]),
        architecture=str(model_dict["architecture"]),
        hidden_dim=int(model_dict["hidden_dim"]),  # type: ignore[arg-type]
        num_layers=int(model_dict["num_layers"]),  # type: ignore[arg-type]
        layer_groups=list(model_dict["layer_groups"]),  # type: ignore[arg-type]
    )

    logger.info(
        "=== Evaluating: model=%s, defense=%s, attack=%s ===",
        model_name, defense_name, attack_name,
    )

    # ── Step 1: Load model ──
    from rdsa.models.model_utils import load_model_and_processor

    model, processor = load_model_and_processor(model_cfg, device=device)

    # ── Step 2: Apply defense (load LoRA weights) ──
    if defense_name == "rdsa":
        checkpoint_dir = Path(args.checkpoint_dir)
        # Try multiple checkpoint locations
        for candidate in [
            checkpoint_dir / model_cfg.architecture,
            checkpoint_dir,
        ]:
            if candidate.exists():
                lora_dir = _find_latest_lora(candidate)
                if lora_dir:
                    from peft import PeftModel
                    logger.info("Loading LoRA weights from %s", lora_dir)
                    model = PeftModel.from_pretrained(model, str(lora_dir))
                    break
        else:
            logger.warning(
                "No RDSA checkpoint found at %s — using vanilla model",
                args.checkpoint_dir,
            )

    # ── Step 3: Load harmful prompts ──
    from rdsa.training.data import SafetyEvalDataset

    eval_dataset = SafetyEvalDataset(
        data_path=args.advbench_path,
        processor=processor,
    )
    prompts = eval_dataset.instructions[: args.max_samples]
    logger.info("Loaded %d harmful prompts", len(prompts))

    # ── Step 4: Run attack ──
    logger.info("Running attack: %s", attack_name)
    attack_results = _run_attack(
        attack_name=attack_name,
        model=model,
        processor=processor,
        prompts=prompts,
        device=device,
        args=args,
        model_name=model_name,
    )

    # ── Step 5: Generate responses ──
    logger.info("Generating responses for %d samples...", len(attack_results))
    # Extract prompts (may be modified by attack, e.g. FigStep uses benign prefix)
    eval_prompts = [r[0] for r in attack_results]
    responses = _generate_responses(model, processor, attack_results, device)

    # ── Step 6: Judge responses ──
    from rdsa.evaluation.judge import GPT4oSafetyJudge
    from rdsa.evaluation.metrics import compute_all_metrics

    logger.info("Judging %d responses with %s...", len(responses), args.judge_model)
    judge = GPT4oSafetyJudge(model_name=args.judge_model)
    judgments = judge.judge_batch(eval_prompts, responses)
    metrics = compute_all_metrics(judgments, responses)

    logger.info("Results: %s", metrics)
    return metrics


def _find_latest_lora(checkpoint_dir: Path) -> Path | None:
    """Find the latest LoRA checkpoint in the given directory."""
    lora_dirs = sorted(checkpoint_dir.glob("epoch_*/lora_weights"), reverse=True)
    if lora_dirs:
        return lora_dirs[0]
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
