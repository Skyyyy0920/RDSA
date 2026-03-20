"""Prepare safe/unsafe contrast data for RDSA subspace identification.

Sources:
- Unsafe: AdvBench (local CSV) + StrongReject (HuggingFace)
- Safe: Alpaca-cleaned (HuggingFace) + hand-crafted benign prompts

Output: JSONL files with {"prompt": "..."} per line.

Usage:
    python scripts/prepare_contrast_data.py
    python scripts/prepare_contrast_data.py --n-samples 2000
    python scripts/prepare_contrast_data.py --unsafe-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Diverse benign prompt templates for safe data generation
SAFE_PROMPT_TEMPLATES = [
    # General knowledge
    "What is {topic}?",
    "Explain {topic} in simple terms.",
    "How does {topic} work?",
    "What are the main characteristics of {topic}?",
    "Describe the history of {topic}.",
    "What is the difference between {a} and {b}?",
    # Science
    "Why is the sky blue?",
    "How do plants produce oxygen?",
    "What causes earthquakes?",
    "Explain how vaccines work.",
    "What is photosynthesis?",
    "How does gravity affect objects?",
    # Math
    "What is the Pythagorean theorem?",
    "Explain the concept of derivatives in calculus.",
    "How do you calculate the area of a circle?",
    "What are prime numbers?",
    # Everyday
    "How do I make pasta from scratch?",
    "What are some tips for better sleep?",
    "How do I change a flat tire?",
    "What are the benefits of regular exercise?",
    "How do I start a vegetable garden?",
    "What is the best way to learn a new language?",
    # Creative
    "Write a short poem about autumn.",
    "Tell me an interesting fact about dolphins.",
    "Suggest a good book for beginners in philosophy.",
    "What are some fun activities for a rainy day?",
    # Professional
    "How do I write a good resume?",
    "What are best practices for giving a presentation?",
    "Explain the basics of project management.",
    "What is machine learning?",
    "How does a database work?",
    # Image-related (for VLMs)
    "Describe what you see in this image.",
    "What objects are present in this picture?",
    "What colors are dominant in this image?",
    "Is there a person in this image?",
    "What is happening in this scene?",
    "Can you read any text in this image?",
    "What is the setting of this photograph?",
    "Describe the mood of this image.",
]

TOPICS = [
    "quantum computing", "machine learning", "climate change",
    "the solar system", "DNA", "the internet", "democracy",
    "evolution", "the stock market", "renewable energy",
    "antibiotics", "black holes", "the human brain",
    "plate tectonics", "the water cycle", "blockchain",
    "artificial intelligence", "the Renaissance", "photosynthesis",
    "the periodic table", "relativity", "the immune system",
    "algorithms", "the French Revolution", "the Big Bang",
    "semiconductors", "the carbon cycle", "linguistics",
    "game theory", "cryptography", "the food chain",
    "neural networks", "probability", "thermodynamics",
    "ecology", "the Roman Empire", "genetics",
    "philosophy of science", "music theory", "astronomy",
    "organic chemistry", "data structures", "typography",
    "architecture", "nutrition", "meteorology",
    "oceanography", "volcanology", "paleontology",
]

COMPARISONS = [
    ("mitosis", "meiosis"),
    ("weather", "climate"),
    ("speed", "velocity"),
    ("mass", "weight"),
    ("bacteria", "viruses"),
    ("hardware", "software"),
    ("RAM", "ROM"),
    ("TCP", "UDP"),
    ("stars", "planets"),
    ("acids", "bases"),
    ("metals", "nonmetals"),
    ("classical", "quantum mechanics"),
    ("DNA", "RNA"),
    ("socialism", "capitalism"),
    ("deductive", "inductive reasoning"),
]


def generate_safe_prompts(n: int, seed: int = 42) -> list[dict[str, str]]:
    """Generate diverse safe prompts.

    Combines templates with topics, plus adds prompts from HuggingFace
    datasets if available.

    Args:
        n: Number of prompts to generate.
        seed: Random seed.

    Returns:
        List of {"prompt": "..."} dicts.
    """
    rng = random.Random(seed)
    prompts: list[str] = []

    # Template-based generation
    for template in SAFE_PROMPT_TEMPLATES:
        if "{topic}" in template:
            for topic in TOPICS:
                prompts.append(template.format(topic=topic))
        elif "{a}" in template and "{b}" in template:
            for a, b in COMPARISONS:
                prompts.append(template.format(a=a, b=b))
        else:
            prompts.append(template)

    # Try loading from HuggingFace datasets
    hf_prompts = _load_hf_safe_prompts()
    prompts.extend(hf_prompts)

    # Deduplicate and shuffle
    prompts = list(set(prompts))
    rng.shuffle(prompts)

    # Truncate to requested size
    prompts = prompts[:n]

    logger.info("Generated %d safe prompts (%d from templates, %d from HF)",
                len(prompts),
                min(len(prompts), len(SAFE_PROMPT_TEMPLATES) * len(TOPICS)),
                len(hf_prompts))

    return [{"prompt": p} for p in prompts]


def _load_hf_safe_prompts() -> list[str]:
    """Try to load safe prompts from HuggingFace datasets."""
    prompts: list[str] = []

    # Try Alpaca
    try:
        from datasets import load_dataset
        logger.info("Loading safe prompts from yahma/alpaca-cleaned...")
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        for item in ds:
            instruction = item.get("instruction", "")
            if instruction and len(instruction) > 10:
                inp = item.get("input", "")
                if inp:
                    prompts.append(f"{instruction}\n{inp}")
                else:
                    prompts.append(instruction)
            if len(prompts) >= 5000:
                break
        logger.info("Loaded %d prompts from alpaca-cleaned", len(prompts))
    except Exception as e:
        logger.warning("Could not load alpaca-cleaned: %s", e)

    return prompts


def load_unsafe_prompts(
    advbench_path: str | None = None,
    n: int = 2000,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Load unsafe prompts from multiple open-access sources.

    Sources (tried in order, all non-gated):
    1. AdvBench CSV (local)
    2. PKU-Alignment/BeaverTails (HuggingFace, open)
    3. LibrAI/do-not-answer (HuggingFace, open)
    4. walledai/StrongReject (HuggingFace, may be gated)

    Args:
        advbench_path: Path to AdvBench CSV. None to skip.
        n: Max number of prompts.
        seed: Random seed.

    Returns:
        List of {"prompt": "..."} dicts.
    """
    rng = random.Random(seed)
    prompts: list[str] = []

    # Source 1: AdvBench CSV (skip header row if present)
    if advbench_path and Path(advbench_path).is_file():
        with open(advbench_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row or not row[0].strip():
                    continue
                text = row[0].strip()
                # Skip header-like rows
                if i == 0 and text.lower() in ("goal", "prompt", "instruction"):
                    continue
                prompts.append(text)
        logger.info("Loaded %d prompts from AdvBench", len(prompts))

    # Source 2: BeaverTails (open, ~330k samples with is_safe labels)
    if len(prompts) < n:
        try:
            from datasets import load_dataset
            logger.info("Loading from PKU-Alignment/BeaverTails...")
            ds = load_dataset(
                "PKU-Alignment/BeaverTails", split="30k_test",
            )
            count = 0
            for item in ds:
                if item.get("is_safe") is False:
                    prompt = item.get("prompt", "")
                    if prompt and len(prompt) > 10:
                        prompts.append(prompt)
                        count += 1
                if len(prompts) >= n * 2:
                    break
            logger.info("Added %d prompts from BeaverTails", count)
        except Exception as e:
            logger.warning("Could not load BeaverTails: %s", e)

    # Source 3: do-not-answer (open, 939 harmful questions)
    if len(prompts) < n:
        try:
            from datasets import load_dataset
            logger.info("Loading from LibrAI/do-not-answer...")
            ds = load_dataset("LibrAI/do-not-answer", split="train")
            count = 0
            for item in ds:
                prompt = item.get("question", "") or item.get("prompt", "")
                if prompt and len(prompt) > 10:
                    prompts.append(prompt)
                    count += 1
            logger.info("Added %d prompts from do-not-answer", count)
        except Exception as e:
            logger.warning("Could not load do-not-answer: %s", e)

    # Source 4: StrongReject (may be gated — try anyway)
    if len(prompts) < n:
        try:
            from datasets import load_dataset
            logger.info("Trying walledai/StrongReject (may require auth)...")
            ds = load_dataset("walledai/StrongReject", split="train")
            count = 0
            for item in ds:
                prompt = item.get("prompt", "") or item.get("forbidden_prompt", "")
                if prompt and len(prompt) > 10:
                    prompts.append(prompt)
                    count += 1
                if len(prompts) >= n * 2:
                    break
            logger.info("Added %d prompts from StrongReject", count)
        except Exception as e:
            logger.warning("Could not load StrongReject (gated?): %s", e)

    # Deduplicate and shuffle
    prompts = list(set(prompts))
    rng.shuffle(prompts)
    prompts = prompts[:n]

    logger.info("Final unsafe prompt count: %d", len(prompts))
    return [{"prompt": p} for p in prompts]


def save_jsonl(samples: list[dict[str, str]], path: str) -> None:
    """Save samples to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare safe/unsafe contrast data for RDSA"
    )
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples per side (safe/unsafe)")
    parser.add_argument("--output-safe", type=str,
                        default="data/safe_prompts.jsonl")
    parser.add_argument("--output-unsafe", type=str,
                        default="data/unsafe_prompts.jsonl")
    parser.add_argument("--advbench-path", type=str,
                        default="data/advbench.csv",
                        help="Path to AdvBench CSV (optional)")
    parser.add_argument("--safe-only", action="store_true",
                        help="Only generate safe prompts")
    parser.add_argument("--unsafe-only", action="store_true",
                        help="Only generate unsafe prompts")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()

    if not args.unsafe_only:
        logger.info("=== Generating safe prompts ===")
        safe = generate_safe_prompts(args.n_samples, seed=args.seed)
        save_jsonl(safe, args.output_safe)

    if not args.safe_only:
        logger.info("=== Loading unsafe prompts ===")
        advbench = args.advbench_path if Path(args.advbench_path).is_file() else None
        unsafe = load_unsafe_prompts(advbench, n=args.n_samples, seed=args.seed)
        save_jsonl(unsafe, args.output_unsafe)

    logger.info("Done. Data ready for: python -m rdsa.identify")


if __name__ == "__main__":
    main()
