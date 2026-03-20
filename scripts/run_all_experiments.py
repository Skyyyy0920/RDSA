"""Master script: run all NeurIPS experiments in order.

Orchestrates Exp 1-10 with proper dependency ordering.
Supports --dry-run, --model, and --start-from for resuming.

Usage:
    python scripts/run_all_experiments.py --model qwen3vl --dry-run
    python scripts/run_all_experiments.py --model all --start-from 3
    python scripts/run_all_experiments.py --model qwen3vl --only 1 2 6
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Experiment definitions: (script_name, description, dependencies)
EXPERIMENTS: dict[int, dict[str, object]] = {
    1: {
        "script": "scripts/exp1_main_comparison.py",
        "name": "Main Comparison (Table 1)",
        "deps": [],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
    2: {
        "script": "scripts/exp2_ablation.py",
        "name": "Ablation Study (Table 2)",
        "deps": [],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
    3: {
        "script": "scripts/exp3_sa_at_analysis.py",
        "name": "SA-AT Analysis (Fig 2-3)",
        "deps": [],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
    4: {
        "script": "scripts/exp4_redundancy_analysis.py",
        "name": "Redundancy Analysis (Fig 4-5)",
        "deps": [],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
    5: {
        "script": "scripts/exp5_entanglement_analysis.py",
        "name": "Entanglement Analysis (Fig 6)",
        "deps": [],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
    6: {
        "script": "scripts/exp6_capability.py",
        "name": "Capability Preservation (Table 4)",
        "deps": [1],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
    7: {
        "script": "scripts/exp7_adaptive_attacks.py",
        "name": "Adaptive Attacks (Table 5)",
        "deps": [1],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
    8: {
        "script": "scripts/exp8_transfer.py",
        "name": "Transfer Attack (Table 6)",
        "deps": [1],
        "args": ["--surrogate={model}", "--dry-run={dry_run}"],
    },
    9: {
        "script": "scripts/exp9_sensitivity.py",
        "name": "Parameter Sensitivity (Fig 7)",
        "deps": [],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
    10: {
        "script": "scripts/exp10_pareto.py",
        "name": "Pareto Frontier (Fig 1)",
        "deps": [1, 9],
        "args": ["--model={model}", "--dry-run={dry_run}"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all RDSA NeurIPS experiments"
    )
    parser.add_argument("--model", type=str, default="qwen3vl",
                        help="Model shortname or 'all'")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from experiment number")
    parser.add_argument("--only", nargs="*", type=int, default=None,
                        help="Run only specific experiment numbers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue even if an experiment fails")
    return parser.parse_args()


def run_experiment(
    exp_id: int,
    model: str,
    dry_run: bool,
) -> bool:
    """Run a single experiment script.

    Returns:
        True if experiment succeeded.
    """
    exp = EXPERIMENTS[exp_id]
    script = exp["script"]
    name = exp["name"]

    # Format arguments
    dry_run_flag = "" if not dry_run else "--dry-run"
    args = []
    for arg_template in exp["args"]:
        arg = arg_template.format(model=model, dry_run=dry_run_flag)
        if "={dry_run}" in arg_template:
            if dry_run:
                args.append("--dry-run")
            continue
        args.append(arg)

    cmd = [sys.executable, script] + args
    logger.info("=" * 60)
    logger.info("Experiment %d: %s", exp_id, name)
    logger.info("CMD: %s", " ".join(cmd))
    logger.info("=" * 60)

    start_time = time.time()

    try:
        subprocess.run(
            cmd, check=True, capture_output=False, text=True
        )
        elapsed = time.time() - start_time
        logger.info(
            "Experiment %d completed in %.1f minutes", exp_id, elapsed / 60
        )
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(
            "Experiment %d FAILED after %.1f minutes (exit code %d)",
            exp_id, elapsed / 60, e.returncode,
        )
        return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/run_all.log", encoding="utf-8"),
        ],
    )
    Path("logs").mkdir(exist_ok=True)

    args = parse_args()

    # Determine which experiments to run
    if args.only:
        exp_ids = sorted(args.only)
    else:
        exp_ids = sorted(
            eid for eid in EXPERIMENTS if eid >= args.start_from
        )

    logger.info("=" * 60)
    logger.info("RDSA NeurIPS Experiment Suite")
    logger.info("Model: %s", args.model)
    logger.info("Experiments: %s", exp_ids)
    logger.info("Dry run: %s", args.dry_run)
    logger.info("=" * 60)

    completed: set[int] = set()
    failed: set[int] = set()
    total_start = time.time()

    for exp_id in exp_ids:
        # Check dependencies
        deps = EXPERIMENTS[exp_id].get("deps", [])
        unmet = [d for d in deps if d not in completed and d not in failed]
        if unmet and not args.dry_run:
            # Dependencies not in our run list — assume they completed earlier
            missing_in_list = [d for d in unmet if d not in exp_ids]
            if missing_in_list:
                logger.warning(
                    "Exp %d has unmet deps %s (not in run list, assuming done)",
                    exp_id, missing_in_list,
                )
            failed_deps = [d for d in deps if d in failed]
            if failed_deps and not args.continue_on_error:
                logger.error(
                    "Skipping Exp %d: dependency %s failed", exp_id, failed_deps
                )
                failed.add(exp_id)
                continue

        success = run_experiment(exp_id, args.model, args.dry_run)

        if success:
            completed.add(exp_id)
        else:
            failed.add(exp_id)
            if not args.continue_on_error:
                logger.error("Stopping due to failure. Use --continue-on-error to proceed.")
                break

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUITE COMPLETE")
    logger.info("Total time: %.1f hours", total_elapsed / 3600)
    logger.info("Completed: %s", sorted(completed))
    logger.info("Failed: %s", sorted(failed))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
