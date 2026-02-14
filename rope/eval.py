"""Main evaluation orchestrator for ROPE benchmark.

Runs the full evaluation loop: for each model x defense x attack,
generates a response, judges severity, and records results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from rope.defenses import DEFENSES
from rope.judge import score_response
from rope.models import MODELS, generate, load_model


def _debug_log(stage: str, message: str, data: str | None = None, log_file=None) -> None:
    """Print a debug message to stderr and optionally to a file."""
    header = f"\n>>> [DEBUG: {stage}] {'=' * (50 - len(stage))}"
    msg = f"{header}\n{message}"
    if data:
        msg += f"\n{data}"
    msg += f"\n{'=' * 65}\n"
    
    print(msg, file=sys.stderr)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def _save_checkpoint(results: list[dict], output_path: str) -> None:
    """Save intermediate results for crash recovery."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _load_checkpoint(output_path: str) -> tuple[list[dict], set[tuple[str, str]]]:
    """Load existing results and determine which (model, defense) pairs are done.

    Returns:
        Tuple of (existing_results, completed_pairs).
    """
    path = Path(output_path)
    if not path.exists():
        return [], set()

    with open(path) as f:
        results = json.load(f)

    completed = set()
    for r in results:
        completed.add((r["model"], r["defense"]))

    return results, completed


def run_eval(
    model_names: list[str],
    defense_names: list[str],
    output_path: str = "results.json",
    tasks_path: str = "data/tasks.json",
    attacks_path: str = "data/attacks.json",
    seed: int = 42,
    judge_model_name: str = "llama3-8b",
    reuse_judge_for_model: str | None = None,
    verbose: bool = False,
    resume: bool = False,
    debug: bool = False,
    max_attacks: int | None = None,
) -> list[dict]:
    """Run full ROPE evaluation.

    Args:
        model_names: List of model keys (e.g., ["llama2-7b", "phi2"]).
        defense_names: List of defense keys (e.g., ["none", "delimiter"]).
        output_path: Where to save raw results JSON.
        tasks_path: Path to tasks.json.
        attacks_path: Path to attacks.json.
        seed: Random seed for reproducibility.
        judge_model_name: Which model to use as judge.
        reuse_judge_for_model: If set, reuse the judge model for this eval model (saves RAM).
        verbose: If True, save full debug info (defended prompt, raw judge output, full response).
        resume: If True, skip (model, defense) pairs already in the output file.
        debug: If True, print per-attack per-stage debug info to stderr and log file.
        max_attacks: If set, only evaluate the first N attacks per (model, defense) pair.

    Returns:
        List of result dicts.

    Example:
        >>> results = run_eval(
        ...     model_names=["phi2"],
        ...     defense_names=["none"],
        ...     output_path="test_results.json",
        ... )
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Setup debug log file
    log_file = None
    if debug:
        log_path = Path(output_path).with_suffix(".debug.log")
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"Debug mode enabled. Logging to stderr and {log_path}")

    # Load data
    print("Loading tasks and attacks...")
    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(attacks_path) as f:
        attacks = json.load(f)

    # Slice attacks if max_attacks is set
    if max_attacks:
        attacks = attacks[:max_attacks]
        print(f"  Limited to first {max_attacks} attacks.")

    # Build task lookup
    task_lookup = {t["id"]: t for t in tasks}

    print(f"  Loaded {len(tasks)} tasks, {len(attacks)} attacks")

    # Handle resume
    results = []
    completed_pairs: set[tuple[str, str]] = set()
    if resume:
        results, completed_pairs = _load_checkpoint(output_path)
        if completed_pairs:
            print(f"  Resuming: {len(completed_pairs)} (model, defense) pairs already done.")

    # Load judge model once (reuse for all evaluations)
    print(f"\nLoading judge model ({judge_model_name})...")
    judge_model, judge_tokenizer = load_model(judge_model_name)

    # Load paraphrase model if needed
    paraphrase_model = None
    paraphrase_tokenizer = None
    if "paraphrase" in defense_names:
        print("Loading paraphrase model (t5-base)...")
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            paraphrase_model = T5ForConditionalGeneration.from_pretrained(
                "t5-base", device_map="auto"
            )
            paraphrase_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        except Exception as e:
            print(f"  Warning: Could not load paraphrase model: {e}")
            print("  Paraphrase defense will pass through unchanged.")

    # Main evaluation loop
    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'=' * 60}")

        if reuse_judge_for_model and model_name == reuse_judge_for_model:
            model, tokenizer = judge_model, judge_tokenizer
        else:
            model, tokenizer = load_model(model_name)

        for defense_name in defense_names:
            # Skip if already computed (resume mode)
            if (model_name, defense_name) in completed_pairs:
                print(f"\n  Defense: {defense_name} [SKIPPED - already in checkpoint]")
                continue

            print(f"\n  Defense: {defense_name}")
            defense_fn = DEFENSES[defense_name]

            pair_results = []

            # Progress bar
            pbar = tqdm(attacks, desc=f"  {model_name}/{defense_name}", ncols=80)

            for idx, attack in enumerate(pbar):
                # Get original task
                task = task_lookup.get(attack["task_id"])
                if task is None:
                    print(f"\n  Warning: Task {attack['task_id']} not found, skipping.")
                    continue

                if debug:
                    _debug_log(
                        "ATTACK",
                        f"Attack #{idx+1} | Type: {attack['type']} | Task ID: {attack['task_id']} | Goal: {attack['goal']}",
                        f"Text: {attack['text'][:200]}...",
                        log_file
                    )

                # Apply defense
                try:
                    if defense_name == "paraphrase":
                        defended_prompt = defense_fn(
                            attack["text"],
                            paraphrase_model,
                            paraphrase_tokenizer,
                        )
                    else:
                        defended_prompt = defense_fn(attack["text"])
                except Exception as e:
                    print(f"\n  Warning: Defense failed: {e}")
                    defended_prompt = attack["text"]

                if debug:
                    _debug_log(
                        "DEFENSE",
                        f"Applied '{defense_name}'",
                        f"Prompt: {defended_prompt[:300]}...",
                        log_file
                    )

                # Generate response
                try:
                    response = generate(model, tokenizer, defended_prompt, max_tokens=256, seed=seed)
                except Exception as e:
                    print(f"\n  Warning: Generation failed: {e}")
                    response = "[ERROR]"

                if debug:
                    _debug_log(
                        "MODEL",
                        f"Response generated ({len(response)} chars)",
                        f"Output: {response[:400]}...",
                        log_file
                    )

                # Judge response
                raw_judge = ""
                try:
                    severity, raw_judge = score_response(
                        task=task["prompt"],
                        attack_goal=attack["goal"],
                        response=response,
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        verbose=verbose or debug,
                    )
                except Exception as e:
                    print(f"\n  Warning: Judging failed: {e}")
                    severity = 0  # Conservative default

                if debug:
                    _debug_log(
                        "JUDGE",
                        f"Severity: {severity}",
                        f"Raw output: {raw_judge.strip()}",
                        log_file
                    )

                # Record result
                result = {
                    "model": model_name,
                    "defense": defense_name,
                    "task_id": task["id"],
                    "task_family": task["family"],
                    "attack_type": attack["type"],
                    "severity": severity,
                    "response": response if verbose else response[:300],
                    "judge_output": raw_judge,
                }
                if verbose:
                    result["defended_prompt"] = defended_prompt
                    result["attack_text"] = attack["text"]
                    result["attack_goal"] = attack["goal"]

                pair_results.append(result)

            # Append pair results and checkpoint
            results.extend(pair_results)
            _save_checkpoint(results, output_path)
            print(f"  Checkpoint saved ({len(results)} results so far)")

        # Free GPU memory (do not delete if we reused the judge)
        if not (reuse_judge_for_model and model_name == reuse_judge_for_model):
            del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final save
    _save_checkpoint(results, output_path)

    if log_file:
        log_file.close()

    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} results to {output_path}")
    print(f"{'=' * 60}\n")

    return results
