"""Main evaluation orchestrator for ROPE benchmark.

Runs the full evaluation loop: for each model x defense x attack,
generates a response, judges severity, and records results.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from rope.defenses import DEFENSES
from rope.judge import score_response
from rope.models import MODELS, generate, load_model


def run_eval(
    model_names: list[str],
    defense_names: list[str],
    output_path: str = "results.json",
    tasks_path: str = "data/tasks.json",
    attacks_path: str = "data/attacks.json",
    seed: int = 42,
    judge_model_name: str = "llama3-8b",
    reuse_judge_for_model: str | None = None,
) -> list[dict]:
    """Run full ROPE evaluation.

    Args:
        model_names: List of model keys (e.g., ["llama2-7b", "phi2"]).
        defense_names: List of defense keys (e.g., ["none", "delimiter"]).
        output_path: Where to save raw results JSON.
        tasks_path: Path to tasks.json.
        attacks_path: Path to attacks.json.
        seed: Random seed for reproducibility.

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

    # Load data
    print("Loading tasks and attacks...")
    with open(tasks_path) as f:
        tasks = json.load(f)
    with open(attacks_path) as f:
        attacks = json.load(f)

    # Build task lookup
    task_lookup = {t["id"]: t for t in tasks}

    print(f"  Loaded {len(tasks)} tasks, {len(attacks)} attacks")

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
    results = []

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'=' * 60}")

        if reuse_judge_for_model and model_name == reuse_judge_for_model:
            model, tokenizer = judge_model, judge_tokenizer
        else:
            model, tokenizer = load_model(model_name)

        for defense_name in defense_names:
            print(f"\n  Defense: {defense_name}")
            defense_fn = DEFENSES[defense_name]

            # Progress bar
            pbar = tqdm(attacks, desc=f"  {model_name}/{defense_name}", ncols=80)

            for attack in pbar:
                # Get original task
                task = task_lookup.get(attack["task_id"])
                if task is None:
                    print(f"\n  Warning: Task {attack['task_id']} not found, skipping.")
                    continue

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

                # Generate response
                try:
                    response = generate(model, tokenizer, defended_prompt, max_tokens=256)
                except Exception as e:
                    print(f"\n  Warning: Generation failed: {e}")
                    response = "[ERROR]"

                # Judge response
                try:
                    severity = score_response(
                        task=task["prompt"],
                        attack_goal=attack["goal"],
                        response=response,
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                    )
                except Exception as e:
                    print(f"\n  Warning: Judging failed: {e}")
                    severity = 0  # Conservative default

                # Record result
                result = {
                    "model": model_name,
                    "defense": defense_name,
                    "task_id": task["id"],
                    "task_family": task["family"],
                    "attack_type": attack["type"],
                    "severity": severity,
                    "response": response[:200],  # Truncate to save space
                }
                results.append(result)

        # Free GPU memory (do not delete if we reused the judge)
        if not (reuse_judge_for_model and model_name == reuse_judge_for_model):
            del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save raw results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Saving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved {len(results)} results")
    print(f"{'=' * 60}\n")

    return results
