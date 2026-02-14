"""Command-line interface for ROPE benchmark.

Provides commands for running evaluations, demos, and listing
available models and defenses.

Usage:
    rope run --models llama2-7b --defenses none,delimiter
    rope demo
    rope list-models
    rope list-defenses
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from rope.defenses import DEFENSES
from rope.eval import run_eval
from rope.metrics import compute_metrics, generate_report, print_summary
from rope.models import MODELS
from rope.validation import calibrate_judge, print_validation_report, validate_results

app = typer.Typer(
    help="ROPE: Reproducible Offline Prompt-injection Evaluation for Local LLMs",
    add_completion=False,
)


@app.command()
def run(
    models: str = typer.Option(
        "llama2-7b,llama3-8b,phi2",
        "--models",
        "-m",
        help="Comma-separated model names",
    ),
    defenses: str = typer.Option(
        "none,delimiter,icl",
        "--defenses",
        "-d",
        help="Comma-separated defense names",
    ),
    output: str = typer.Option(
        "results.json",
        "--output",
        "-o",
        help="Output file path for raw results",
    ),
    tasks: str = typer.Option(
        "data/tasks.json",
        "--tasks",
        help="Path to tasks file",
    ),
    attacks: str = typer.Option(
        "data/attacks.json",
        "--attacks",
        help="Path to attacks file",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    judge: str = typer.Option(
        "llama3-8b",
        "--judge",
        "-j",
        help="Model to use as judge (e.g., llama3-8b, phi2)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Save full debug info (defended prompts, raw judge output, full responses)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from checkpoint: skip (model, defense) pairs already in the output file",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-D",
        help="Enable per-stage debug logging to terminal and file",
    ),
    max_attacks: int = typer.Option(
        None,
        "--max-attacks",
        help="Limit number of attacks to evaluate (useful for quick debug runs)",
    ),
) -> None:
    """Run full ROPE evaluation.

    Example:
        rope run --models llama2-7b --defenses none,delimiter
        rope run --models phi2 --judge phi2 --verbose
        rope run --resume -o results.json
        rope run --debug --max-attacks 3
    """
    # Parse inputs
    model_list = [m.strip() for m in models.split(",")]
    defense_list = [d.strip() for d in defenses.split(",")]

    # Validate models
    for m in model_list:
        if m not in MODELS:
            typer.echo(f"Error: Unknown model: {m}", err=True)
            typer.echo(f"Available: {list(MODELS.keys())}", err=True)
            sys.exit(1)

    # Validate judge
    if judge not in MODELS:
        typer.echo(f"Error: Unknown judge model: {judge}", err=True)
        typer.echo(f"Available: {list(MODELS.keys())}", err=True)
        sys.exit(1)

    # Validate defenses
    for d in defense_list:
        if d not in DEFENSES:
            typer.echo(f"Error: Unknown defense: {d}", err=True)
            typer.echo(f"Available: {list(DEFENSES.keys())}", err=True)
            sys.exit(1)

    # Check data files exist
    if not Path(tasks).exists():
        typer.echo(f"Error: Tasks file not found: {tasks}", err=True)
        typer.echo("Run from the rope-bench root directory or specify --tasks path.", err=True)
        sys.exit(1)
    if not Path(attacks).exists():
        typer.echo(f"Error: Attacks file not found: {attacks}", err=True)
        typer.echo("Run from the rope-bench root directory or specify --attacks path.", err=True)
        sys.exit(1)

    # Print configuration
    typer.echo("=" * 70)
    typer.echo("ROPE EVALUATION")
    typer.echo("=" * 70)
    typer.echo(f"Models: {model_list}")
    typer.echo(f"Defenses: {defense_list}")
    typer.echo(f"Judge: {judge}")
    typer.echo(f"Output: {output}")
    typer.echo(f"Verbose: {verbose}")
    typer.echo(f"Debug: {debug}")
    typer.echo(f"Max Attacks: {max_attacks}")
    typer.echo(f"Resume: {resume}")
    typer.echo(f"Seed: {seed}")
    typer.echo("=" * 70 + "\n")

    # Run evaluation
    results = run_eval(
        model_names=model_list,
        defense_names=defense_list,
        output_path=output,
        tasks_path=tasks,
        attacks_path=attacks,
        seed=seed,
        judge_model_name=judge,
        verbose=verbose,
        resume=resume,
        debug=debug,
        max_attacks=max_attacks,
    )

    # Compute and display metrics
    metrics = compute_metrics(results)
    print_summary(metrics, results)

    # Run validation sanity checks
    issues = validate_results(results)
    print_validation_report(issues)

    # Save metrics CSV
    metrics_path = output.replace(".json", "_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    typer.echo(f"  Metrics saved to {metrics_path}")

    # Generate detailed report
    report_path = output.replace(".json", "_report.txt")
    generate_report(results, report_path)


@app.command()
def demo(
    cpu: bool = typer.Option(
        False,
        "--cpu",
        help="CPU-only demo: use Phi-2 only (one model in RAM). Use this if you have no GPU or hit 'paging file too small'.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Save full debug info (defended prompts, raw judge output, full responses)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-D",
        help="Enable per-stage debug logging to terminal and file",
    ),
    max_attacks: int = typer.Option(
        None,
        "--max-attacks",
        help="Override number of attacks (default 10 for --cpu, 20 for GPU)",
    ),
) -> None:
    """Run quick demo (1 model, 2 defenses, first 20 attacks).

    Without GPU or if you see "paging file too small", run: rope demo --cpu

    Example:
        rope demo
        rope demo --cpu
        rope demo --debug --max-attacks 3
    """
    import torch

    typer.echo("Running ROPE demo...\n")
    typer.echo(f"Debug: {debug}")
    if max_attacks:
        typer.echo(f"Max Attacks: {max_attacks}")

    tasks_path = "data/tasks.json"
    attacks_path = "data/attacks.json"

    # Check files exist
    if not Path(tasks_path).exists():
        typer.echo(f"Error: Tasks file not found: {tasks_path}", err=True)
        typer.echo("Run from the rope-bench root directory.", err=True)
        sys.exit(1)
    if not Path(attacks_path).exists():
        typer.echo(f"Error: Attacks file not found: {attacks_path}", err=True)
        sys.exit(1)

    if cpu:
        # CPU-only: one small model (Phi-2), reused as judge. Fewer attacks to keep time reasonable.
        if not torch.cuda.is_available():
            typer.echo("CPU mode: using Phi-2 only (loaded once for both eval and judge).")
            typer.echo(
                "NOTE: Phi-2 (2.7B) is a weak judge. Severity scores may be inaccurate.\n"
                "      Use --debug to inspect raw judge outputs. For reliable results,\n"
                "      run on GPU with llama3-8b as judge (rope demo without --cpu).\n"
            )
        n_attacks = max_attacks if max_attacks else 10
        results = run_eval(
            model_names=["phi2"],
            defense_names=["none"],
            output_path="demo_results.json",
            attacks_path=attacks_path,
            judge_model_name="phi2",
            reuse_judge_for_model="phi2",
            verbose=verbose,
            debug=debug,
            max_attacks=n_attacks,
        )
    else:
        # GPU demo: llama2-7b + judge llama3-8b, 20 attacks, 2 defenses
        if not torch.cuda.is_available():
            typer.echo(
                "Warning: CUDA not available. Loading 7B+8B on CPU can need 50GB+ RAM and may fail with "
                "'paging file too small'. For a CPU-only demo run: rope demo --cpu\n",
                err=True,
            )
        n_attacks = max_attacks if max_attacks else 20
        results = run_eval(
            model_names=["llama2-7b"],
            defense_names=["none", "delimiter"],
            output_path="demo_results.json",
            attacks_path=attacks_path,
            verbose=verbose,
            debug=debug,
            max_attacks=n_attacks,
        )

    # Display results
    metrics = compute_metrics(results)
    print_summary(metrics, results)

    # Run validation sanity checks
    issues = validate_results(results)
    print_validation_report(issues)

    # Save metrics CSV
    metrics_path = "demo_results_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    typer.echo(f"\n  Metrics saved to {metrics_path}")

    # Generate detailed report
    report_path = "demo_results_report.txt"
    generate_report(results, report_path)
    typer.echo(f"  Report saved to {report_path}")

    typer.echo("\nDemo complete! Run full evaluation with: rope run")


@app.command(name="validate-judge")
def validate_judge(
    judge: str = typer.Option(
        "phi2",
        "--judge",
        "-j",
        help="Model to use as judge (e.g., phi2, llama3-8b)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed judge output for each example",
    ),
) -> None:
    """Test judge model accuracy on hand-labeled calibration examples.

    Use this to verify the judge is working correctly before running evaluations.

    Example:
        rope validate-judge --judge phi2 --verbose
        rope validate-judge --judge llama3-8b
    """
    from rope.models import load_model

    if judge not in MODELS:
        typer.echo(f"Error: Unknown judge model: {judge}", err=True)
        typer.echo(f"Available: {list(MODELS.keys())}", err=True)
        sys.exit(1)

    typer.echo(f"\nLoading judge model ({judge})...")
    judge_model, judge_tokenizer = load_model(judge)

    report = calibrate_judge(judge_model, judge_tokenizer, verbose=verbose)

    if report["accuracy"] < 0.5:
        typer.echo("Judge failed calibration. Consider using a different model.")
        sys.exit(1)


@app.command(name="list-models")
def list_models() -> None:
    """List available models."""
    typer.echo("Available models:")
    for name, hf_id in MODELS.items():
        typer.echo(f"  {name}: {hf_id}")


@app.command(name="list-defenses")
def list_defenses() -> None:
    """List available defenses."""
    typer.echo("Available defenses:")
    for name in DEFENSES:
        typer.echo(f"  {name}")


if __name__ == "__main__":
    app()
