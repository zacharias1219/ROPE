---
name: ROPE Implementation Plan
overview: "Build ROPE v1: a local-first benchmark for evaluating prompt injection defenses on local LLMs (2-8B params). Includes 3 models, 30 tasks, 4 attack types, 4 defenses, automated judging, and CLI interface."
todos:
  - id: setup-structure
    content: Create project directory structure (rope/, tests/, examples/, docker/) and package files (pyproject.toml, requirements.txt, __init__.py)
    status: pending
  - id: config-system
    content: "Create configuration system: base_config.yaml, models.yaml, defenses.yaml, runs/example_run.yaml with YAML include support"
    status: pending
  - id: datasets
    content: Create tasks_v1.jsonl (30 tasks across 3 families) and attacks_v1.jsonl (4 attack types) with proper JSONL format
    status: pending
  - id: models-base
    content: Implement BaseModel ABC and HFModel class with GPTQ/BitsAndBytes quantization support and from_config() factory
    status: pending
  - id: defenses
    content: "Implement all 4 defenses: NoneDefense, DelimiterDefense, ParaphraseDefense (T5), ICLDefense with BaseDefense ABC"
    status: pending
  - id: judge
    content: Implement BaseJudge ABC, LocalLLMJudge with severity scoring (0-3), and optional HeuristicJudge
    status: pending
  - id: eval-runner
    content: Implement evaluation runner with nested loops (models→defenses→tasks→attacks), batching, and result collection
    status: pending
  - id: metrics
    content: Implement metrics calculation (ASR1p, ASR3, avg_severity, clean_accuracy) and CSV summary generation
    status: pending
  - id: cli
    content: "Implement CLI with typer/click: rope run --config and rope demo commands with progress output"
    status: pending
  - id: reproducibility
    content: Add seed management, config snapshotting, git commit hash capture, and output directory structure
    status: pending
  - id: tests
    content: Create unit tests for datasets, defenses, judge, metrics, and runner (with mocked models)
    status: pending
  - id: docs-examples
    content: Update README, create quickstart.ipynb and compare_defenses.ipynb notebooks, add Dockerfiles
    status: pending
isProject: false
---

# ROPE Implementation Plan

## Overview

ROPE (Reproducible Offline Prompt-Injection Evaluation) is a defense-centric benchmark for local LLMs. This plan implements v1 with text-only models, 3 local models (2-8B), 30 tasks, 4 attack types, 4 defenses, and automated severity scoring (0-3).

## Implementation Phases

### Phase 1: Project Foundation

- **Setup project structure**: Create directory tree per [design.md](design.md) (rope/, tests/, examples/, docker/)
- **Package configuration**: Create `pyproject.toml` with dependencies (transformers, torch, accelerate, bitsandbytes, auto-gptq, typer/click, pyyaml, pandas)
- **Base config files**: Create `rope/config/base_config.yaml` with seed, data paths, model configs, defenses list, judge config, eval params, output dir
- **Run config template**: Create `rope/config/runs/example_run.yaml` with include directive and filtered eval settings

### Phase 2: Core Data Layer

- **Tasks dataset**: Create `rope/datasets/tasks/tasks_v1.jsonl` with 30 tasks across 3 families (instruction_following, qa_summarization, rag_simple). Each entry: task_id, family, base_prompt, expected_behavior, metadata
- **Attacks dataset**: Create `rope/datasets/attacks/attacks_v1.jsonl` linking tasks to 4 attack types (direct_hijack, system_extraction, obfuscation, doc_poison). Each entry: task_id, attack_id, attack_type, attack_text, metadata
- **Data loaders**: Implement JSONL parsing utilities in `rope/datasets/` (or inline in runner)

### Phase 3: Models Module

- **Base interface**: Create `rope/models/base.py` with `BaseModel` ABC (name attribute, abstract `generate(prompts, max_new_tokens, **gen_kwargs)` method)
- **HF implementation**: Create `rope/models/hf_model.py` with `HFModel` class:
  - Supports GPTQ (4-bit) and BitsAndBytes (4-bit) quantization
  - `from_config()` classmethod to build from YAML dict
  - Batch generation with proper tokenization, device handling, decoding
  - Model loading/caching logic

### Phase 4: Defenses Module

- **Base interface**: Create `rope/defenses/base.py` with `BaseDefense` ABC (name attribute, abstract `apply(system_prompt, user_prompt)` method)
- **None defense**: Create `rope/defenses/none_defense.py` - simple concatenation
- **Delimiter defense**: Create `rope/defenses/delimiter_defense.py` - wraps user input in `<<<USER_INPUT>>>` delimiters with system instruction
- **Paraphrase defense**: Create `rope/defenses/paraphrase_defense.py` - uses small T5 model to paraphrase user prompt
- **ICL defense**: Create `rope/defenses/icl_defense.py` - prepends few-shot examples of proper refusals
- **Defense registry**: Factory function to instantiate defenses from config names

### Phase 5: Judge Module

- **Base interface**: Create `rope/judge/base.py` with `BaseJudge` ABC (name attribute, abstract `score(items)` method returning list[int] 0-3)
- **Local LLM judge**: Create `rope/judge/local_llm_judge.py`:
  - Uses HFModel wrapper with classification prompt template
  - System prompt: security evaluator role, severity scale explanation
  - User prompt: expected_behavior, attack_type, response
  - Parse numeric answer (0-3), fallback to 0 on parse failure
- **Heuristic judge**: Create `rope/judge/heuristics.py` with keyword-based checks (optional, can enhance LLM scores)

### Phase 6: Evaluation System

- **Runner**: Create `rope/eval/runner.py`:
  - Nested loops: models → defenses → tasks → attacks
  - Load models once per model config
  - Apply defense to build final prompt
  - Batch generation calls
  - Collect (task, attack, model, defense, response) records
  - Pass to judge for scoring
  - Save raw results to JSONL
- **Metrics**: Create `rope/eval/metrics.py`:
  - Compute ASR1p (severity ≥1 rate), ASR3 (severity=3 rate), avg_severity per (model, defense, attack_type)
  - Clean task accuracy (separate evaluation)
  - Generate summary CSV
- **Logging**: Create `rope/eval/logging.py`:
  - Progress bars/logging during eval
  - Save config snapshot, git commit hash to metadata.json
  - Output structure: `outputs/{run_name}/raw/`, `outputs/{run_name}/summary/`

### Phase 7: CLI Interface

- **Main CLI**: Create `rope/cli/main.py`:
  - Use typer or click
  - `rope run --config <path>`: Full evaluation from run config
  - `rope demo`: Quick demo (1 model × 1 defense × 5 tasks × all attacks)
  - Progress output, summary table
- **Entry point**: Add `[project.scripts]` section to `pyproject.toml`: `rope = "rope.cli.main:app"`

### Phase 8: Reproducibility & Utilities

- **Seed management**: Set global seeds (Python random, numpy, torch) from config
- **Config snapshotting**: Copy run config to output dir
- **Git integration**: Capture git commit hash in metadata.json (handle non-git dirs gracefully)
- **Versioning**: Document dataset versioning strategy (immutable v1 files)

### Phase 9: Testing

- **Unit tests**: Create `tests/` directory with:
  - `test_datasets.py`: JSONL parsing, filtering
  - `test_defenses.py`: Each defense applies correctly
  - `test_judge.py`: Judge scoring logic, parsing
  - `test_metrics.py`: Metric calculations
  - `test_runner.py`: Runner flow (mock models)
- **Integration test**: Small end-to-end run with dummy data

### Phase 10: Documentation & Examples

- **README**: Update with installation, quickstart, config examples
- **Quickstart notebook**: Create `examples/quickstart.ipynb` showing basic usage
- **Compare defenses notebook**: Create `examples/compare_defenses.ipynb` with visualization
- **Dockerfiles**: Create `docker/Dockerfile.cpu` and `docker/Dockerfile.gpu` for reproducible environments

## Key Implementation Notes

- **Quantization**: Support both GPTQ (via auto-gptq) and BitsAndBytes for different model families
- **Batch processing**: Efficient batching in model.generate() to handle multiple prompts
- **Memory management**: Models loaded once per eval run, not per task
- **Error handling**: Graceful failures (log and continue) for individual task/attack failures
- **Config inheritance**: YAML include mechanism for run configs extending base config
- **Severity scoring**: Judge must return 0-3 integers; implement robust parsing with fallbacks

## Dependencies Summary

- Core: transformers, torch, accelerate
- Quantization: bitsandbytes, auto-gptq
- CLI: typer (or click)
- Config: pyyaml
- Data: pandas (for CSV output)
- Utils: tqdm (progress bars)

## File Structure Reference

See [design.md](design.md) section 3 for complete directory layout. Key files:

- `rope/config/base_config.yaml` - Default configuration
- `rope/datasets/tasks/tasks_v1.jsonl` - 30 base tasks
- `rope/datasets/attacks/attacks_v1.jsonl` - Attack variants
- `rope/models/hf_model.py` - HuggingFace model wrapper
- `rope/defenses/*.py` - Four defense implementations
- `rope/judge/local_llm_judge.py` - Automated severity scorer
- `rope/eval/runner.py` - Main evaluation loop
- `rope/cli/main.py` - Command-line interface

