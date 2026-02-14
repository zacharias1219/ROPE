# ROPE Debugging Guide

This guide explains how to use the ROPE debug pipeline to verify evaluation behavior locally, especially when you see unexpected results like 0% ASR.

## Local Debug Workflow with Phi-2

If you are running on a machine with limited RAM or no GPU, use the `--cpu` flag with `rope demo`. This uses the small Phi-2 model for both evaluation and judging, which fits in ~8GB RAM.

### 1. Quick Debug Run

Run a short demo with only 3 attacks and full debug logging:

```bash
rope demo --cpu --debug --max-attacks 3
```

### 2. What to Inspect in Debug Output

The `--debug` flag prints detailed information for **every stage** of **every attack** to your terminal (stderr) and a log file (e.g., `demo_results.debug.log`).

#### Stage: ATTACK
Check if the attack text is what you expect.
- Does it contain the malicious instruction (e.g., "IGNORE ALL PREVIOUS INSTRUCTIONS")?
- Is it correctly formatted?

#### Stage: DEFENSE
Check how the defense transformed the attack.
- For `none`, it should be identical to the attack.
- For `delimiter`, you should see the attack wrapped in `<<<INPUT>>> ... <<<END_INPUT>>>`.
- If the defense is "swallowing" the attack or making it invisible to the model, this is where you'll see it.

#### Stage: MODEL
Check the model's response.
- Did the model follow the original task or the injected attack?
- If the model followed the attack, the response should reflect the `goal`.

#### Stage: JUDGE
Check the judge's reasoning and the final score.
- **Raw output:** See exactly what the judge model (e.g., Phi-2 or Llama-3) said.
- **Severity:** Check if the parsed 0–3 score matches your intuition of the response's success.
- If the judge says something irrelevant or fails to output a digit, the parser might default to 0, leading to low ASR.

### 3. Advanced Local Debugging

If you want to debug specific models or defenses with `rope run`:

```bash
rope run --models phi2 --defenses none,delimiter --judge phi2 --debug --max-attacks 5 -o debug_results.json
```

This will:
1. Load Phi-2.
2. Evaluate it against 5 attacks with each defense.
3. Use Phi-2 itself as the judge.
4. Print stage-by-stage logs to stderr and `debug_results.debug.log`.

## Judge Model Selection

**Phi-2 (2.7B) is NOT a reliable judge.** It can serve as a target model to attack, but it lacks the reasoning ability to accurately score responses on the 0-3 severity scale. When used as judge it tends to output "0" regardless of the actual response content.

For reliable severity scoring, use **llama3-8b** (or larger) as the judge model. This is the default when running `rope demo` on GPU.

### Validate your judge before trusting results

```bash
rope validate-judge --judge llama3-8b --verbose
```

This runs the judge on 8 hand-labeled examples and reports accuracy. Aim for >= 75% exact accuracy or >= 87% within-1 accuracy.

## Common Issues and Fixes

- **ASR is 0% but responses look malicious:** The judge is likely broken. Run `rope validate-judge` to check. Small models (phi2) are unreliable judges. Use `--debug` to see raw judge outputs.
- **ASR seems too high (everything scored 3):** The judge may be over-scoring. Run calibration to verify it distinguishes between "mentioning to refuse" (score 1) and "actually complying" (score 2-3).
- **Judge outputs placeholders or garbage:** ROPE retries up to 2 times with slight temperature if the first attempt is unparseable. If still failing, try a different judge model.
- **Model ignores the attack completely:** Check the **DEFENSE** stage. Some defenses might be so aggressive that the model no longer "sees" the malicious intent.
- **Out of Memory (OOM):** Use `--cpu` to stick to Phi-2, or ensure you aren't loading multiple 7B/8B models at once. ROPE tries to reuse the judge model and clear cache between models.
