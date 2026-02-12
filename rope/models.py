"""Model loading and generation for ROPE benchmark.

Supports loading HuggingFace models with 4-bit quantization for
memory-efficient evaluation on consumer GPUs.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model registry: short name -> HuggingFace model ID
MODELS: dict[str, str] = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi2": "microsoft/phi-2",
}


def load_model(
    model_name: str,
    quantize: bool = True,
    device_map: str = "auto",
) -> tuple:
    """Load a model with optional 4-bit quantization.

    Args:
        model_name: Key from MODELS dict (e.g., "llama2-7b") or a HuggingFace model ID.
        quantize: Whether to use 4-bit quantization. Set False for CPU/testing.
        device_map: Device placement strategy. "auto" for GPU, "cpu" for CPU.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ValueError: If model_name is not recognized and not a valid HF model ID.

    Example:
        >>> model, tokenizer = load_model("phi2")
        >>> print(type(model).__name__)
        'Phi2ForCausalLM'
    """
    # Resolve model ID
    if model_name in MODELS:
        model_id = MODELS[model_name]
    else:
        # Allow passing HuggingFace model IDs directly
        model_id = model_name

    # Configure quantization
    quantization_config = None
    if quantize and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif not torch.cuda.is_available():
        device_map = "cpu"

    use_cpu = device_map == "cpu"
    kwargs = {
        "quantization_config": quantization_config,
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if use_cpu:
        kwargs["low_cpu_mem_usage"] = True

    print(f"Loading {model_name} ({model_id})...")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set padding token if missing (common for decoder-only models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = getattr(model, "device", "unknown")
    print(f"  Loaded {model_name} on {device}")

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int | None = None,
) -> str:
    """Generate text from a prompt.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        prompt: Input text.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling probability.
        seed: Optional random seed for this generation call.

    Returns:
        Generated text (prompt removed).

    Example:
        >>> model, tokenizer = load_model("phi2")
        >>> response = generate(model, tokenizer, "Hello, my name is")
        >>> print(type(response))
        <class 'str'>
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens (remove prompt)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response
