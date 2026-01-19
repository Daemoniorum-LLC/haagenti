#!/usr/bin/env python3
"""
Test inference quality between original and compressed models.

Usage:
    python3 test_inference.py --original <path> --compressed <path>

Or with default paths:
    python3 test_inference.py
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import time

def load_model_with_weights(model_name: str, weights_path: str):
    """Load model architecture and replace weights from safetensors file."""
    print(f"Loading model architecture from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading weights from {weights_path}...")
    state_dict = load_file(weights_path)

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    return model

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compare_logits(model1, model2, tokenizer, prompt: str):
    """Compare logits between two models for the same input."""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        logits1 = model1(**inputs).logits
        logits2 = model2(**inputs).logits

    # Compute similarity metrics
    cosine_sim = torch.nn.functional.cosine_similarity(
        logits1.flatten().unsqueeze(0),
        logits2.flatten().unsqueeze(0)
    ).item()

    mse = torch.nn.functional.mse_loss(logits1, logits2).item()

    # Top-k agreement
    top5_1 = torch.topk(logits1[0, -1], k=5).indices
    top5_2 = torch.topk(logits2[0, -1], k=5).indices
    top5_agreement = len(set(top5_1.tolist()) & set(top5_2.tolist())) / 5

    return {
        "cosine_similarity": cosine_sim,
        "mse": mse,
        "top5_agreement": top5_agreement,
    }

def main():
    parser = argparse.ArgumentParser(description="Test inference quality")
    parser.add_argument("--original", type=str,
                       default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/model.safetensors",
                       help="Path to original model weights")
    parser.add_argument("--compressed", type=str,
                       default="/tmp/qwen-compressed-30pct-int4.safetensors",
                       help="Path to compressed model weights")
    parser.add_argument("--model-name", type=str,
                       default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="HuggingFace model name for architecture")
    args = parser.parse_args()

    # Expand paths
    import glob
    import os

    original_path = os.path.expanduser(args.original)
    if '*' in original_path:
        matches = glob.glob(original_path)
        if matches:
            original_path = matches[0]
        else:
            print(f"No match for: {original_path}")
            return

    compressed_path = os.path.expanduser(args.compressed)

    print("\n=== Inference Quality Test ===\n")
    print(f"Original:   {original_path}")
    print(f"Compressed: {compressed_path}")
    print(f"Model:      {args.model_name}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    print("\nLoading original model...")
    start = time.time()
    model_original = load_model_with_weights(args.model_name, original_path)
    print(f"  Loaded in {time.time() - start:.1f}s")

    print("\nLoading compressed model...")
    start = time.time()
    model_compressed = load_model_with_weights(args.model_name, compressed_path)
    print(f"  Loaded in {time.time() - start:.1f}s")

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In machine learning, a neural network",
        "The meaning of life is",
        "To solve this math problem: 2 + 2 =",
    ]

    print("\n" + "="*60)
    print("GENERATION COMPARISON")
    print("="*60)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt!r}")
        print("-" * 40)

        start = time.time()
        output_original = generate_text(model_original, tokenizer, prompt, max_new_tokens=30)
        time_original = time.time() - start

        start = time.time()
        output_compressed = generate_text(model_compressed, tokenizer, prompt, max_new_tokens=30)
        time_compressed = time.time() - start

        print(f"Original ({time_original:.2f}s):")
        print(f"  {output_original}")
        print(f"Compressed ({time_compressed:.2f}s):")
        print(f"  {output_compressed}")

        # Check if outputs match
        match = "MATCH" if output_original == output_compressed else "DIFFER"
        print(f"Status: {match}")

    print("\n" + "="*60)
    print("LOGITS COMPARISON")
    print("="*60)

    for prompt in test_prompts[:3]:  # Just first 3 for logits
        print(f"\nPrompt: {prompt!r}")
        metrics = compare_logits(model_original, model_compressed, tokenizer, prompt)
        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        print(f"  MSE: {metrics['mse']:.6e}")
        print(f"  Top-5 Agreement: {metrics['top5_agreement']*100:.0f}%")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Aggregate metrics
    all_metrics = [compare_logits(model_original, model_compressed, tokenizer, p) for p in test_prompts]
    avg_cosine = sum(m['cosine_similarity'] for m in all_metrics) / len(all_metrics)
    avg_top5 = sum(m['top5_agreement'] for m in all_metrics) / len(all_metrics)

    print(f"\nAverage Logit Cosine Similarity: {avg_cosine:.6f}")
    print(f"Average Top-5 Token Agreement: {avg_top5*100:.0f}%")

    if avg_cosine > 0.99:
        print("\nQuality: EXCELLENT - Outputs should be nearly identical")
    elif avg_cosine > 0.95:
        print("\nQuality: GOOD - Minor differences expected")
    elif avg_cosine > 0.90:
        print("\nQuality: ACCEPTABLE - Some differences expected")
    else:
        print("\nQuality: DEGRADED - Significant differences expected")

if __name__ == "__main__":
    main()
