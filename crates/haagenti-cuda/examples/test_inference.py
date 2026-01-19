#!/usr/bin/env python3
"""
Test inference quality by comparing original vs compressed model outputs.

This script compares:
1. Token logits (softmax probabilities)
2. Top-k token predictions
3. Generated text output

Usage:
    python3 test_inference.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# Check for required packages
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("This test requires: pip install torch transformers")
    sys.exit(1)


def find_qwen_model():
    """Find Qwen model in HuggingFace cache."""
    import os
    home = os.environ.get("HOME", "")
    cache_dir = Path(home) / ".cache/huggingface/hub"
    model_dir = cache_dir / "models--Qwen--Qwen2.5-0.5B-Instruct"

    if model_dir.exists():
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            for entry in snapshots.iterdir():
                if entry.is_dir():
                    model_file = entry / "model.safetensors"
                    if model_file.exists():
                        return entry
    return None


def load_safetensors_weights(path):
    """Load weights directly from safetensors file."""
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    return tensors


def compare_logits(original_logits, compressed_logits):
    """Compare logits between original and compressed model."""
    # Convert to float32 for comparison
    orig = original_logits.float()
    comp = compressed_logits.float()

    # Cosine similarity of logit vectors
    cos_sim = torch.nn.functional.cosine_similarity(
        orig.flatten().unsqueeze(0),
        comp.flatten().unsqueeze(0)
    ).item()

    # Top-1 accuracy
    orig_top1 = orig.argmax(dim=-1)
    comp_top1 = comp.argmax(dim=-1)
    top1_match = (orig_top1 == comp_top1).float().mean().item()

    # Top-5 accuracy
    _, orig_top5 = orig.topk(5, dim=-1)
    _, comp_top5 = comp.topk(5, dim=-1)
    top5_match = 0
    for i in range(orig_top5.shape[0]):
        for j in range(orig_top5.shape[1]):
            orig_set = set(orig_top5[i, j].tolist())
            comp_set = set(comp_top5[i, j].tolist())
            top5_match += len(orig_set & comp_set) / 5
    top5_match /= (orig_top5.shape[0] * orig_top5.shape[1])

    # KL divergence
    orig_probs = torch.softmax(orig, dim=-1)
    comp_probs = torch.softmax(comp, dim=-1)
    kl_div = torch.nn.functional.kl_div(
        comp_probs.log(), orig_probs, reduction='batchmean'
    ).item()

    return {
        "cosine_similarity": cos_sim,
        "top1_accuracy": top1_match,
        "top5_overlap": top5_match,
        "kl_divergence": kl_div,
    }


def main():
    print("=" * 60)
    print("HCT Inference Quality Test")
    print("=" * 60)

    # Find model paths
    model_dir = find_qwen_model()
    if model_dir is None:
        print("ERROR: Qwen2.5-0.5B-Instruct not found in cache")
        print("Run: pip install transformers && python -c \"from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')\"")
        return 1

    original_path = model_dir / "model.safetensors"
    # Try 90% retention first, fall back to 70%
    compressed_path = Path("/tmp/qwen-compressed-90pct-int4.safetensors")
    if not compressed_path.exists():
        compressed_path = Path("/tmp/qwen-compressed-70pct-int4.safetensors")

    if not compressed_path.exists():
        print(f"ERROR: Compressed model not found: {compressed_path}")
        print("Run: RETENTION=0.70 cargo run --release --example create_compressed_model -p haagenti")
        return 1

    print(f"\nOriginal:   {original_path}")
    print(f"Compressed: {compressed_path}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Load original model
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
    )
    original_model.eval()

    # Load compressed weights
    print("Loading compressed weights...")
    compressed_weights = load_safetensors_weights(str(compressed_path))

    # Create a copy of original model and swap weights
    print("Creating compressed model...")
    compressed_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
    )

    # Update model state dict with compressed weights
    state_dict = compressed_model.state_dict()
    for name, tensor in compressed_weights.items():
        if name in state_dict:
            if state_dict[name].shape == tensor.shape:
                state_dict[name] = tensor.to(state_dict[name].dtype)

    compressed_model.load_state_dict(state_dict)
    compressed_model.eval()

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "In Python, to read a file you use",
        "The square root of 144 is",
        "The chemical formula for water is",
    ]

    print("\n" + "=" * 60)
    print("Logit Comparison")
    print("=" * 60)

    all_metrics = []

    with torch.no_grad():
        for prompt in test_prompts:
            print(f"\nPrompt: \"{prompt}\"")

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")

            # Get logits from both models
            orig_outputs = original_model(**inputs)
            comp_outputs = compressed_model(**inputs)

            orig_logits = orig_outputs.logits
            comp_logits = comp_outputs.logits

            # Compare
            metrics = compare_logits(orig_logits, comp_logits)
            all_metrics.append(metrics)

            print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
            print(f"  Top-1 accuracy:    {metrics['top1_accuracy']:.4f}")
            print(f"  Top-5 overlap:     {metrics['top5_overlap']:.4f}")
            print(f"  KL divergence:     {metrics['kl_divergence']:.6f}")

            # Show top predictions
            orig_next = orig_logits[0, -1, :].argmax()
            comp_next = comp_logits[0, -1, :].argmax()

            orig_token = tokenizer.decode([orig_next])
            comp_token = tokenizer.decode([comp_next])

            print(f"  Original next token:   \"{orig_token}\"")
            print(f"  Compressed next token: \"{comp_token}\"")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    avg_cos = np.mean([m["cosine_similarity"] for m in all_metrics])
    avg_top1 = np.mean([m["top1_accuracy"] for m in all_metrics])
    avg_top5 = np.mean([m["top5_overlap"] for m in all_metrics])
    avg_kl = np.mean([m["kl_divergence"] for m in all_metrics])

    print(f"\nAverage Cosine Similarity: {avg_cos:.6f}")
    print(f"Average Top-1 Accuracy:    {avg_top1:.4f}")
    print(f"Average Top-5 Overlap:     {avg_top5:.4f}")
    print(f"Average KL Divergence:     {avg_kl:.6f}")

    # Quality assessment
    print("\nQuality Assessment:")
    if avg_cos >= 0.999 and avg_top1 >= 0.95:
        print("  EXCELLENT - Production ready")
        status = 0
    elif avg_cos >= 0.99 and avg_top1 >= 0.90:
        print("  GOOD - Suitable for most uses")
        status = 0
    elif avg_cos >= 0.95:
        print("  ACCEPTABLE - Minor quality degradation")
        status = 0
    else:
        print("  DEGRADED - Consider higher retention")
        status = 1

    # Generation test
    print("\n" + "=" * 60)
    print("Text Generation Comparison")
    print("=" * 60)

    gen_prompt = "Write a short greeting: Hello, I am"
    print(f"\nPrompt: \"{gen_prompt}\"")

    inputs = tokenizer(gen_prompt, return_tensors="pt")

    with torch.no_grad():
        orig_gen = original_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        comp_gen = compressed_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    orig_text = tokenizer.decode(orig_gen[0], skip_special_tokens=True)
    comp_text = tokenizer.decode(comp_gen[0], skip_special_tokens=True)

    print(f"\nOriginal:   {orig_text}")
    print(f"Compressed: {comp_text}")

    if orig_text == comp_text:
        print("\nGeneration: IDENTICAL")
    else:
        print("\nGeneration: DIFFERENT (may still be acceptable)")

    return status


if __name__ == "__main__":
    sys.exit(main())
