"""Overfit a fixed train-only subset to verify the HAN training path."""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from essay_scoring.data import DEFAULT_DATA_PATH, load_training_split, manifest_path
from essay_scoring.models.han import (
    MAX_SENTENCES, MAX_WORDS, HANRegressor, _encode, _pad_batch, _sentences,
    training_device, truncation_statistics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    if args.samples < 2 or args.steps < 1:
        parser.error("Need at least two samples and one optimizer step.")
    if args.run_dir.exists():
        parser.error("Choose a new run directory; existing runs are never overwritten.")
    split, fingerprint = load_training_split(args.data, manifest_path(1), 1)
    essays = split["train"][:args.samples]
    torch.manual_seed(42)
    device = training_device()
    sentences = [_sentences(essay.text, MAX_SENTENCES, MAX_WORDS) for essay in essays]
    counts = Counter(word for essay in sentences for sentence in essay for word in sentence)
    vocabulary = {word: i for i, (word, _) in enumerate(counts.most_common(30_000), 2)}
    tokens = _pad_batch([_encode(s, vocabulary) for s in sentences]).to(device)
    targets = torch.tensor([(essay.score - 2) / 10 for essay in essays], device=device)
    model = HANRegressor(len(vocabulary) + 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.eval()
    with torch.no_grad():
        predicted = model(tokens)
        initial = F.mse_loss(predicted, targets).item()
        padding_delta = (predicted - model(F.pad(tokens, (0, 2, 0, 2)))).abs().max().item()
    if predicted.shape != targets.shape or not torch.isfinite(predicted).all():
        raise ValueError("Invalid regression output shape or nonfinite scores.")
    if not torch.all((predicted >= 0) & (predicted <= 1)) or padding_delta > 1e-6:
        raise ValueError("Regression bounds or padding invariance failed.")
    constant_loss = ((targets - targets.mean()) ** 2).mean().item()
    threshold = min(initial * 0.25, constant_loss * 0.25)
    history = [{"step": 0, "train_loss": None, "eval_train_loss": initial}]
    print(f"Device {device}; shape {list(tokens.shape)}; initial MSE {initial:.6f}; "
          f"constant-predictor MSE {constant_loss:.6f}; pass threshold {threshold:.6f}", flush=True)
    start = time.perf_counter()
    gradient_norm = None
    final = initial
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        loss = F.mse_loss(model(tokens), targets)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}.")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise ValueError(f"Missing/nonfinite gradient at step {step}.")
        if step == 1:
            gradient_norm = sum(g.square().sum().item() for g in grads) ** 0.5
            before = model.score.weight.detach().clone()
        optimizer.step()
        if step == 1 and (gradient_norm == 0 or torch.equal(before, model.score.weight)):
            raise ValueError("Optimizer did not update the regression head.")
        if step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                final = F.mse_loss(model(tokens), targets).item()
            history.append({"step": step, "train_loss": loss.item(), "eval_train_loss": final})
            print(f"Step {step}/{args.steps} | train MSE {loss.item():.6f} | "
                  f"eval train MSE {final:.6f}", flush=True)
            if step >= 40 and final < threshold:
                break
    passed = final < threshold
    report = {
        "status": "PASS" if passed else "FAIL", "seed": 42, "device": str(device),
        "dataset_sha256": fingerprint,
        "manifest_sha256": hashlib.sha256(manifest_path(1).read_bytes()).hexdigest(),
        "essay_ids": [essay.essay_id for essay in essays], "scope": "first train IDs in frozen manifest",
        "samples": len(essays), "steps": step, "learning_rate": 1e-3, "optimizer": "AdamW",
        "input_shape": list(tokens.shape), "output_shape": list(predicted.shape),
        "vocabulary_size": len(vocabulary), "initial_normalized_mse": initial,
        "final_normalized_mse": final, "constant_predictor_normalized_mse": constant_loss,
        "pass_threshold": threshold, "first_gradient_norm": gradient_norm,
        "finite_loss_and_gradients": True, "optimizer_updated_head": True,
        "padding_max_difference": padding_delta, "training_seconds": time.perf_counter() - start,
        "train_truncation": truncation_statistics(split["train"]),
        "test_content_and_labels_loaded": False,
    }
    args.run_dir.mkdir(parents=True)
    (args.run_dir / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("step", "train_loss", "eval_train_loss"))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit("HAN failed the train-only overfit criterion.")


if __name__ == "__main__":
    main()
