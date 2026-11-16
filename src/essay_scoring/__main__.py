"""Train, evaluate, and use prompt-specific essay scoring models."""

import argparse
import csv
import hashlib
import json
import pickle
import statistics
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from essay_scoring.data import (
    DEFAULT_DATA_PATH,
    SCORE_RANGES,
    SPLIT_VERSION,
    create_manifest,
    load_asap,
    manifest_path,
    read_manifest,
)
from essay_scoring.metrics import clip_scores, evaluate_scores, round_scores


def _manifest_file(args: argparse.Namespace, essay_set: int) -> Path:
    return args.manifest or manifest_path(essay_set)


def prepare(args: argparse.Namespace) -> None:
    manifest, essays = create_manifest(args.data, args.essay_set, args.seed)
    target = _manifest_file(args, args.essay_set)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to replace a different frozen manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")

    by_id = {essay.essay_id: essay for essay in essays}
    train_essays = [by_id[essay_id] for essay_id in manifest["splits"]["train"]]
    lengths = sorted(len(essay.text.split()) for essay in train_essays)
    counts = Counter(essay.score for essay in train_essays)
    summary = {
        "essay_set": args.essay_set,
        "dataset_sha256": manifest["dataset_sha256"],
        "source_rows": manifest["audit"]["source_rows"],
        "clean_rows": len(essays),
        "statistics_scope": "train only; test labels excluded",
        "score_range": {"min": SCORE_RANGES[args.essay_set][0], "max": SCORE_RANGES[args.essay_set][1]},
        "observed_scores": {
            str(score): counts.get(score, 0)
            for score in range(SCORE_RANGES[args.essay_set][0], SCORE_RANGES[args.essay_set][1] + 1)
        },
        "word_count": {
            "min": lengths[0],
            "median": statistics.median(lengths),
            "mean": round(statistics.mean(lengths), 2),
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": lengths[-1],
        },
        "split_counts": {name: len(ids) for name, ids in manifest["splits"].items()},
        "duplicate_text_across_splits": 0,
    }
    report_path = Path("data/analysis") / f"set-{args.essay_set}-summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(target), "summary": str(report_path), **summary}, indent=2))


def train(args: argparse.Namespace) -> None:
    if args.run_dir.exists():
        raise ValueError(f"Choose a new run directory. {args.run_dir} already exists.")

    essays, fingerprint = load_asap(args.data, args.essay_set)
    manifest_file = _manifest_file(args, args.essay_set)
    split = read_manifest(manifest_file, essays, fingerprint, args.essay_set)
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if args.seed != manifest["seed"]:
        raise ValueError("Training seed must match the frozen split seed.")
    bounds = SCORE_RANGES[args.essay_set]
    validation = split["validation"]
    settings = {}
    if args.model == "ridge":
        from essay_scoring.models.ridge import fit_ridge

        model = fit_ridge(split["train"])
        predicted = model.predict([essay.text for essay in validation]).tolist()
        metrics = evaluate_scores([essay.score for essay in validation], predicted, bounds)
    else:
        from essay_scoring.models.han import fit_han

        settings = {"epochs": args.epochs, "batch_size": args.batch_size}
        model, metrics = fit_han(split["train"], validation, bounds, seed=args.seed, **settings)

    run = {
        "essay_set": args.essay_set,
        "model": args.model,
        "model_settings": settings,
        "seed": args.seed,
        "split_version": SPLIT_VERSION,
        "dataset_sha256": fingerprint,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "splits": {name: [essay.essay_id for essay in group] for name, group in split.items()},
        "validation_metrics": metrics,
    }

    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.run_dir.parent, prefix=".essay-run-") as staging_dir:
        staging = Path(staging_dir)
        if args.model == "ridge":
            with (staging / "model.pkl").open("wb") as file:
                pickle.dump(model, file)
        else:
            import torch

            torch.save(model, staging / "model.pt")
        (staging / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        staging.rename(args.run_dir)

    print(json.dumps({"run_dir": str(args.run_dir), "validation_metrics": metrics}, indent=2))


def _read_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _model_file(run_dir: Path, run: dict) -> Path:
    return run_dir / ("model.pkl" if run["model"] == "ridge" else "model.pt")


def finalize(args: argparse.Namespace) -> None:
    """Record one validation-selected run before opening the test set."""
    if not args.note.strip():
        raise ValueError("Final model selection needs a nonempty validation-based note.")
    run = _read_run(args.run_dir)
    selection = {
        "essay_set": run["essay_set"],
        "model": run["model"],
        "selection_note": args.note,
        "validation_metrics": run["validation_metrics"],
        "run_sha256": hashlib.sha256((args.run_dir / "run.json").read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(_model_file(args.run_dir, run).read_bytes()).hexdigest(),
    }
    path = args.run_dir.parent / "final-selection.json"
    rendered = json.dumps(selection, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"A different final model is already recorded at {path}.")
    path.write_text(rendered, encoding="utf-8")
    print(json.dumps({"selection_file": str(path), "essay_set": run["essay_set"], "model": run["model"]}, indent=2))


def _predict(run_dir: Path, run: dict, texts: list[str]) -> list[float]:
    if run["model"] == "ridge":
        with (run_dir / "model.pkl").open("rb") as file:
            model = pickle.load(file)
        return model.predict(texts).tolist()
    from essay_scoring.models.han import predict_han

    return predict_han(run_dir / "model.pt", texts, SCORE_RANGES[run["essay_set"]])


def evaluate(args: argparse.Namespace) -> None:
    if not args.final_model:
        raise ValueError("Test is sealed. Finalize one run on validation before evaluation.")
    run = _read_run(args.run_dir)
    selection_path = args.run_dir.parent / "final-selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("run_sha256") != hashlib.sha256((args.run_dir / "run.json").read_bytes()).hexdigest()
            or selection.get("model_sha256") != hashlib.sha256(_model_file(args.run_dir, run).read_bytes()).hexdigest()):
        raise ValueError("The selected run or checkpoint changed after finalization.")
    essay_set = run["essay_set"]
    essays, fingerprint = load_asap(args.data, essay_set)
    if fingerprint != run["dataset_sha256"]:
        raise ValueError("Use the same ASAP TSV file used for training.")
    manifest_file = _manifest_file(args, essay_set)
    if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != run["manifest_sha256"]:
        raise ValueError("Use the exact split manifest saved with the trained run.")
    split = read_manifest(manifest_file, essays, fingerprint, essay_set)
    if {name: [essay.essay_id for essay in group] for name, group in split.items()} != run["splits"]:
        raise ValueError("Run and manifest split IDs differ.")

    test = split["test"]
    predicted = _predict(args.run_dir, run, [essay.text for essay in test])
    metrics = evaluate_scores([essay.score for essay in test], predicted, SCORE_RANGES[essay_set])
    result = {"essay_set": essay_set, "model": run["model"], "test_metrics": metrics}
    print(json.dumps(result, indent=2))


def predict(args: argparse.Namespace) -> None:
    text = args.essay_file.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("The essay file is empty.")

    run = _read_run(args.run_dir)
    if args.essay_set != run["essay_set"]:
        raise ValueError("Input essay set does not match the trained run.")
    bounds = SCORE_RANGES[run["essay_set"]]
    clipped = clip_scores(_predict(args.run_dir, run, [text]), bounds)
    result = {
        "essay_set": run["essay_set"],
        "model": run["model"],
        "predicted_score": float(clipped[0]),
        "rounded_score": int(round_scores(clipped)[0]),
        "score_range": {"min": bounds[0], "max": bounds[1]},
    }
    print(json.dumps(result, indent=2))


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m essay_scoring", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Audit labeled data and freeze a split manifest.")
    prepare_parser.add_argument("--essay-set", type=int, choices=SCORE_RANGES, required=True)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.set_defaults(handler=prepare)
    train_parser = commands.add_parser("train", help="Fit a model and report validation metrics.")
    train_parser.add_argument("--essay-set", type=int, choices=SCORE_RANGES, required=True)
    train_parser.add_argument("--model", choices=("ridge", "han"), required=True)
    train_parser.add_argument("--seed", type=int, default=42, help="Split and training seed (default 42).")
    train_parser.add_argument("--epochs", type=_positive_int, default=5, help="HAN epochs (default 5).")
    train_parser.add_argument("--batch-size", type=_positive_int, default=16, help="HAN batch size (default 16).")
    train_parser.set_defaults(handler=train)

    evaluate_parser = commands.add_parser("evaluate", help="Score the held-out test split for a saved run.")
    evaluate_parser.add_argument("--final-model", action="store_true", help="Explicitly open test after model selection.")
    evaluate_parser.set_defaults(handler=evaluate)

    finalize_parser = commands.add_parser("finalize", help="Freeze the validation-selected run before test evaluation.")
    finalize_parser.add_argument("--note", required=True, help="Why this run was selected on validation.")
    finalize_parser.set_defaults(handler=finalize)

    predict_parser = commands.add_parser("predict", help="Score one UTF-8 essay file using a saved run.")
    predict_parser.add_argument("--essay-set", type=int, choices=SCORE_RANGES, required=True)
    predict_parser.add_argument("--essay-file", type=Path, required=True)
    predict_parser.set_defaults(handler=predict)

    for command in (prepare_parser, train_parser, evaluate_parser):
        command.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
        command.add_argument("--manifest", type=Path, help="Frozen split manifest path.")
    for command in (train_parser, evaluate_parser, finalize_parser, predict_parser):
        command.add_argument("--run-dir", type=Path, required=True)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
