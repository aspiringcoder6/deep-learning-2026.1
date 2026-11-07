"""Train, evaluate, and use prompt-specific essay scoring models."""

import argparse
import csv
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from essay_scoring.data import (
    DEFAULT_DATA_PATH,
    SCORE_RANGES,
    SPLIT_VERSION,
    load_asap,
    split_essays,
)
from essay_scoring.metrics import clip_scores, evaluate_scores, round_scores
from essay_scoring.models.han import fit_han, predict_han
from essay_scoring.models.ridge import fit_ridge


def train(args: argparse.Namespace) -> None:
    if args.run_dir.exists():
        raise ValueError(f"Choose a new run directory. {args.run_dir} already exists.")

    essays, fingerprint = load_asap(args.data, args.essay_set)
    split = split_essays(essays, args.seed)
    bounds = SCORE_RANGES[args.essay_set]
    validation = split["validation"]
    settings = {}
    if args.model == "ridge":
        model = fit_ridge(split["train"])
        predicted = model.predict([essay.text for essay in validation]).tolist()
        metrics = evaluate_scores([essay.score for essay in validation], predicted, bounds)
    else:
        settings = {"epochs": args.epochs, "batch_size": args.batch_size}
        model, metrics = fit_han(split["train"], validation, bounds, seed=args.seed, **settings)

    run = {
        "essay_set": args.essay_set,
        "model": args.model,
        "model_settings": settings,
        "seed": args.seed,
        "split_version": SPLIT_VERSION,
        "dataset_sha256": fingerprint,
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
            torch.save(model, staging / "model.pt")
        (staging / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        staging.rename(args.run_dir)

    print(json.dumps({"run_dir": str(args.run_dir), "validation_metrics": metrics}, indent=2))


def _read_run(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _predict(run_dir: Path, run: dict, texts: list[str]) -> list[float]:
    if run["model"] == "ridge":
        with (run_dir / "model.pkl").open("rb") as file:
            model = pickle.load(file)
        return model.predict(texts).tolist()
    return predict_han(run_dir / "model.pt", texts, SCORE_RANGES[run["essay_set"]])


def evaluate(args: argparse.Namespace) -> None:
    run = _read_run(args.run_dir)
    essay_set = run["essay_set"]
    essays, fingerprint = load_asap(args.data, essay_set)
    if fingerprint != run["dataset_sha256"]:
        raise ValueError("Use the same ASAP TSV file used for training.")

    by_id = {essay.essay_id: essay for essay in essays}
    test = [by_id[essay_id] for essay_id in run["splits"]["test"]]
    predicted = _predict(args.run_dir, run, [essay.text for essay in test])
    metrics = evaluate_scores([essay.score for essay in test], predicted, SCORE_RANGES[essay_set])
    result = {"essay_set": essay_set, "model": run["model"], "test_metrics": metrics}
    print(json.dumps(result, indent=2))


def predict(args: argparse.Namespace) -> None:
    text = args.essay_file.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("The essay file is empty.")

    run = _read_run(args.run_dir)
    bounds = SCORE_RANGES[run["essay_set"]]
    clipped = clip_scores(_predict(args.run_dir, run, [text]), bounds)
    result = {
        "essay_set": run["essay_set"],
        "model": run["model"],
        "prediction": float(clipped[0]),
        "score": int(round_scores(clipped)[0]),
        "score_range": bounds,
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
    train_parser = commands.add_parser("train", help="Fit a model and report validation metrics.")
    train_parser.add_argument("--essay-set", type=int, choices=SCORE_RANGES, required=True)
    train_parser.add_argument("--model", choices=("ridge", "han"), required=True)
    train_parser.add_argument("--seed", type=int, default=42, help="Split and training seed (default 42).")
    train_parser.add_argument("--epochs", type=_positive_int, default=5, help="HAN epochs (default 5).")
    train_parser.add_argument("--batch-size", type=_positive_int, default=16, help="HAN batch size (default 16).")
    train_parser.set_defaults(handler=train)

    evaluate_parser = commands.add_parser("evaluate", help="Score the held-out test split for a saved run.")
    evaluate_parser.set_defaults(handler=evaluate)

    predict_parser = commands.add_parser("predict", help="Score one UTF-8 essay file using a saved run.")
    predict_parser.add_argument("--essay-file", type=Path, required=True)
    predict_parser.set_defaults(handler=predict)

    for command in (train_parser, evaluate_parser):
        command.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    for command in (train_parser, evaluate_parser, predict_parser):
        command.add_argument("--run-dir", type=Path, required=True)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
