from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_utils import load_processed_dataset, stratified_train_test_split
from evaluation import evaluate_and_save
from neural_network import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Оценка модели на пользовательском наборе рукописных символов."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("custom_dataset"))
    parser.add_argument("--cache", type=Path, default=Path("cache/custom_32.npz"))
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="test",
    )
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=Path("results_custom_eval"))
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    nn, _metadata = load_model(args.model)
    X, y, _paths = load_processed_dataset(
        args.data,
        cache_path=args.cache,
        rebuild_cache=args.rebuild_cache,
    )

    if args.split == "all":
        X_eval, y_eval = X, y
    else:
        train_idx, test_idx = stratified_train_test_split(
            y,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        idx = train_idx if args.split == "train" else test_idx
        X_eval, y_eval = X[idx], y[idx]

    metrics = evaluate_and_save(
        nn,
        X_eval,
        y_eval,
        args.results,
        f"пользовательский набор — {args.split}",
    )
    print((args.results / "метрики.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
