from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_processed_dataset, stratified_train_test_split
from evaluation import evaluate_and_save
from neural_network import load_model


MODEL_LABELS = {
    "baseline": "Базовая модель",
    "extended": "Расширенная аугментация",
    "adapted": "Адаптированная модель",
    "final": "Итоговая модель",
    "wide": "Широкий пул аугментации",
    "pool": "Широкий пул аугментации",
    "balanced64": "Сбалансированная модель 64",
    "balanced128": "Сбалансированная модель 128",
    "domain": "Domain-adaptation модель",
}


def parse_model(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Используй МЕТКА=ПУТЬ")
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def pretty_label(label: str) -> str:
    return MODEL_LABELS.get(label.lower(), label)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сравнение нескольких моделей на одной тестовой выборке пользовательского почерка."
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="Можно повторять: --model baseline=models/best_model.pkl",
    )
    parser.add_argument("--data", type=Path, default=Path("custom_dataset"))
    parser.add_argument("--cache", type=Path, default=Path("cache/custom_32.npz"))
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=Path("results_comparison"))
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, _paths = load_processed_dataset(
        args.data,
        cache_path=args.cache,
        rebuild_cache=False,
    )
    _train_idx, test_idx = stratified_train_test_split(
        y,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    X_test, y_test = X[test_idx], y[test_idx]

    rows = []
    args.results.mkdir(parents=True, exist_ok=True)

    for label, model_path in args.model:
        nn, _metadata = load_model(model_path)
        display = pretty_label(label)
        out = args.results / label
        metrics = evaluate_and_save(
            nn,
            X_test,
            y_test,
            out,
            f"тестовая выборка пользовательского почерка — {display}",
        )
        rows.append({
            "model_key": label,
            "model": display,
            "samples": len(y_test),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_roc_auc": metrics["macro_roc_auc"],
        })

    with (args.results / "сравнение_моделей.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "модель",
            "количество_изображений",
            "доля_верных_ответов",
            "макро_точность_положительных_предсказаний",
            "макро_полнота",
            "макро_f1",
            "взвешенный_f1",
            "макро_roc_auc",
        ])
        for row in rows:
            writer.writerow([
                row["model"],
                row["samples"],
                f"{row['accuracy']:.6f}",
                f"{row['macro_precision']:.6f}",
                f"{row['macro_recall']:.6f}",
                f"{row['macro_f1']:.6f}",
                f"{row['weighted_f1']:.6f}",
                f"{row['macro_roc_auc']:.6f}",
            ])

    labels = [r["model"] for r in rows]
    f1 = [100 * r["macro_f1"] for r in rows]
    accuracy = [100 * r["accuracy"] for r in rows]

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.8), facecolor="white")

    bars1 = ax.bar(
        x - width / 2, accuracy, width,
        label="Доля верных ответов",
        facecolor="white", edgecolor="black", hatch="///",
        linewidth=1.0,
    )
    bars2 = ax.bar(
        x + width / 2, f1, width,
        label="Макро-F1",
        facecolor="0.65", edgecolor="black", hatch="...",
        linewidth=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Значение, %")
    ax.set_title("Сравнение моделей на одной тестовой выборке")
    ax.legend()
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="0.80")
    ax.set_ylim(max(0, min(accuracy + f1) - 8), 100)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(
        args.results / "сравнение_моделей.png",
        dpi=240, facecolor="white", bbox_inches="tight"
    )
    plt.close(fig)

    print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
    for row in rows:
        print(
            f"{row['model']}: "
            f"доля верных ответов={row['accuracy']:.4f}, "
            f"макро-F1={row['macro_f1']:.4f}, "
            f"ROC-AUC={row['macro_roc_auc']:.4f}"
        )
    print(f"Результаты сохранены: {args.results.resolve()}")


if __name__ == "__main__":
    main()
