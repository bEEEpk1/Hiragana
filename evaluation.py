from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from config import KANA, NUM_CLASSES, ROMAJI
from metrics import (
    classification_metrics,
    confusion_matrix,
    plot_confusion_matrix,
    plot_error_map,
    plot_roc_curves,
    roc_ovr,
    save_per_class_metrics,
)


def evaluate_probabilities_and_save(
    probs: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    dataset_name: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Ожидались вероятности формы (N, {NUM_CLASSES}), "
            f"получено {probs.shape}."
        )
    if len(probs) != len(y):
        raise ValueError(
            "Число строк вероятностей не совпадает с числом меток."
        )

    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y, y_pred, NUM_CLASSES)
    metrics = classification_metrics(y, y_pred, NUM_CLASSES)
    curves, macro_auc = roc_ovr(y, probs, NUM_CLASSES)
    metrics["macro_roc_auc"] = macro_auc

    save_per_class_metrics(
        metrics,
        ROMAJI,
        KANA,
        output_dir / "метрики_по_классам.csv",
    )
    plot_confusion_matrix(
        cm,
        KANA,
        output_dir / "матрица_ошибок.png",
    )
    plot_error_map(
        cm,
        KANA,
        output_dir / "карта_ошибок.png",
    )
    plot_roc_curves(
        curves,
        KANA,
        output_dir / "roc_кривые.png",
    )

    with (output_dir / "roc_auc_по_классам.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["номер_класса", "ромадзи", "хирагана", "auc"])
        for curve in curves:
            i = curve["class_id"]
            writer.writerow([
                i,
                ROMAJI[i],
                KANA[i],
                f"{curve['auc']:.6f}",
            ])

    with (output_dir / "предсказания.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "номер_изображения",
            "истинный_класс_ромадзи",
            "истинный_символ",
            "предсказанный_класс_ромадзи",
            "предсказанный_символ",
            "оценка_лучшего_класса",
            "верно",
        ])
        for row_idx, (true_id, pred_id, row_probs) in enumerate(
            zip(y, y_pred, probs)
        ):
            pred_id = int(pred_id)
            true_id = int(true_id)
            writer.writerow([
                row_idx,
                ROMAJI[true_id],
                KANA[true_id],
                ROMAJI[pred_id],
                KANA[pred_id],
                f"{float(row_probs[pred_id]):.6f}",
                "да" if true_id == pred_id else "нет",
            ])

    text = (
        f"Набор данных: {dataset_name}\n"
        f"Количество изображений: {len(y)}\n"
        f"Доля верных ответов: {metrics['accuracy']:.6f}\n"
        f"Макро-точность положительных предсказаний: {metrics['macro_precision']:.6f}\n"
        f"Макро-полнота: {metrics['macro_recall']:.6f}\n"
        f"Макро-F1: {metrics['macro_f1']:.6f}\n"
        f"Взвешенный F1: {metrics['weighted_f1']:.6f}\n"
        f"Макро ROC-AUC: {macro_auc:.6f}\n"
    )
    (output_dir / "метрики.txt").write_text(text, encoding="utf-8")
    return metrics


def evaluate_and_save(
    nn,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    dataset_name: str,
) -> dict:
    return evaluate_probabilities_and_save(
        nn.predict_proba(X),
        y,
        output_dir,
        dataset_name,
    )
