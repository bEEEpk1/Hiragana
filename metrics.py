from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from config import ROMAJI


def _select_unicode_font() -> str | None:
    """Выбирает шрифт с поддержкой кириллицы и японских символов, если он установлен."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Arial Unicode MS",
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Yu Gothic",
        "YuGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "MS Gothic",
    ]
    for name in candidates:
        if name in available:
            return name
    return None

def _print_style() -> None:
    """Единый стиль всех иллюстраций для отчета и ЧБ-печати."""
    unicode_font = _select_unicode_font()
    if unicode_font:
        plt.rcParams["font.family"] = unicode_font

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "grid.color": "0.85",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 120,
    })


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict:
    per_class = []
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    supports = np.asarray([m["support"] for m in per_class], dtype=float)
    total_support = float(supports.sum())
    weights = supports / total_support if total_support else np.zeros_like(supports)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean([m["precision"] for m in per_class])),
        "macro_recall": float(np.mean([m["recall"] for m in per_class])),
        "macro_f1": float(np.mean([m["f1"] for m in per_class])),
        "weighted_precision": float(np.sum(weights * np.asarray([m["precision"] for m in per_class]))),
        "weighted_recall": float(np.sum(weights * np.asarray([m["recall"] for m in per_class]))),
        "weighted_f1": float(np.sum(weights * np.asarray([m["f1"] for m in per_class]))),
        "per_class": per_class,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    return metrics_from_confusion_matrix(confusion_matrix(y_true, y_pred, num_classes))


def roc_curve_binary(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order].astype(np.int64)
    positives = int(y_sorted.sum())
    negatives = int(len(y_sorted) - positives)

    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    sorted_scores = scores[order]
    previous_score = None

    for label, score in zip(y_sorted, sorted_scores):
        if previous_score is not None and score != previous_score:
            tpr.append(tp / positives)
            fpr.append(fp / negatives)
        if label == 1:
            tp += 1
        else:
            fp += 1
        previous_score = score

    tpr.append(1.0)
    fpr.append(1.0)
    fpr_arr = np.asarray(fpr, dtype=float)
    tpr_arr = np.asarray(tpr, dtype=float)
    auc = float(np.trapezoid(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc


def roc_ovr(y_true: np.ndarray, probs: np.ndarray, num_classes: int):
    curves = []
    for class_id in range(num_classes):
        binary = (y_true == class_id).astype(np.int64)
        fpr, tpr, auc = roc_curve_binary(binary, probs[:, class_id])
        curves.append({"class_id": class_id, "fpr": fpr, "tpr": tpr, "auc": auc})
    macro_auc = float(np.mean([curve["auc"] for curve in curves]))
    return curves, macro_auc


def save_per_class_metrics(metrics: dict, romaji: list[str], kana: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "номер_класса",
            "ромадзи",
            "хирагана",
            "точность_положительных_предсказаний",
            "полнота",
            "f1",
            "количество_примеров",
        ])
        for class_id, row in enumerate(metrics["per_class"]):
            writer.writerow([
                class_id,
                romaji[class_id],
                kana[class_id],
                f"{row['precision']:.6f}",
                f"{row['recall']:.6f}",
                f"{row['f1']:.6f}",
                row["support"],
            ])


def _plot_learning_pair(history: dict, output: Path, best_epoch: int | None = None, max_epoch: int | None = None) -> None:
    _print_style()
    output.parent.mkdir(parents=True, exist_ok=True)

    total = len(history["train_loss"])
    stop = total if max_epoch is None else min(total, max_epoch)
    epochs = np.arange(1, stop + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    axes[0].plot(
        epochs, history["train_loss"][:stop],
        color="black", linestyle="-", linewidth=1.6,
        label="Обучающая выборка",
    )
    axes[0].plot(
        epochs, history["val_loss"][:stop],
        color="0.45", linestyle="--", linewidth=1.6,
        label="Валидационная выборка",
    )
    axes[0].set_title("Изменение функции потерь")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Значение функции потерь")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        epochs, history["train_accuracy"][:stop],
        color="black", linestyle="-", linewidth=1.5,
        label="Доля верных ответов на обучении",
    )
    axes[1].plot(
        epochs, history["val_accuracy"][:stop],
        color="0.35", linestyle="--", linewidth=1.5,
        label="Доля верных ответов на валидации",
    )
    axes[1].plot(
        epochs, history["val_macro_f1"][:stop],
        color="0.65", linestyle="-.", linewidth=1.5,
        label="Макро-F1 на валидации",
    )
    axes[1].set_title("Изменение качества классификации")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Значение метрики")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    if best_epoch is not None and best_epoch <= stop:
        for ax in axes:
            ax.axvline(best_epoch, color="0.15", linestyle=":", linewidth=1.0)
        axes[1].text(
            best_epoch,
            axes[1].get_ylim()[0],
            f"  лучшая эпоха: {best_epoch}",
            rotation=90,
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(
    history: dict,
    output: Path,
    best_epoch: int | None = None,
) -> None:
    """Полный график обучения и отдельный увеличенный фрагмент первых 50 эпох."""
    output = Path(output)
    _plot_learning_pair(history, output, best_epoch=best_epoch)

    zoom_name = output.with_name(output.stem + "_первые_50_эпох" + output.suffix)
    _plot_learning_pair(
        history,
        zoom_name,
        best_epoch=best_epoch,
        max_epoch=50,
    )


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output: Path) -> None:
    """Матрица ошибок в ЧБ-гамме; подписываются только ненулевые клетки."""
    _print_style()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.5, 13.0))
    image = ax.imshow(cm, interpolation="nearest", cmap="Greys", vmin=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Количество изображений")

    ax.set_title("Матрица ошибок классификации")
    ax.set_xlabel("Предсказанный символ")
    ax.set_ylabel("Истинный символ")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    vmax = max(float(cm.max()), 1.0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            if value == 0:
                continue
            text_color = "white" if value >= 0.55 * vmax else "black"
            ax.text(j, i, str(value), ha="center", va="center",
                    fontsize=6.5, color=text_color)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="0.88", linestyle="-", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_error_map(cm: np.ndarray, labels: list[str], output: Path) -> None:
    """Отдельная карта только ошибочных классификаций: диагональ занулена."""
    _print_style()
    output.parent.mkdir(parents=True, exist_ok=True)

    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    fig, ax = plt.subplots(figsize=(14.5, 13.0))
    image = ax.imshow(errors, interpolation="nearest", cmap="Greys", vmin=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Количество ошибок")

    ax.set_title("Карта ошибочных классификаций")
    ax.set_xlabel("Предсказанный символ")
    ax.set_ylabel("Истинный символ")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    vmax = max(float(errors.max()), 1.0)
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            value = int(errors[i, j])
            if value == 0:
                continue
            text_color = "white" if value >= 0.55 * vmax else "black"
            ax.text(j, i, str(value), ha="center", va="center",
                    fontsize=7, color=text_color)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="0.90", linestyle="-", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    curves: list[dict],
    labels: list[str],
    output: Path,
    max_classes: int = 6,
) -> None:
    """ЧБ ROC-кривые нескольких наиболее сложных классов."""
    _print_style()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_romaji = [
        "e",  # え
        "ho",  # ほ
        "ta",  # た
        "a",  # あ
        "u",  # う
        "ha",  # は
    ]

    selected_ids = [ROMAJI.index(name) for name in selected_romaji]

    hardest = [
        curve
        for curve in curves
        if curve["class_id"] in selected_ids
    ]

    line_styles = ["-", "--", "-.", ":", "-", "--"]
    markers = ["o", "s", "^", "D", "x", "+"]

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for idx, curve in enumerate(hardest):
        i = curve["class_id"]
        # Маркеры редкие, чтобы кривая не превращалась в шум.
        markevery = max(1, len(curve["fpr"]) // 8)
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            color=str(0.10 + 0.12 * idx),
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=1.5,
            marker=markers[idx % len(markers)],
            markersize=3.5,
            markevery=markevery,
            label=f"{labels[i]} (AUC = {curve['auc']:.6f})",
        )

    ax.plot(
        [0, 1], [0, 1],
        color="0.70", linestyle=":", linewidth=1.2,
        label="Случайный классификатор",
    )
    ax.set_xlabel("Доля ложноположительных решений (FPR)")
    ax.set_ylabel("Доля истинноположительных решений (TPR)")
    ax.set_title("ROC-кривые: схема «один против остальных»")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_domain_adaptation_curves(
    history: dict,
    output: Path,
    best_epoch: int | None = None,
) -> None:
    """Печатный график обучения с двумя validation-доменами и LR."""
    _print_style()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(
        1,
        len(history["selection_score"]) + 1,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.5, 5.2),
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        color="black",
        linestyle="-",
        linewidth=1.4,
        label="Обучающая выборка",
    )
    axes[0].plot(
        epochs,
        history["kaggle_val_loss"],
        color="0.4",
        linestyle="--",
        linewidth=1.3,
        label="Исходная валидация",
    )
    axes[0].plot(
        epochs,
        history["custom_val_loss"],
        color="0.7",
        linestyle="-.",
        linewidth=1.3,
        label="Пользовательская валидация",
    )
    axes[0].set_title("Функция потерь")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Значение функции потерь")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        epochs,
        history["kaggle_val_macro_f1"],
        color="0.25",
        linestyle="--",
        linewidth=1.4,
        label="Макро-F1: исходная валидация",
    )
    axes[1].plot(
        epochs,
        history["custom_val_macro_f1"],
        color="0.65",
        linestyle="-.",
        linewidth=1.4,
        label="Макро-F1: пользовательская валидация",
    )
    axes[1].plot(
        epochs,
        history["selection_score"],
        color="black",
        linestyle="-",
        linewidth=1.6,
        label="Комбинированный критерий",
    )
    axes[1].set_title("Качество классификации")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Значение метрики")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    axes[2].plot(
        epochs,
        history["learning_rate"],
        color="black",
        linestyle="-",
        linewidth=1.5,
    )
    axes[2].set_title("Изменение скорости обучения")
    axes[2].set_xlabel("Эпоха")
    axes[2].set_ylabel("Скорость обучения")
    axes[2].grid(True)
    axes[2].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, -3),
    )

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(
                best_epoch,
                color="0.25",
                linestyle=":",
                linewidth=1.0,
            )
        axes[1].text(
            best_epoch,
            axes[1].get_ylim()[0],
            f"  лучшая эпоха: {best_epoch}",
            rotation=90,
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)
