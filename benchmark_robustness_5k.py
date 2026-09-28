from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from config import IMAGE_SIZE, KANA, NUM_CLASSES, ROMAJI
from data_utils import load_processed_dataset, stratified_train_test_split
from evaluation import evaluate_probabilities_and_save
from inference_utils import adjust_stroke_width
from metrics import classification_metrics
from neural_network import load_model


GROUPS = [
    ("почти_исходные", "Почти исходные"),
    ("тонкие_штрихи", "Тонкие штрихи"),
    ("повороты_и_сдвиги", "Повороты и сдвиги"),
    ("масштаб_и_растяжение", "Масштаб и растяжение"),
    ("комбинированные", "Комбинированные"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness benchmark на 5000 synthetic test-примерах."
    )
    p.add_argument("--custom-data", type=Path, default=Path("custom_dataset"))
    p.add_argument("--custom-cache", type=Path, default=Path("cache/custom_32.npz"))
    p.add_argument("--model", type=Path, default=Path("models/best_model_final.pkl"))
    p.add_argument("--results", type=Path, default=Path("results_robustness_5k"))
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--custom-train-ratio", type=float, default=0.60)
    p.add_argument("--rebuild-cache", action="store_true")
    return p.parse_args()


def _pil(arr):
    return Image.fromarray(
        np.uint8(np.clip(arr, 0, 1) * 255),
        mode="L",
    )


def _arr(pil):
    return np.asarray(pil, dtype=np.float32) / 255.0


def shift(arr, dx, dy):
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    canvas.paste(_pil(arr), (int(dx), int(dy)))
    return _arr(canvas)


def rotate(arr, angle):
    return _arr(
        _pil(arr).rotate(
            float(angle),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
    )


def scale_stretch(arr, sx, sy):
    nw = max(8, int(round(IMAGE_SIZE * float(sx))))
    nh = max(8, int(round(IMAGE_SIZE * float(sy))))
    resized = _pil(arr).resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    canvas.paste(
        resized,
        ((IMAGE_SIZE - nw) // 2, (IMAGE_SIZE - nh) // 2),
    )
    return _arr(canvas)


def shear(arr, degrees):
    sh = np.tan(np.deg2rad(float(degrees)))
    cy = (IMAGE_SIZE - 1) / 2.0
    return _arr(
        _pil(arr).transform(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Transform.AFFINE,
            (1.0, -sh, sh * cy, 0.0, 1.0, 0.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
    )


def create_variant(base, group_id, rng):
    arr = np.asarray(base, dtype=np.float32).copy()

    if group_id == 0:
        if rng.random() < 0.65:
            arr = shift(
                arr,
                rng.integers(-1, 2),
                rng.integers(-1, 2),
            )
        if rng.random() < 0.30:
            arr = _arr(
                ImageEnhance.Brightness(_pil(arr)).enhance(
                    float(rng.uniform(0.94, 1.0))
                )
            )
        return np.clip(arr, 0, 1), 0.0

    if group_id == 1:
        delta = float(
            rng.choice(np.asarray([0.125, 0.25, 0.375, 0.50]))
        )
        arr = adjust_stroke_width(arr, -delta)
        if rng.random() < 0.40:
            arr = shift(
                arr,
                rng.integers(-1, 2),
                rng.integers(-1, 2),
            )
        return np.clip(arr, 0, 1), delta

    if group_id == 2:
        arr = rotate(arr, rng.uniform(-6.0, 6.0))
        arr = shift(
            arr,
            rng.integers(-2, 3),
            rng.integers(-2, 3),
        )
        return np.clip(arr, 0, 1), 0.0

    if group_id == 3:
        global_scale = float(rng.uniform(0.95, 1.05))
        arr = scale_stretch(
            arr,
            global_scale * rng.uniform(0.94, 1.06),
            global_scale * rng.uniform(0.94, 1.06),
        )
        return np.clip(arr, 0, 1), 0.0

    thinning = float(
        rng.choice(
            np.asarray([0.0, 0.125, 0.25]),
            p=np.asarray([0.40, 0.35, 0.25]),
        )
    )
    if thinning > 0:
        arr = adjust_stroke_width(arr, -thinning)

    arr = rotate(arr, rng.uniform(-5.0, 5.0))
    arr = shift(
        arr,
        rng.integers(-2, 3),
        rng.integers(-2, 3),
    )
    arr = scale_stretch(
        arr,
        rng.uniform(0.96, 1.04),
        rng.uniform(0.94, 1.06),
    )

    if rng.random() < 0.35:
        arr = shear(arr, rng.uniform(-3.0, 3.0))
    if rng.random() < 0.15:
        arr = _arr(
            _pil(arr).filter(
                ImageFilter.GaussianBlur(
                    radius=float(rng.uniform(0.10, 0.30))
                )
            )
        )
    if rng.random() < 0.25:
        arr = _arr(
            ImageEnhance.Brightness(_pil(arr)).enhance(
                float(rng.uniform(0.92, 1.00))
            )
        )

    return np.clip(arr, 0, 1), thinning


def balanced_plan(y, n, rng):
    per_class = n // NUM_CLASSES
    remainder = n % NUM_CLASSES
    result = []

    for class_id in range(NUM_CLASSES):
        candidates = np.where(y == class_id)[0]
        if len(candidates) == 0:
            raise RuntimeError(
                f"В test отсутствует класс {ROMAJI[class_id]}."
            )
        count = per_class + (1 if class_id < remainder else 0)
        chosen = rng.choice(candidates, size=count, replace=True)
        result.extend(chosen.tolist())

    result = np.asarray(result, dtype=np.int64)
    rng.shuffle(result)
    return result


def metrics_from_probs(probs, y):
    pred = np.argmax(probs, axis=1)
    m = classification_metrics(y, pred, NUM_CLASSES)
    return {
        "accuracy": float(m["accuracy"]),
        "macro_f1": float(m["macro_f1"]),
        "errors": int(np.sum(pred != y)),
    }


def save_bar_plot(rows, metric, ylabel, title, output):
    labels = [r["label"] for r in rows]
    before = [r[f"before_{metric}"] for r in rows]
    after = [r[f"after_{metric}"] for r in rows]

    if metric == "macro_f1":
        before = [v * 100 for v in before]
        after = [v * 100 for v in after]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    ax.bar(
        x - width / 2,
        before,
        width,
        facecolor="white",
        edgecolor="black",
        hatch="///",
        label="До коррекции толщины",
    )
    ax.bar(
        x + width / 2,
        after,
        width,
        facecolor="0.75",
        edgecolor="black",
        hatch="...",
        label="После коррекции толщины",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", color="0.82")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()

    if args.samples < 500 or args.samples % 5 != 0:
        raise ValueError("--samples должен быть >= 500 и делиться на 5.")

    args.results.mkdir(parents=True, exist_ok=True)
    nn, _metadata = load_model(args.model)

    X, y, paths = load_processed_dataset(
        args.custom_data,
        cache_path=args.custom_cache,
        rebuild_cache=args.rebuild_cache,
    )
    _train_idx, test_idx = stratified_train_test_split(
        y,
        train_ratio=args.custom_train_ratio,
        seed=args.seed,
    )

    X_test = X[test_idx]
    y_test = y[test_idx]
    paths_test = [paths[int(i)] for i in test_idx]

    rng = np.random.default_rng(args.seed + 5000)
    per_group = args.samples // 5

    X_before = np.empty((args.samples, IMAGE_SIZE * IMAGE_SIZE), dtype=np.float32)
    X_after = np.empty_like(X_before)
    y_all = np.empty(args.samples, dtype=np.int64)
    group_ids = np.empty(args.samples, dtype=np.int64)
    source_indices = np.empty(args.samples, dtype=np.int64)
    corrections = np.empty(args.samples, dtype=np.float32)

    cursor = 0
    for group_id, (group_key, _group_label) in enumerate(GROUPS):
        plan = balanced_plan(y_test, per_group, rng)

        for source_idx in plan:
            base = X_test[int(source_idx)].reshape(IMAGE_SIZE, IMAGE_SIZE)
            variant, correction = create_variant(base, group_id, rng)
            normalized = (
                adjust_stroke_width(variant, correction)
                if correction > 0
                else variant
            )

            X_before[cursor] = variant.reshape(-1)
            X_after[cursor] = normalized.reshape(-1)
            y_all[cursor] = y_test[int(source_idx)]
            group_ids[cursor] = group_id
            source_indices[cursor] = int(source_idx)
            corrections[cursor] = correction
            cursor += 1

        print(
            f"Группа {group_id + 1}/5: {group_key} — {per_group} изображений"
        )

    probs_before = nn.predict_proba(X_before)
    probs_after = nn.predict_proba(X_after)

    overall_before = metrics_from_probs(probs_before, y_all)
    overall_after = metrics_from_probs(probs_after, y_all)

    evaluate_probabilities_and_save(
        probs_before,
        y_all,
        args.results / "до_нормализации",
        "robustness 5000 — до коррекции толщины",
    )
    evaluate_probabilities_and_save(
        probs_after,
        y_all,
        args.results / "после_нормализации",
        "robustness 5000 — после коррекции толщины",
    )

    rows = []
    for group_id, (key, label) in enumerate(GROUPS):
        mask = group_ids == group_id
        before = metrics_from_probs(probs_before[mask], y_all[mask])
        after = metrics_from_probs(probs_after[mask], y_all[mask])

        row = {
            "group": key,
            "label": label,
            "samples": int(mask.sum()),
            "before_accuracy": before["accuracy"],
            "before_macro_f1": before["macro_f1"],
            "before_errors": before["errors"],
            "after_accuracy": after["accuracy"],
            "after_macro_f1": after["macro_f1"],
            "after_errors": after["errors"],
        }
        rows.append(row)

        # Полный набор ROC/матриц для каждой из пяти групп сильно
        # раздувает время построения графиков и дублирует информацию.
        # Для главной проблемной группы "тонкие штрихи" сохраняем
        # полный набор артефактов до/после; для остальных достаточно
        # общей таблицы и сравнительных графиков.
        if key == "тонкие_штрихи":
            evaluate_probabilities_and_save(
                probs_before[mask],
                y_all[mask],
                args.results / "тонкие_штрихи" / "до",
                f"{label} — до коррекции толщины",
            )
            evaluate_probabilities_and_save(
                probs_after[mask],
                y_all[mask],
                args.results / "тонкие_штрихи" / "после",
                f"{label} — после коррекции толщины",
            )

    with (args.results / "группы_сравнение.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (args.results / "synthetic_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "номер",
            "группа",
            "истинный_класс",
            "символ",
            "исходный_test_индекс",
            "исходный_файл",
            "коррекция_толщины_px",
        ])
        for i in range(args.samples):
            cls = int(y_all[i])
            src = int(source_indices[i])
            writer.writerow([
                i,
                GROUPS[int(group_ids[i])][0],
                ROMAJI[cls],
                KANA[cls],
                src,
                paths_test[src],
                f"{float(corrections[i]):.3f}",
            ])

    save_bar_plot(
        rows,
        "macro_f1",
        "Макро-F1, %",
        "Robustness benchmark: качество по типам искажений",
        args.results / "macro_f1_по_группам.png",
    )
    save_bar_plot(
        rows,
        "errors",
        "Количество ошибок",
        "Robustness benchmark: ошибки по типам искажений",
        args.results / "ошибки_по_группам.png",
    )

    summary = {
        "samples": args.samples,
        "source_closed_test_samples": int(len(y_test)),
        "note": (
            "5000 synthetic variants derived only from the fixed "
            "191-image custom test. This is a robustness stress-test, "
            "not 5000 independent real observations."
        ),
        "overall_before": overall_before,
        "overall_after": overall_after,
        "groups": rows,
        "seed": args.seed,
    }
    (args.results / "robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Robustness benchmark — 5000 изображений",
        "",
        (
            "5000 изображений — синтетические варианты фиксированных "
            "191 реальных custom test-примеров. Это стресс-тест "
            "устойчивости, а не независимая выборка из 5000 наблюдений."
        ),
        "",
        "## Общий результат",
        "",
        "| Вариант | Accuracy | Macro-F1 | Ошибок |",
        "|---|---:|---:|---:|",
        (
            f"| До коррекции | {overall_before['accuracy']*100:.2f}% | "
            f"{overall_before['macro_f1']*100:.2f}% | "
            f"{overall_before['errors']} |"
        ),
        (
            f"| После коррекции | {overall_after['accuracy']*100:.2f}% | "
            f"{overall_after['macro_f1']*100:.2f}% | "
            f"{overall_after['errors']} |"
        ),
        "",
        "## По группам",
        "",
        "| Группа | N | F1 до | F1 после | Ошибок до → после |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        report.append(
            f"| {r['label']} | {r['samples']} | "
            f"{r['before_macro_f1']*100:.2f}% | "
            f"{r['after_macro_f1']*100:.2f}% | "
            f"{r['before_errors']} → {r['after_errors']} |"
        )

    (args.results / "ROBUSTNESS_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n=== ROBUSTNESS 5K ===")
    print(
        f"До коррекции: Accuracy={overall_before['accuracy']:.6f}, "
        f"Macro-F1={overall_before['macro_f1']:.6f}, "
        f"ошибок={overall_before['errors']}"
    )
    print(
        f"После коррекции: Accuracy={overall_after['accuracy']:.6f}, "
        f"Macro-F1={overall_after['macro_f1']:.6f}, "
        f"ошибок={overall_after['errors']}"
    )
    for r in rows:
        print(
            f"{r['label']}: "
            f"F1 {r['before_macro_f1']:.4f} → {r['after_macro_f1']:.4f}, "
            f"ошибки {r['before_errors']} → {r['after_errors']}"
        )


if __name__ == "__main__":
    main()
