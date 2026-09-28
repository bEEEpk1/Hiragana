from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from config import IMAGE_SIZE, KANA, NUM_CLASSES, ROMAJI
from data_utils import (
    augment_training_set,
    augment_training_set_extended,
    build_or_load_augmented_pool,
    build_or_load_domain_balanced_pool,
    build_finetune_pool,
    stratified_subtrain_validation_split,
    load_processed_dataset,
    save_split_manifest,
    stratified_train_val_test_split,
    stratified_train_test_split,
    save_train_test_manifest,
)
from metrics import (
    classification_metrics,
    confusion_matrix,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_roc_curves,
    roc_ovr,
    save_per_class_metrics,
)
from neural_network import NeuralNetwork, load_model, save_model
from evaluation import evaluate_and_save
from metrics import plot_domain_adaptation_curves
from report_visuals import save_augmentation_examples


def parse_args():
    parser = argparse.ArgumentParser(description="Курсовая работа: обучение нейронной сети для распознавания хираганы")
    parser.add_argument("--mode", choices=["train", "train-adapted", "train-wide-pool", "train-domain-balanced", "experiments", "tune-hidden", "tune-l2", "tune-augment"], default="train")
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--cache", type=Path, default=Path("cache/hiragana_32.npz"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--model", type=Path, default=Path("models/best_model_extended.pkl"))
    parser.add_argument("--results", type=Path, default=Path("results_extended"))

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--init", choices=["1/n", "2/n", "xavier", "he"], default="he")
    parser.add_argument("--batch-size", default="1", help="full, 1, 32, 64 ...")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--augment-copies", type=int, default=3)
    parser.add_argument(
        "--augmentation",
        choices=["none", "basic-static", "extended-online"],
        default="extended-online",
        help="Режим аугментации TRAIN.",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--experiment-epochs", type=int, default=35)
    parser.add_argument("--experiment-hidden", type=int, default=64)
    parser.add_argument("--tuning-epochs", type=int, default=35)
    parser.add_argument(
        "--custom-data",
        type=Path,
        default=Path("custom_dataset"),
        help="Пользовательский размеченный датасет с romaji-папками.",
    )
    parser.add_argument(
        "--custom-cache",
        type=Path,
        default=Path("cache/custom_32.npz"),
    )
    parser.add_argument(
        "--custom-train-ratio",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--custom-repeat",
        type=int,
        default=3,
        help="Во сколько раз усилить вес оригиналов custom_train перед on-the-fly augmentation.",
    )
    parser.add_argument(
        "--custom-manifest",
        type=Path,
        default=Path("results_adapted/custom_split_manifest.csv"),
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=100000,
        help="Размер широкого аугментированного обучающего пула.",
    )
    parser.add_argument(
        "--pool-cache",
        type=Path,
        default=Path("cache/augmented_pool_100k.npz"),
        help="Кэш широкого обучающего пула.",
    )
    parser.add_argument(
        "--rebuild-pool",
        action="store_true",
        help="Пересоздать широкий аугментированный пул.",
    )
    parser.add_argument(
        "--custom-validation-ratio",
        type=float,
        default=0.20,
        help="Доля custom_train, выделяемая во внутреннюю пользовательскую валидацию.",
    )
    parser.add_argument(
        "--pool-custom-fraction",
        type=float,
        default=0.50,
        help="Доля пользовательского домена в широком обучающем пуле.",
    )
    parser.add_argument(
        "--selection-custom-weight",
        type=float,
        default=0.50,
        help="Вес пользовательской валидации в критерии выбора лучшей эпохи.",
    )
    parser.add_argument("--lr-patience", type=int, default=12)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=0.00005)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--finetune-learning-rate", type=float, default=0.0001)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument("--finetune-pool-size", type=int, default=20000)
    parser.add_argument("--finetune-custom-fraction", type=float, default=0.70)
    parser.add_argument(
        "--resume-stage1",
        type=Path,
        default=None,
        help="Загрузить checkpoint основного этапа и сразу перейти к fine-tuning.",
    )
    return parser.parse_args()


def parse_batch_size(value: str):
    if str(value).lower() == "full":
        return "full"
    batch = int(value)
    if batch < 1:
        raise ValueError("batch-size должен быть >= 1 или 'full'.")
    return batch


def prepare_data(args):
    X, y, paths = load_processed_dataset(
        args.data,
        cache_path=args.cache,
        rebuild_cache=args.rebuild_cache,
    )
    train_idx, val_idx, test_idx = stratified_train_val_test_split(y, seed=args.seed)
    args.results.mkdir(parents=True, exist_ok=True)
    save_split_manifest(paths, y, train_idx, val_idx, test_idx, args.results / "split_manifest.csv")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Разбиение: обучение={len(X_train)}, валидация={len(X_val)}, тест={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_test_outputs(nn, X_test, y_test, history, metadata, results_dir: Path):
    y_pred = nn.predict(X_test)
    probs = nn.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred, NUM_CLASSES)
    metrics = classification_metrics(y_test, y_pred, NUM_CLASSES)
    curves, macro_auc = roc_ovr(y_test, probs, NUM_CLASSES)
    metrics["macro_roc_auc"] = macro_auc

    results_dir.mkdir(parents=True, exist_ok=True)
    save_per_class_metrics(metrics, ROMAJI, KANA, results_dir / "per_class_metrics.csv")
    plot_learning_curves(history, results_dir / "кривые_обучения.png", best_epoch=best_info["best_epoch"])
    plot_confusion_matrix(cm, ROMAJI, results_dir / "confusion_matrix.png")
    plot_roc_curves(curves, ROMAJI, results_dir / "roc_curves.png")

    with (results_dir / "roc_auc_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "romaji", "hiragana", "auc"])
        for curve in curves:
            i = curve["class_id"]
            writer.writerow([i, ROMAJI[i], KANA[i], f"{curve['auc']:.6f}"])

    text = (
        f"Accuracy:        {metrics['accuracy']:.6f}\n"
        f"Macro Precision: {metrics['macro_precision']:.6f}\n"
        f"Macro Recall:    {metrics['macro_recall']:.6f}\n"
        f"Macro F1:        {metrics['macro_f1']:.6f}\n"
        f"Macro ROC-AUC:   {macro_auc:.6f}\n"
    )
    (results_dir / "metrics.txt").write_text(text, encoding="utf-8")
    print("\n=== TEST METRICS ===")
    print(text)

    metadata["test_metrics"] = {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "macro_roc_auc": macro_auc,
    }
    return metadata


def train_final(args):
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(args)

    epoch_data_factory = None
    X_train_for_fit = X_train
    y_train_for_fit = y_train

    if args.augmentation == "none" or args.augment_copies <= 0:
        print(f"Augmentation: none | train={len(X_train)}")
    elif args.augmentation == "basic-static":
        X_train_for_fit, y_train_for_fit = augment_training_set(
            X_train,
            y_train,
            copies=args.augment_copies,
            seed=args.seed,
        )
        print(
            f"Augmentation: basic-static | "
            f"{len(X_train)} -> {len(X_train_for_fit)}"
        )
    elif args.augmentation == "extended-online":
        expected = len(X_train) * (args.augment_copies + 1)

        def epoch_data_factory(epoch: int):
            # Новый deterministic seed на каждой эпохе:
            # повторный запуск с тем же --seed полностью воспроизводим.
            return augment_training_set_extended(
                X_train,
                y_train,
                copies=args.augment_copies,
                seed=args.seed + epoch * 1009,
            )

        print(
            "Augmentation: extended-online | "
            f"каждая эпоха: {len(X_train)} -> {expected} "
            "(варианты генерируются заново)"
        )

    batch_size = parse_batch_size(args.batch_size)
    layer_sizes = [IMAGE_SIZE * IMAGE_SIZE, args.hidden, NUM_CLASSES]
    nn = NeuralNetwork(layer_sizes, seed=args.seed)
    nn.initialize_weights(args.init)

    print("\n=== FINAL TRAINING ===")
    print(
        f"architecture={layer_sizes}, init={args.init}, batch={batch_size}, "
        f"lr={args.learning_rate}, l2={args.l2}, epochs={args.epochs}, "
        f"augmentation={args.augmentation}, copies={args.augment_copies}"
    )
    if args.patience <= 0:
        print(
            f"Ранняя остановка отключена: будут выполнены все {args.epochs} эпох. "
            "После обучения будут восстановлены веса лучшей эпохи по валидационному макро-F1."
        )

    started = time.time()
    history, best_info = nn.train(
        X_train_for_fit,
        y_train_for_fit,
        X_val,
        y_val,
        batch_size=batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_reg=args.l2,
        patience=args.patience,
        verbose=True,
        epoch_data_factory=epoch_data_factory,
    )
    elapsed = time.time() - started

    metadata = {
        "class_romaji": ROMAJI,
        "class_hiragana": KANA,
        "image_size": IMAGE_SIZE,
        "preprocessing": "grayscale -> invert -> crop -> preserve aspect -> center -> 32x32 -> [0,1]",
        "architecture": layer_sizes,
        "initialization": args.init,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "lambda_l2": args.l2,
        "augmentation_mode": args.augmentation,
        "augmentation_copies": args.augment_copies,
        "augmentation_details": {
            "rotation_degrees": "approximately +/-10",
            "translation_final_px": "approximately +/-3",
            "global_scale": "0.90..1.10",
            "stretch_x_y": "0.88..1.12",
            "shear_degrees": "approximately +/-7",
            "stroke_thickness": "thin / unchanged / thick",
            "mild_blur_probability": 0.25,
            "on_the_fly": args.augmentation == "extended-online",
        },
        "seed": args.seed,
        "best_epoch": best_info["best_epoch"],
        "best_validation_macro_f1": best_info["best_validation_macro_f1"],
        "training_seconds": elapsed,
    }

    # Сохраняем обученную модель ДО построения отчетных метрик и графиков.
    # Если визуализация или экспорт результатов упадут, веса обучения не потеряются.
    save_model(args.model, nn, metadata)

    metadata = save_test_outputs(nn, X_test, y_test, history, metadata, args.results)
    # Перезаписываем файл, добавляя итоговые TEST-метрики в metadata.
    save_model(args.model, nn, metadata)
    (args.results / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Модель сохранена: {args.model.resolve()}")
    print(f"Лучшая эпоха: {best_info['best_epoch']}")
    print(f"Лучший валидационный макро-F1: {best_info['best_validation_macro_f1']:.4f}")
    print(f"Время обучения: {elapsed:.1f} сек.")



def train_adapted(args):
    """Обучение модели с адаптацией к пользовательскому почерку.

    TRAIN:
      Kaggle train + custom_train (60% каждого custom-класса).
    VALIDATION:
      исходный Kaggle validation.
    TEST:
      1) исходный Kaggle test;
      2) закрытый custom_test (40% каждого custom-класса).

    Custom test никогда не участвует в backpropagation или early stopping.
    """
    # 1. Исходный Kaggle split остается тем же.
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(args)

    # 2. Пользовательский датасет.
    X_custom, y_custom, custom_paths = load_processed_dataset(
        args.custom_data,
        cache_path=args.custom_cache,
        rebuild_cache=args.rebuild_cache,
    )
    custom_train_idx, custom_test_idx = stratified_train_test_split(
        y_custom,
        train_ratio=args.custom_train_ratio,
        seed=args.seed,
    )
    save_train_test_manifest(
        args.custom_data,
        custom_paths,
        y_custom,
        custom_train_idx,
        custom_test_idx,
        args.custom_manifest,
    )

    X_custom_train = X_custom[custom_train_idx]
    y_custom_train = y_custom[custom_train_idx]
    X_custom_test = X_custom[custom_test_idx]
    y_custom_test = y_custom[custom_test_idx]

    print(
        f"Пользовательский набор: обучение={len(X_custom_train)}, "
        f"тест={len(X_custom_test)} "
        f"(доля обучения={args.custom_train_ratio:.2f})"
    )

    # 3. Для history считаем train на уникальных оригиналах.
    X_train_eval = np.concatenate([X_train, X_custom_train], axis=0)
    y_train_eval = np.concatenate([y_train, y_custom_train], axis=0)

    # 4. Custom train усиливаем одинаковым repeat по всем его объектам.
    repeat = max(1, int(args.custom_repeat))
    X_custom_weighted = np.repeat(X_custom_train, repeat, axis=0)
    y_custom_weighted = np.repeat(y_custom_train, repeat, axis=0)

    X_weighted_base = np.concatenate([X_train, X_custom_weighted], axis=0)
    y_weighted_base = np.concatenate([y_train, y_custom_weighted], axis=0)

    expected_epoch = len(X_weighted_base) * (args.augment_copies + 1)
    print(
        "Исходная обучающая база: "
        f"основной набор={len(X_train)}, "
        f"уникальные пользовательские примеры={len(X_custom_train)}, "
        f"коэффициент усиления={repeat} -> "
        f"объектов до аугментации={len(X_weighted_base)}"
    )
    print(
        f"Extended on-the-fly augmentation: "
        f"каждая эпоха примерно {len(X_weighted_base)} -> {expected_epoch}"
    )

    def epoch_data_factory(epoch: int):
        return augment_training_set_extended(
            X_weighted_base,
            y_weighted_base,
            copies=args.augment_copies,
            seed=args.seed + epoch * 1009,
        )

    batch_size = parse_batch_size(args.batch_size)
    layer_sizes = [IMAGE_SIZE * IMAGE_SIZE, args.hidden, NUM_CLASSES]
    nn = NeuralNetwork(layer_sizes, seed=args.seed)
    nn.initialize_weights(args.init)

    print("\n=== ОБУЧЕНИЕ АДАПТИРОВАННОЙ МОДЕЛИ ===")
    print(
        f"архитектура={layer_sizes}, инициализация={args.init}, "
        f"размер_пакета={batch_size}, скорость_обучения={args.learning_rate}, "
        f"L2={args.l2}, эпох={args.epochs}, "
        f"усиление_пользовательских_примеров={repeat}, "
        f"аугментированных_копий={args.augment_copies}"
    )
    if args.patience <= 0:
        print(
            f"Ранняя остановка отключена: будут выполнены все {args.epochs} эпох. "
            "После обучения будут восстановлены веса лучшей эпохи по валидационному макро-F1."
        )

    started = time.time()
    history, best_info = nn.train(
        X_train_eval,
        y_train_eval,
        X_val,
        y_val,
        batch_size=batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_reg=args.l2,
        patience=args.patience,
        verbose=True,
        epoch_data_factory=epoch_data_factory,
    )
    elapsed = time.time() - started

    model_path = args.model
    results_dir = args.results

    # В adapted-режиме безопасные дефолты, чтобы не перезаписать extended model.
    if str(model_path) == "models/best_model_extended.pkl":
        model_path = Path("models/best_model_adapted.pkl")
    if str(results_dir) == "results_extended":
        results_dir = Path("results_adapted")

    metadata = {
        "class_romaji": ROMAJI,
        "class_hiragana": KANA,
        "image_size": IMAGE_SIZE,
        "architecture": layer_sizes,
        "initialization": args.init,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "lambda_l2": args.l2,
        "training_mode": "kaggle_plus_custom_adaptation",
        "kaggle_train_samples": len(X_train),
        "custom_train_unique_samples": len(X_custom_train),
        "custom_test_samples": len(X_custom_test),
        "custom_train_ratio": args.custom_train_ratio,
        "custom_repeat": repeat,
        "augmentation_mode": "extended-online",
        "augmentation_copies": args.augment_copies,
        "validation_source": "Kaggle validation only",
        "seed": args.seed,
        "best_epoch": best_info["best_epoch"],
        "best_validation_macro_f1": best_info["best_validation_macro_f1"],
        "training_seconds": elapsed,
    }

    # Сохраняем модель до отчетной визуализации.
    save_model(model_path, nn, metadata)

    # Learning curves едины для процесса обучения.
    from metrics import plot_learning_curves, plot_domain_adaptation_curves
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(
        history,
        results_dir / "кривые_обучения.png",
        best_epoch=best_info["best_epoch"],
    )
    save_augmentation_examples(
        X_custom_train,
        y_custom_train,
        results_dir / "примеры_аугментации.png",
        seed=args.seed + 50000,
        rows=4,
        variants=6,
    )

    print("\n=== ТЕСТ НА ИСХОДНОМ НАБОРЕ ===")
    kaggle_metrics = evaluate_and_save(
        nn,
        X_test,
        y_test,
        results_dir / "исходный_тест",
        "исходная тестовая выборка",
    )
    print((results_dir / "исходный_тест" / "метрики.txt").read_text(encoding="utf-8"))

    print("\n=== ТЕСТ НА ПОЛЬЗОВАТЕЛЬСКОМ ПОЧЕРКЕ ===")
    custom_metrics = evaluate_and_save(
        nn,
        X_custom_test,
        y_custom_test,
        results_dir / "пользовательский_тест",
        "тестовая выборка пользовательского почерка",
    )
    print((results_dir / "пользовательский_тест" / "метрики.txt").read_text(encoding="utf-8"))

    metadata["kaggle_test_metrics"] = {
        k: v for k, v in kaggle_metrics.items() if k != "per_class"
    }
    metadata["custom_test_metrics"] = {
        k: v for k, v in custom_metrics.items() if k != "per_class"
    }

    save_model(model_path, nn, metadata)
    (results_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Модель сохранена: {model_path.resolve()}")
    print(f"Файл фиксированного разбиения пользовательского набора: {args.custom_manifest.resolve()}")
    print(f"Лучшая эпоха: {best_info['best_epoch']}")
    print(f"Лучший валидационный макро-F1: {best_info['best_validation_macro_f1']:.4f}")
    print(f"Время обучения: {elapsed:.1f} сек.")


def train_wide_pool(args):
    """Обучение на широком кэшированном пуле примерно из 100 тыс. изображений."""
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(args)

    X_custom, y_custom, custom_paths = load_processed_dataset(
        args.custom_data,
        cache_path=args.custom_cache,
        rebuild_cache=args.rebuild_cache,
    )
    custom_train_idx, custom_test_idx = stratified_train_test_split(
        y_custom,
        train_ratio=args.custom_train_ratio,
        seed=args.seed,
    )
    save_train_test_manifest(
        args.custom_data,
        custom_paths,
        y_custom,
        custom_train_idx,
        custom_test_idx,
        args.custom_manifest,
    )

    X_custom_train = X_custom[custom_train_idx]
    y_custom_train = y_custom[custom_train_idx]
    X_custom_test = X_custom[custom_test_idx]
    y_custom_test = y_custom[custom_test_idx]

    print(
        f"Пользовательский набор: обучение={len(X_custom_train)}, "
        f"тест={len(X_custom_test)} "
        f"(доля обучения={args.custom_train_ratio:.2f})"
    )

    # Только для расчета learning curves.
    X_train_eval = np.concatenate(
        [X_train, X_custom_train],
        axis=0,
    )
    y_train_eval = np.concatenate(
        [y_train, y_custom_train],
        axis=0,
    )

    repeat = max(1, int(args.custom_repeat))
    X_custom_weighted = np.repeat(
        X_custom_train,
        repeat,
        axis=0,
    )
    y_custom_weighted = np.repeat(
        y_custom_train,
        repeat,
        axis=0,
    )

    X_weighted_base = np.concatenate(
        [X_train, X_custom_weighted],
        axis=0,
    )
    y_weighted_base = np.concatenate(
        [y_train, y_custom_weighted],
        axis=0,
    )

    print(
        "База для широкого пула: "
        f"основной набор={len(X_train)}, "
        f"уникальные пользовательские примеры={len(X_custom_train)}, "
        f"коэффициент усиления={repeat}, "
        f"взвешенная база={len(X_weighted_base)}"
    )

    X_pool, y_pool = build_or_load_augmented_pool(
        X_weighted_base,
        y_weighted_base,
        pool_size=args.pool_size,
        cache_path=args.pool_cache,
        seed=args.seed + 60000,
        rebuild=args.rebuild_pool,
    )

    batch_size = parse_batch_size(args.batch_size)
    if batch_size == "full":
        raise ValueError(
            "Для режима train-wide-pool нужен числовой mini-batch."
        )

    layer_sizes = [
        IMAGE_SIZE * IMAGE_SIZE,
        args.hidden,
        NUM_CLASSES,
    ]
    nn = NeuralNetwork(
        layer_sizes,
        seed=args.seed,
    )
    nn.initialize_weights(args.init)

    print(
        "\n=== ОБУЧЕНИЕ НА ШИРОКОМ "
        "АУГМЕНТИРОВАННОМ ПУЛЕ ==="
    )
    print(
        f"архитектура={layer_sizes}, "
        f"инициализация={args.init}, "
        f"размер_пула={len(y_pool)}, "
        f"размер_пакета={batch_size}, "
        f"скорость_обучения={args.learning_rate}, "
        f"L2={args.l2}, эпох={args.epochs}"
    )
    print(
        f"Шагов обновления за эпоху: "
        f"{int(np.ceil(len(y_pool) / batch_size))}"
    )
    if args.patience <= 0:
        print(
            "Ранняя остановка отключена. После обучения "
            "будут восстановлены веса лучшей эпохи."
        )

    started = time.time()
    history, best_info = nn.train_uint8_pool(
        X_pool_uint8=X_pool,
        y_pool=y_pool,
        X_train_eval=X_train_eval,
        y_train_eval=y_train_eval,
        X_val=X_val,
        y_val=y_val,
        batch_size=int(batch_size),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_reg=args.l2,
        patience=args.patience,
        verbose=True,
    )
    elapsed = time.time() - started

    model_path = args.model
    results_dir = args.results
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "class_romaji": ROMAJI,
        "class_hiragana": KANA,
        "image_size": IMAGE_SIZE,
        "architecture": layer_sizes,
        "initialization": args.init,
        "batch_size": int(batch_size),
        "learning_rate": args.learning_rate,
        "lambda_l2": args.l2,
        "training_mode": "wide_augmented_pool",
        "pool_size": int(len(y_pool)),
        "pool_storage_dtype": "uint8",
        "pool_cache": str(args.pool_cache),
        "weighted_base_samples": int(
            len(X_weighted_base)
        ),
        "kaggle_train_samples": int(
            len(X_train)
        ),
        "custom_train_unique_samples": int(
            len(X_custom_train)
        ),
        "custom_test_samples": int(
            len(X_custom_test)
        ),
        "custom_train_ratio": args.custom_train_ratio,
        "custom_repeat": repeat,
        "validation_source": "Kaggle validation only",
        "seed": args.seed,
        "best_epoch": best_info["best_epoch"],
        "best_validation_macro_f1": best_info[
            "best_validation_macro_f1"
        ],
        "training_seconds": elapsed,
    }

    save_model(
        model_path,
        nn,
        metadata,
    )

    plot_learning_curves(
        history,
        results_dir / "кривые_обучения.png",
        best_epoch=best_info["best_epoch"],
    )
    save_augmentation_examples(
        X_custom_train,
        y_custom_train,
        results_dir / "примеры_аугментации.png",
        seed=args.seed + 70000,
        rows=4,
        variants=6,
    )

    print("\n=== ТЕСТ НА ИСХОДНОМ НАБОРЕ ===")
    kaggle_metrics = evaluate_and_save(
        nn,
        X_test,
        y_test,
        results_dir / "исходный_тест",
        "исходная тестовая выборка",
    )
    print(
        (
            results_dir
            / "исходный_тест"
            / "метрики.txt"
        ).read_text(encoding="utf-8")
    )

    print(
        "\n=== ТЕСТ НА ПОЛЬЗОВАТЕЛЬСКОМ "
        "ПОЧЕРКЕ ==="
    )
    custom_metrics = evaluate_and_save(
        nn,
        X_custom_test,
        y_custom_test,
        results_dir / "пользовательский_тест",
        "тестовая выборка пользовательского почерка",
    )
    print(
        (
            results_dir
            / "пользовательский_тест"
            / "метрики.txt"
        ).read_text(encoding="utf-8")
    )

    metadata["kaggle_test_metrics"] = {
        k: v
        for k, v in kaggle_metrics.items()
        if k != "per_class"
    }
    metadata["custom_test_metrics"] = {
        k: v
        for k, v in custom_metrics.items()
        if k != "per_class"
    }

    save_model(
        model_path,
        nn,
        metadata,
    )
    (
        results_dir / "model_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Модель сохранена: "
        f"{model_path.resolve()}"
    )
    print(
        f"Лучшая эпоха: "
        f"{best_info['best_epoch']} | "
        f"валидационный макро-F1="
        f"{best_info['best_validation_macro_f1']:.6f}"
    )
    print(
        f"Время обучения без построения пула: "
        f"{elapsed:.1f} сек."
    )


def train_domain_balanced(args):
    """Финальный эксперимент: domain-balanced pool + scheduler + fine-tuning."""
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(args)

    X_custom, y_custom, custom_paths = load_processed_dataset(
        args.custom_data,
        cache_path=args.custom_cache,
        rebuild_cache=args.rebuild_cache,
    )

    # Внешний custom test остается тем же, что и в предыдущих экспериментах.
    custom_train_idx, custom_test_idx = stratified_train_test_split(
        y_custom,
        train_ratio=args.custom_train_ratio,
        seed=args.seed,
    )

    X_custom_outer_train = X_custom[custom_train_idx]
    y_custom_outer_train = y_custom[custom_train_idx]
    X_custom_test = X_custom[custom_test_idx]
    y_custom_test = y_custom[custom_test_idx]

    # Внутри 288 custom train выделяем отдельную validation-часть.
    custom_subtrain_idx, custom_val_idx = stratified_subtrain_validation_split(
        y_custom_outer_train,
        validation_ratio=args.custom_validation_ratio,
        seed=args.seed + 101,
    )

    X_custom_subtrain = X_custom_outer_train[custom_subtrain_idx]
    y_custom_subtrain = y_custom_outer_train[custom_subtrain_idx]
    X_custom_val = X_custom_outer_train[custom_val_idx]
    y_custom_val = y_custom_outer_train[custom_val_idx]

    print(
        f"Пользовательский набор: "
        f"адаптационное обучение={len(X_custom_subtrain)}, "
        f"внутренняя валидация={len(X_custom_val)}, "
        f"закрытый тест={len(X_custom_test)}"
    )

    # Широкий 50/50 пул строится только из Kaggle train и custom subtrain.
    X_pool, y_pool, domain_pool = build_or_load_domain_balanced_pool(
        X_kaggle=X_train,
        y_kaggle=y_train,
        X_custom=X_custom_subtrain,
        y_custom=y_custom_subtrain,
        pool_size=args.pool_size,
        custom_fraction=args.pool_custom_fraction,
        cache_path=args.pool_cache,
        seed=args.seed + 80000 + args.hidden,
        rebuild=args.rebuild_pool,
    )

    print(
        "Фактический состав пула: "
        f"исходный домен={(domain_pool == 0).sum()}, "
        f"пользовательский домен={(domain_pool == 1).sum()}"
    )

    # Для train curves используем уникальные исходные train-объекты.
    X_train_eval = np.concatenate(
        [X_train, X_custom_subtrain],
        axis=0,
    )
    y_train_eval = np.concatenate(
        [y_train, y_custom_subtrain],
        axis=0,
    )

    layer_sizes = [
        IMAGE_SIZE * IMAGE_SIZE,
        args.hidden,
        NUM_CLASSES,
    ]
    batch_size = parse_batch_size(args.batch_size)
    if batch_size == "full":
        raise ValueError("Нужен числовой mini-batch.")

    if args.resume_stage1 is not None:
        nn, loaded_metadata = load_model(args.resume_stage1)
        print(
            f"Основной этап загружен из checkpoint: "
            f"{args.resume_stage1.resolve()}"
        )
        history = None
        best_info = {
            "best_epoch": int(
                loaded_metadata.get("best_epoch", 0)
            ),
            "best_selection_score": float(
                loaded_metadata.get("best_selection_score", 0.0)
            ),
            "best_kaggle_validation_macro_f1": float(
                loaded_metadata.get(
                    "best_kaggle_validation_macro_f1", 0.0
                )
            ),
            "best_custom_validation_macro_f1": float(
                loaded_metadata.get(
                    "best_custom_validation_macro_f1", 0.0
                )
            ),
        }
        stage1_seconds = float(
            loaded_metadata.get("seconds", 0.0)
        )
    else:
        nn = NeuralNetwork(
            layer_sizes,
            seed=args.seed,
        )
        nn.initialize_weights(args.init)

        print("\n=== СБАЛАНСИРОВАННОЕ DOMAIN-ADAPTATION ОБУЧЕНИЕ ===")
        print(
            f"архитектура={layer_sizes}, "
            f"размер_пула={len(y_pool)}, "
            f"доля пользовательского домена={args.pool_custom_fraction:.2f}, "
            f"размер_пакета={batch_size}, "
            f"начальный LR={args.learning_rate}, "
            f"L2={args.l2}, "
            f"максимум эпох={args.epochs}"
        )

        started = time.time()
        history, best_info = nn.train_domain_adaptation_pool(
            X_pool_uint8=X_pool,
            y_pool=y_pool,
            X_train_eval=X_train_eval,
            y_train_eval=y_train_eval,
            X_kaggle_val=X_val,
            y_kaggle_val=y_val,
            X_custom_val=X_custom_val,
            y_custom_val=y_custom_val,
            batch_size=int(batch_size),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lambda_reg=args.l2,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            min_learning_rate=args.min_learning_rate,
            early_stopping_patience=args.patience,
            selection_custom_weight=args.selection_custom_weight,
            verbose=True,
        )
        stage1_seconds = time.time() - started

    results_dir = args.results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint сохраняется до построения графиков и fine-tuning.
    stage1_checkpoint = (
        results_dir / "checkpoint_после_основного_этапа.pkl"
    )

    if args.resume_stage1 is None:
        stage1_metadata = {
            "architecture": layer_sizes,
            "training_mode": "domain_balanced_stage1_checkpoint",
            "pool_size": args.pool_size,
            "pool_custom_fraction": args.pool_custom_fraction,
            "custom_subtrain_samples": len(X_custom_subtrain),
            "custom_validation_samples": len(X_custom_val),
            "custom_test_samples": len(X_custom_test),
            "best_epoch": best_info["best_epoch"],
            "best_selection_score": best_info[
                "best_selection_score"
            ],
            "best_kaggle_validation_macro_f1": best_info[
                "best_kaggle_validation_macro_f1"
            ],
            "best_custom_validation_macro_f1": best_info[
                "best_custom_validation_macro_f1"
            ],
            "seconds": stage1_seconds,
            "seed": args.seed,
        }
        save_model(
            stage1_checkpoint,
            nn,
            stage1_metadata,
        )
        print(
            f"Checkpoint основного этапа сохранён: "
            f"{stage1_checkpoint.resolve()}"
        )

    if history is not None:
        plot_domain_adaptation_curves(
            history,
            results_dir / "кривые_domain_adaptation.png",
            best_epoch=best_info["best_epoch"],
        )

    # Оценка checkpoint до fine-tuning.
    pre_ft_kaggle = evaluate_and_save(
        nn,
        X_test,
        y_test,
        results_dir / "до_finetuning" / "исходный_тест",
        "исходная тестовая выборка до финальной адаптации",
    )
    pre_ft_custom = evaluate_and_save(
        nn,
        X_custom_test,
        y_custom_test,
        results_dir / "до_finetuning" / "пользовательский_тест",
        "пользовательский тест до финальной адаптации",
    )

    # ---------------- fine-tuning 70/30 ----------------
    print("\n=== ФИНАЛЬНАЯ АДАПТАЦИЯ ===")
    print(
        f"пул={args.finetune_pool_size}, "
        f"custom={args.finetune_custom_fraction:.2f}, "
        f"batch={args.finetune_batch_size}, "
        f"LR={args.finetune_learning_rate}, "
        f"эпох={args.finetune_epochs}"
    )

    X_ft_pool, y_ft_pool = build_finetune_pool(
        X_kaggle=X_train,
        y_kaggle=y_train,
        X_custom=X_custom_subtrain,
        y_custom=y_custom_subtrain,
        pool_size=args.finetune_pool_size,
        custom_fraction=args.finetune_custom_fraction,
        seed=args.seed + 90000 + args.hidden,
    )

    ft_started = time.time()
    ft_history, ft_best = nn.train_domain_adaptation_pool(
        X_pool_uint8=X_ft_pool,
        y_pool=y_ft_pool,
        X_train_eval=X_train_eval,
        y_train_eval=y_train_eval,
        X_kaggle_val=X_val,
        y_kaggle_val=y_val,
        X_custom_val=X_custom_val,
        y_custom_val=y_custom_val,
        batch_size=args.finetune_batch_size,
        epochs=args.finetune_epochs,
        learning_rate=args.finetune_learning_rate,
        lambda_reg=args.l2,
        lr_patience=8,
        lr_factor=0.5,
        min_learning_rate=min(
            args.min_learning_rate,
            args.finetune_learning_rate / 4,
        ),
        early_stopping_patience=0,
        selection_custom_weight=args.selection_custom_weight,
        verbose=True,
    )
    ft_seconds = time.time() - ft_started

    plot_domain_adaptation_curves(
        ft_history,
        results_dir / "кривые_finetuning.png",
        best_epoch=ft_best["best_epoch"],
    )

    save_augmentation_examples(
        X_custom_subtrain,
        y_custom_subtrain,
        results_dir / "примеры_аугментации.png",
        seed=args.seed + 91000,
        rows=4,
        variants=6,
    )

    print("\n=== ФИНАЛЬНЫЙ ТЕСТ НА ИСХОДНОМ НАБОРЕ ===")
    final_kaggle = evaluate_and_save(
        nn,
        X_test,
        y_test,
        results_dir / "финальная_модель" / "исходный_тест",
        "исходная тестовая выборка",
    )
    print(
        (
            results_dir
            / "финальная_модель"
            / "исходный_тест"
            / "метрики.txt"
        ).read_text(encoding="utf-8")
    )

    print("\n=== ФИНАЛЬНЫЙ ТЕСТ НА ПОЛЬЗОВАТЕЛЬСКОМ ПОЧЕРКЕ ===")
    final_custom = evaluate_and_save(
        nn,
        X_custom_test,
        y_custom_test,
        results_dir / "финальная_модель" / "пользовательский_тест",
        "закрытая тестовая выборка пользовательского почерка",
    )
    print(
        (
            results_dir
            / "финальная_модель"
            / "пользовательский_тест"
            / "метрики.txt"
        ).read_text(encoding="utf-8")
    )

    metadata = {
        "architecture": layer_sizes,
        "training_mode": "domain_balanced_plus_finetuning",
        "pool_size": args.pool_size,
        "pool_custom_fraction": args.pool_custom_fraction,
        "custom_outer_train_samples": len(X_custom_outer_train),
        "custom_subtrain_samples": len(X_custom_subtrain),
        "custom_validation_samples": len(X_custom_val),
        "custom_test_samples": len(X_custom_test),
        "selection_custom_weight": args.selection_custom_weight,
        "stage1": {
            "best_epoch": best_info["best_epoch"],
            "best_selection_score": best_info["best_selection_score"],
            "best_kaggle_validation_macro_f1": best_info[
                "best_kaggle_validation_macro_f1"
            ],
            "best_custom_validation_macro_f1": best_info[
                "best_custom_validation_macro_f1"
            ],
            "seconds": stage1_seconds,
        },
        "finetuning": {
            "epochs": args.finetune_epochs,
            "pool_size": args.finetune_pool_size,
            "custom_fraction": args.finetune_custom_fraction,
            "learning_rate": args.finetune_learning_rate,
            "best_epoch": ft_best["best_epoch"],
            "best_selection_score": ft_best["best_selection_score"],
            "seconds": ft_seconds,
        },
        "before_finetuning": {
            "kaggle_macro_f1": pre_ft_kaggle["macro_f1"],
            "custom_macro_f1": pre_ft_custom["macro_f1"],
        },
        "final_test": {
            "kaggle_accuracy": final_kaggle["accuracy"],
            "kaggle_macro_f1": final_kaggle["macro_f1"],
            "custom_accuracy": final_custom["accuracy"],
            "custom_macro_f1": final_custom["macro_f1"],
            "custom_macro_roc_auc": final_custom["macro_roc_auc"],
        },
        "seed": args.seed,
    }

    save_model(
        args.model,
        nn,
        metadata,
    )
    (
        results_dir / "model_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Модель сохранена: {args.model.resolve()}")
    print(
        f"Основной этап: лучшая эпоха={best_info['best_epoch']}, "
        f"custom validation F1={best_info['best_custom_validation_macro_f1']:.4f}"
    )
    print(
        f"Fine-tuning: лучшая эпоха={ft_best['best_epoch']}, "
        f"комбинированный критерий={ft_best['best_selection_score']:.4f}"
    )

def run_experiments(args):
    """Повторяет основной эксперимент лабораторной №9: 4 инициализации × 3 режима GD.

    Важно: сравнение идет ТОЛЬКО по validation, test здесь не используется.
    """
    X_train, y_train, X_val, y_val, _X_test, _y_test = prepare_data(args)

    init_methods = ["1/n", "2/n", "xavier", "he"]
    batch_types = {"batch": "full", "mini-batch": 32, "stochastic": 1}
    lr_map = {"full": 0.1, 32: 0.01, 1: 0.001}
    layer_sizes = [IMAGE_SIZE * IMAGE_SIZE, args.experiment_hidden, NUM_CLASSES]

    rows = []
    best = None
    print("\n=== EXPERIMENTS: 4 initialization × 3 gradient modes ===")
    print("TEST НЕ ИСПОЛЬЗУЕТСЯ. Критерий выбора — Validation Macro F1.")

    for init in init_methods:
        for batch_name, batch_size in batch_types.items():
            lr = lr_map[batch_size]
            print(f"\n--- {init} + {batch_name} | lr={lr} ---")
            nn = NeuralNetwork(layer_sizes, seed=args.seed)
            nn.initialize_weights(init)
            started = time.time()
            history, best_info = nn.train(
                X_train,
                y_train,
                X_val,
                y_val,
                batch_size=batch_size,
                epochs=args.experiment_epochs,
                learning_rate=lr,
                lambda_reg=args.l2,
                patience=max(5, min(args.patience, 8)),
                verbose=True,
            )
            seconds = time.time() - started
            row = {
                "initialization": init,
                "gradient_mode": batch_name,
                "batch_size": batch_size,
                "learning_rate": lr,
                "hidden_size": args.experiment_hidden,
                "lambda_l2": args.l2,
                "best_epoch": best_info["best_epoch"],
                "validation_macro_f1": best_info["best_validation_macro_f1"],
                "seconds": seconds,
            }
            rows.append(row)
            if best is None or row["validation_macro_f1"] > best["validation_macro_f1"]:
                best = row

    output = args.results / "experiments.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (args.results / "best_experiment.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== BEST VALIDATION CONFIG ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Таблица экспериментов: {output.resolve()}")
    print("После выбора параметров запусти --mode train и передай нужные --init/--batch-size/--learning-rate.")



def _validation_result(nn, X_val, y_val):
    y_pred = nn.predict(X_val)
    metrics = classification_metrics(y_val, y_pred, NUM_CLASSES)
    return {
        "validation_accuracy": metrics["accuracy"],
        "validation_macro_precision": metrics["macro_precision"],
        "validation_macro_recall": metrics["macro_recall"],
        "validation_macro_f1": metrics["macro_f1"],
    }


def _save_sweep(rows, output: Path, best_output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["validation_macro_f1"])
    best_output.write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== BEST VALIDATION CONFIG ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Таблица: {output.resolve()}")
    return best


def tune_hidden(args):
    """Подбор размера скрытого слоя. TEST не используется."""
    X_train, y_train, X_val, y_val, _X_test, _y_test = prepare_data(args)

    hidden_values = [25, 64, 128, 256]
    rows = []

    print("\n=== TUNE HIDDEN SIZE ===")
    print("Фиксировано: init=2/n, stochastic GD, lr=0.001, L2=0.001, augmentation=0")
    print("TEST НЕ ИСПОЛЬЗУЕТСЯ.")

    for hidden in hidden_values:
        layer_sizes = [IMAGE_SIZE * IMAGE_SIZE, hidden, NUM_CLASSES]
        nn = NeuralNetwork(layer_sizes, seed=args.seed)
        nn.initialize_weights("2/n")

        print(f"\n--- hidden={hidden} ---")
        started = time.time()
        history, best_info = nn.train(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=1,
            epochs=args.tuning_epochs,
            learning_rate=0.001,
            lambda_reg=0.001,
            patience=max(5, min(args.patience, 8)),
            verbose=True,
        )
        seconds = time.time() - started
        validation = _validation_result(nn, X_val, y_val)

        rows.append({
            "hidden_size": hidden,
            "initialization": "2/n",
            "gradient_mode": "stochastic",
            "batch_size": 1,
            "learning_rate": 0.001,
            "lambda_l2": 0.001,
            "augmentation_copies": 0,
            "best_epoch": best_info["best_epoch"],
            **validation,
            "seconds": seconds,
        })

    _save_sweep(
        rows,
        args.results / "tune_hidden.csv",
        args.results / "best_hidden.json",
    )


def tune_l2(args):
    """Подбор коэффициента L2 для уже выбранного hidden. TEST не используется."""
    X_train, y_train, X_val, y_val, _X_test, _y_test = prepare_data(args)

    l2_values = [0.0, 0.0001, 0.001, 0.01]
    rows = []

    print("\n=== TUNE L2 ===")
    print(
        f"Фиксировано: hidden={args.hidden}, init=2/n, stochastic GD, "
        "lr=0.001, augmentation=0"
    )
    print("TEST НЕ ИСПОЛЬЗУЕТСЯ.")

    for l2 in l2_values:
        layer_sizes = [IMAGE_SIZE * IMAGE_SIZE, args.hidden, NUM_CLASSES]
        nn = NeuralNetwork(layer_sizes, seed=args.seed)
        nn.initialize_weights("2/n")

        print(f"\n--- L2={l2} ---")
        started = time.time()
        history, best_info = nn.train(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=1,
            epochs=args.tuning_epochs,
            learning_rate=0.001,
            lambda_reg=l2,
            patience=max(5, min(args.patience, 8)),
            verbose=True,
        )
        seconds = time.time() - started
        validation = _validation_result(nn, X_val, y_val)

        rows.append({
            "hidden_size": args.hidden,
            "initialization": "2/n",
            "gradient_mode": "stochastic",
            "batch_size": 1,
            "learning_rate": 0.001,
            "lambda_l2": l2,
            "augmentation_copies": 0,
            "best_epoch": best_info["best_epoch"],
            **validation,
            "seconds": seconds,
        })

    _save_sweep(
        rows,
        args.results / "tune_l2.csv",
        args.results / "best_l2.json",
    )


def tune_augment(args):
    """Подбор числа аугментированных копий TRAIN. TEST не используется."""
    X_train, y_train, X_val, y_val, _X_test, _y_test = prepare_data(args)

    copies_values = [0, 1, 2]
    rows = []

    print("\n=== TUNE AUGMENTATION ===")
    print(
        f"Фиксировано: hidden={args.hidden}, init=2/n, stochastic GD, "
        f"lr=0.001, L2={args.l2}"
    )
    print("TEST НЕ ИСПОЛЬЗУЕТСЯ.")

    for copies in copies_values:
        X_train_aug, y_train_aug = augment_training_set(
            X_train, y_train, copies=copies, seed=args.seed
        )
        layer_sizes = [IMAGE_SIZE * IMAGE_SIZE, args.hidden, NUM_CLASSES]
        nn = NeuralNetwork(layer_sizes, seed=args.seed)
        nn.initialize_weights("2/n")

        print(
            f"\n--- augmentation copies={copies} | "
            f"train={len(X_train)} -> {len(X_train_aug)} ---"
        )
        started = time.time()
        history, best_info = nn.train(
            X_train_aug,
            y_train_aug,
            X_val,
            y_val,
            batch_size=1,
            epochs=args.tuning_epochs,
            learning_rate=0.001,
            lambda_reg=args.l2,
            patience=max(5, min(args.patience, 8)),
            verbose=True,
        )
        seconds = time.time() - started
        validation = _validation_result(nn, X_val, y_val)

        rows.append({
            "hidden_size": args.hidden,
            "initialization": "2/n",
            "gradient_mode": "stochastic",
            "batch_size": 1,
            "learning_rate": 0.001,
            "lambda_l2": args.l2,
            "augmentation_copies": copies,
            "train_samples_after_augmentation": len(X_train_aug),
            "best_epoch": best_info["best_epoch"],
            **validation,
            "seconds": seconds,
        })

    _save_sweep(
        rows,
        args.results / "tune_augment.csv",
        args.results / "best_augment.json",
    )


def main():
    args = parse_args()
    if args.mode == "train-domain-balanced":
        train_domain_balanced(args)
    elif args.mode == "train-wide-pool":
        train_wide_pool(args)
    elif args.mode == "train-adapted":
        train_adapted(args)
    elif args.mode == "experiments":
        run_experiments(args)
    elif args.mode == "tune-hidden":
        tune_hidden(args)
    elif args.mode == "tune-l2":
        tune_l2(args)
    elif args.mode == "tune-augment":
        tune_augment(args)
    else:
        train_final(args)


if __name__ == "__main__":
    main()
