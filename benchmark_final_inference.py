from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import NUM_CLASSES
from data_utils import load_processed_dataset, stratified_train_test_split
from inference_utils import adjust_batch_stroke_width, live_brush_correction_px
from metrics import classification_metrics
from neural_network import load_model


def parse_args():
    p = argparse.ArgumentParser(
        description='Финальный benchmark нормализации толщины штриха.'
    )
    p.add_argument('--custom-data', type=Path, default=Path('custom_dataset'))
    p.add_argument('--custom-cache', type=Path, default=Path('cache/custom_32.npz'))
    p.add_argument('--model', type=Path, default=Path('models/best_model_final.pkl'))
    p.add_argument('--results', type=Path, default=Path('results_final_benchmark'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--custom-train-ratio', type=float, default=0.60)
    p.add_argument('--rebuild-cache', action='store_true')
    return p.parse_args()


def evaluate(nn, X, y):
    probs = nn.predict_proba(X)
    pred = np.argmax(probs, axis=1)
    m = classification_metrics(y, pred, NUM_CLASSES)
    return {
        'accuracy': float(m['accuracy']),
        'macro_f1': float(m['macro_f1']),
        'errors': int(np.sum(pred != y)),
        'pred': pred,
    }


def main():
    args = parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    nn, metadata = load_model(args.model)

    X, y, _paths = load_processed_dataset(
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

    baseline = evaluate(nn, X_test, y_test)

    # 1. Regression check: what happens if someone thickens every input.
    fixed_rows = []
    base_pred = baseline['pred']
    for delta in [0.0, 0.125, 0.25, 0.375, 0.50]:
        X_variant = X_test if delta == 0 else adjust_batch_stroke_width(X_test, delta)
        result = evaluate(nn, X_variant, y_test)
        changed = int(np.sum(result['pred'] != base_pred))
        fixed_rows.append({
            'delta_px': delta,
            'accuracy': result['accuracy'],
            'macro_f1': result['macro_f1'],
            'errors': result['errors'],
            'changed_predictions_vs_original': changed,
        })

    # 2. Stress test: simulate thinning associated with a thinner live brush,
    # then apply exactly the production correction for that brush size.
    brush_sizes = [8, 10, 12, 14, 16, 18, 22, 28]
    stress_rows = []
    for brush in brush_sizes:
        correction = live_brush_correction_px(brush)
        # Controlled domain shift: remove approximately as much stroke width
        # as the live policy is designed to restore.
        X_stressed = (
            X_test
            if correction == 0
            else adjust_batch_stroke_width(X_test, -correction)
        )
        raw = evaluate(nn, X_stressed, y_test)
        X_recovered = (
            X_stressed
            if correction == 0
            else adjust_batch_stroke_width(X_stressed, correction)
        )
        recovered = evaluate(nn, X_recovered, y_test)
        stress_rows.append({
            'simulated_brush': float(brush),
            'correction_px': float(correction),
            'raw_accuracy': raw['accuracy'],
            'raw_macro_f1': raw['macro_f1'],
            'raw_errors': raw['errors'],
            'normalized_accuracy': recovered['accuracy'],
            'normalized_macro_f1': recovered['macro_f1'],
            'normalized_errors': recovered['errors'],
        })

    with (args.results / 'fixed_thickening.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fixed_rows[0].keys())
        w.writeheader(); w.writerows(fixed_rows)

    with (args.results / 'thin_brush_stress.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=stress_rows[0].keys())
        w.writeheader(); w.writerows(stress_rows)

    # Print-friendly plot.
    fig, ax = plt.subplots(figsize=(9.5, 5.5), facecolor='white')
    xs = [r['simulated_brush'] for r in stress_rows]
    raw_f1 = [100 * r['raw_macro_f1'] for r in stress_rows]
    norm_f1 = [100 * r['normalized_macro_f1'] for r in stress_rows]
    ax.plot(xs, raw_f1, marker='o', linestyle='--', color='0.45', label='Без нормализации')
    ax.plot(xs, norm_f1, marker='s', linestyle='-', color='black', label='Адаптивная нормализация')
    ax.set_xlabel('Имитируемая толщина кисти GUI')
    ax.set_ylabel('Макро-F1, %')
    ax.set_title('Устойчивость распознавания к истончению штрихов')
    ax.set_xticks(xs)
    ax.grid(True, linestyle=':', color='0.82')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.results / 'thin_brush_stress.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    summary = {
        'model': str(args.model),
        'custom_closed_test_samples': int(len(y_test)),
        'baseline': {k: v for k, v in baseline.items() if k != 'pred'},
        'production_policy': {
            'brush_8_delta_px': live_brush_correction_px(8),
            'brush_13_delta_px': live_brush_correction_px(13),
            'brush_18_delta_px': live_brush_correction_px(18),
            'brush_28_delta_px': live_brush_correction_px(28),
            'loaded_images_modified': False,
        },
        'fixed_thickening': fixed_rows,
        'thin_brush_stress': stress_rows,
    }
    (args.results / 'benchmark_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print('=== FINAL INFERENCE BENCHMARK ===')
    print(
        f"Baseline custom test: Accuracy={baseline['accuracy']:.6f}, "
        f"Macro-F1={baseline['macro_f1']:.6f}, errors={baseline['errors']}/{len(y_test)}"
    )
    print('\nFixed thickening on normal test images:')
    for r in fixed_rows:
        print(
            f"  +{r['delta_px']:.3f}px -> F1={r['macro_f1']:.6f}, "
            f"errors={r['errors']}, changed={r['changed_predictions_vs_original']}"
        )
    print('\nSynthetic thin-brush stress:')
    for r in stress_rows:
        print(
            f"  brush={r['simulated_brush']:.0f}, correction=+{r['correction_px']:.3f}px | "
            f"raw F1={r['raw_macro_f1']:.6f}, errors={r['raw_errors']} -> "
            f"normalized F1={r['normalized_macro_f1']:.6f}, errors={r['normalized_errors']}"
        )


if __name__ == '__main__':
    main()
