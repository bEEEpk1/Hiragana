from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from config import IMAGE_SIZE, KANA
from data_utils import augment_image_array_extended


def save_augmentation_examples(
    X: np.ndarray,
    y: np.ndarray,
    output: Path,
    seed: int = 42,
    rows: int = 4,
    variants: int = 6,
) -> None:
    """Сохраняет отчетный пример работы аугментации.

    Для каждой строки берется реальный обучающий образец:
    исходное изображение + несколько независимо сгенерированных вариантов.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in [
        "Arial Unicode MS",
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Yu Gothic",
        "YuGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
    ]:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break

    rng = np.random.default_rng(seed)

    # По возможности берем разные классы.
    class_ids = np.unique(y)
    chosen_classes = rng.choice(
        class_ids,
        size=min(rows, len(class_ids)),
        replace=False,
    )

    fig, axes = plt.subplots(
        len(chosen_classes),
        variants + 1,
        figsize=(2.0 * (variants + 1), 2.05 * len(chosen_classes)),
        facecolor="white",
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row, class_id in enumerate(chosen_classes):
        indices = np.where(y == class_id)[0]
        base_idx = int(rng.choice(indices))
        base = X[base_idx].reshape(IMAGE_SIZE, IMAGE_SIZE)

        images = [base]
        for _ in range(variants):
            images.append(augment_image_array_extended(base, rng))

        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(
                    "Исходное" if col == 0 else f"Вариант {col}",
                    fontsize=10,
                )
            if col == 0:
                ax.text(
                    -0.10, 0.5,
                    KANA[int(class_id)],
                    transform=ax.transAxes,
                    ha="right", va="center",
                    fontsize=18,
                )

    fig.suptitle("Примеры аугментации обучающих изображений", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)
