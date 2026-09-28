from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_utils import augment_image_array_extended, preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Предпросмотр расширенной аугментации одного символа."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("augmentation_preview.png"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with Image.open(args.image) as image:
        base = preprocess_image(image)

    rng = np.random.default_rng(args.seed)
    images = [base]
    for _ in range(max(1, args.count - 1)):
        images.append(augment_image_array_extended(base, rng))

    cols = 4
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 2 * rows))
    axes = np.asarray(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title("исходное" if i == 0 else f"вариант {i}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    plt.close()
    print(f"Предпросмотр сохранён: {args.output.resolve()}")


if __name__ == "__main__":
    main()
