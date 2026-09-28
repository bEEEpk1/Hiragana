from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from config import IMAGE_SIZE

# Live GUI policy fixed after A/B testing.
LIVE_BRUSH_MIN = 8.0
LIVE_BRUSH_TARGET = 18.0
MAX_LIVE_THICKENING_PX = 0.50
MORPH_WORK_SCALE = 8


def adjust_stroke_width(
    processed: np.ndarray,
    delta_px: float,
    work_scale: int = MORPH_WORK_SCALE,
) -> np.ndarray:
    """Меняет толщину белого штриха на черном 32x32 фоне.

    delta_px > 0  -> мягкое утолщение
    delta_px < 0  -> мягкое истончение
    delta_px == 0 -> изображение не меняется

    Для дробных изменений операция выполняется в увеличенном пространстве:
    при scale=8 один шаг 3x3 morphology соответствует примерно 0.125 px
    в итоговом 32x32 изображении.
    """
    arr = np.asarray(processed, dtype=np.float32)
    if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Ожидалось изображение {(IMAGE_SIZE, IMAGE_SIZE)}, "
            f"получено {arr.shape}."
        )

    delta_px = float(delta_px)
    steps = int(round(abs(delta_px) * work_scale))
    if steps <= 0:
        return arr.copy()

    pil = Image.fromarray(
        np.uint8(np.clip(arr, 0, 1) * 255),
        mode='L',
    )
    work_size = IMAGE_SIZE * work_scale
    pil = pil.resize(
        (work_size, work_size),
        Image.Resampling.BICUBIC,
    )

    filt = (
        ImageFilter.MaxFilter(size=3)
        if delta_px > 0
        else ImageFilter.MinFilter(size=3)
    )
    for _ in range(steps):
        pil = pil.filter(filt)

    pil = pil.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )
    return np.clip(
        np.asarray(pil, dtype=np.float32) / 255.0,
        0.0,
        1.0,
    )


def live_brush_correction_px(
    brush_size: float,
    min_brush: float = LIVE_BRUSH_MIN,
    target_brush: float = LIVE_BRUSH_TARGET,
    max_delta_px: float = MAX_LIVE_THICKENING_PX,
) -> float:
    """Адаптивная коррекция толщины для live-canvas.

    Зафиксированная production-политика:
      brush <= 8  -> +0.50 px
      brush 13    -> примерно +0.25 px
      brush >= 18 -> 0 px

    Толстые штрихи никогда дополнительно не расширяются.
    """
    brush = float(brush_size)
    if brush >= target_brush:
        return 0.0
    if brush <= min_brush:
        return float(max_delta_px)

    fraction = (target_brush - brush) / (target_brush - min_brush)
    return float(max_delta_px * fraction)


def normalize_live_canvas(
    processed: np.ndarray,
    brush_size: float,
) -> tuple[np.ndarray, float]:
    """Применяет production-нормализацию live-ввода и возвращает delta."""
    delta = live_brush_correction_px(brush_size)
    return adjust_stroke_width(processed, delta), delta


def adjust_batch_stroke_width(
    X: np.ndarray,
    delta_px: float,
) -> np.ndarray:
    """Batch-версия для benchmark; X имеет форму (N, 1024)."""
    X = np.asarray(X, dtype=np.float32)
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = adjust_stroke_width(
            X[i].reshape(IMAGE_SIZE, IMAGE_SIZE),
            delta_px,
        ).reshape(-1)
    return out
