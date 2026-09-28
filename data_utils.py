from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config import (
    ALIAS_TO_CLASS_ID,
    CONTENT_SIZE,
    IMAGE_SIZE,
    KANA,
    NUM_CLASSES,
    ROMAJI,
    SUPPORTED_EXTENSIONS,
)


def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Преобразует исходное изображение в квадрат image_size x image_size.

    Штрихи символа становятся значениями около 1, фон — около 0.
    Один и тот же preprocessing используется и при обучении, и при предсказании.
    """
    gray = ImageOps.autocontrast(image.convert("L"))
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    ink = 1.0 - arr

    # Белый фон после инверсии близок к 0. Небольшой порог отсекает шум JPEG.
    mask = ink > 0.08
    if not mask.any():
        raise ValueError("На изображении не найден символ.")

    ys, xs = np.where(mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    ink = ink[top:bottom, left:right]

    if ink.size == 0:
        raise ValueError("После preprocessing изображение оказалось пустым.")

    h, w = ink.shape
    scale = min(CONTENT_SIZE / max(w, 1), CONTENT_SIZE / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    ink_img = Image.fromarray(np.uint8(np.clip(ink, 0, 1) * 255), mode="L")
    ink_img = ink_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (image_size, image_size), 0)
    x = (image_size - new_w) // 2
    y = (image_size - new_h) // 2
    canvas.paste(ink_img, (x, y))

    return np.asarray(canvas, dtype=np.float32) / 255.0


def augment_image_array(
    image: np.ndarray,
    rng: np.random.Generator,
    max_angle: float = 7.0,
    max_shift: int = 2,
) -> np.ndarray:
    """Мягкая аугментация: небольшой поворот и сдвиг.

    Flip и большие повороты не используются, чтобы не менять смысл символа.
    """
    pil = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255), mode="L")
    angle = float(rng.uniform(-max_angle, max_angle))
    pil = pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    shifted = Image.new("L", pil.size, 0)
    shifted.paste(pil, (dx, dy))
    return np.asarray(shifted, dtype=np.float32) / 255.0



def augment_image_array_extended(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Расширенная мягкая аугментация рукописной хираганы.

    Цель — приблизить TRAIN к реальному пользовательскому почерку, не меняя
    смысл символа. Преобразования выполняются на 64x64, затем изображение
    возвращается к IMAGE_SIZE, чтобы изменение толщины и геометрии было мягче.

    Случайно применяются:
    - rotation: примерно ±10°
    - translation: примерно ±3 px в масштабе 32x32
    - global scale: 0.90..1.10
    - stretch X/Y: 0.88..1.12
    - shear: примерно ±7°
    - изменение толщины штриха
    - редкий слабый blur

    Flip и большие повороты запрещены.
    """
    base = Image.fromarray(
        np.uint8(np.clip(image, 0, 1) * 255),
        mode="L",
    )

    work_size = IMAGE_SIZE * 2
    pil = base.resize((work_size, work_size), Image.Resampling.BICUBIC)

    # 1. Толщина штриха. Белые пиксели = штрих, черные = фон.
    # MaxFilter расширяет светлую область, MinFilter немного сужает.
    if rng.random() < 0.70:
        mode = int(rng.integers(0, 3))  # 0=thin, 1=unchanged, 2=thick
        if mode == 0:
            pil = pil.filter(ImageFilter.MinFilter(size=3))
        elif mode == 2:
            pil = pil.filter(ImageFilter.MaxFilter(size=3))

    # 2. Scale + независимое растяжение по X/Y.
    global_scale = float(rng.uniform(0.90, 1.10))
    stretch_x = float(rng.uniform(0.88, 1.12))
    stretch_y = float(rng.uniform(0.88, 1.12))
    sx = global_scale * stretch_x
    sy = global_scale * stretch_y

    new_w = max(8, int(round(work_size * sx)))
    new_h = max(8, int(round(work_size * sy)))
    resized = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)

    scaled_canvas = Image.new("L", (work_size, work_size), 0)
    paste_x = (work_size - new_w) // 2
    paste_y = (work_size - new_h) // 2
    scaled_canvas.paste(resized, (paste_x, paste_y))
    pil = scaled_canvas

    # 3. Mild shear по X. PIL ожидает обратное affine-преобразование.
    if rng.random() < 0.55:
        shear_degrees = float(rng.uniform(-7.0, 7.0))
        shear = np.tan(np.deg2rad(shear_degrees))
        cy = (work_size - 1) / 2.0
        pil = pil.transform(
            (work_size, work_size),
            Image.Transform.AFFINE,
            (1.0, -shear, shear * cy, 0.0, 1.0, 0.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    # 4. Небольшой наклон.
    angle = float(rng.uniform(-10.0, 10.0))
    pil = pil.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )

    # 5. Сдвиг. ±4 px на 64x64 ~= ±2 px на итоговом 32x32.
    dx = int(rng.integers(-6, 7))
    dy = int(rng.integers(-6, 7))
    shifted = Image.new("L", (work_size, work_size), 0)
    shifted.paste(pil, (dx, dy))
    pil = shifted

    # 6. Редкое слабое сглаживание, имитирующее различную отрисовку/скан.
    if rng.random() < 0.25:
        pil = pil.filter(
            ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.60)))
        )

    # 7. Небольшая вариативность интенсивности штриха.
    # Фон остается черным, меняется только яркость светлого штриха.
    if rng.random() < 0.25:
        pil = ImageEnhance.Brightness(pil).enhance(
            float(rng.uniform(0.78, 1.00))
        )

    pil = pil.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def augment_training_set_extended(
    X: np.ndarray,
    y: np.ndarray,
    copies: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Создает одну новую расширенную augmented-версию TRAIN.

    При on-the-fly обучении вызывается заново перед каждой эпохой с новым seed.
    Исходные объекты всегда сохраняются в выборке.
    """
    if copies <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    total = len(X) * (copies + 1)
    X_out = np.empty((total, X.shape[1]), dtype=np.float32)
    y_out = np.empty(total, dtype=np.int64)

    cursor = 0
    for i in range(len(X)):
        base = X[i].reshape(IMAGE_SIZE, IMAGE_SIZE)

        X_out[cursor] = X[i]
        y_out[cursor] = y[i]
        cursor += 1

        for _ in range(copies):
            aug = augment_image_array_extended(base, rng)
            X_out[cursor] = aug.reshape(-1)
            y_out[cursor] = y[i]
            cursor += 1

    perm = rng.permutation(total)
    return X_out[perm], y_out[perm]


def build_or_load_augmented_pool(
    X_base: np.ndarray,
    y_base: np.ndarray,
    pool_size: int,
    cache_path: Path,
    seed: int = 42,
    rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Создает или загружает широкий кэшированный обучающий пул.

    Пул строится только из уже выделенной обучающей части.
    Все базовые изображения включаются хотя бы один раз, остальные позиции
    заполняются независимыми расширенными аугментациями.

    X хранится как uint8: для 100 000 изображений 32x32 это около 100 МБ.
    В float32 переводится только текущий mini-batch.
    """
    pool_size = int(pool_size)
    if pool_size < len(X_base):
        raise ValueError(
            f"Размер пула ({pool_size}) меньше базовой обучающей "
            f"выборки ({len(X_base)})."
        )

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=False)
        cached_size = int(data["pool_size"])
        cached_seed = int(data["seed"])
        cached_image_size = int(data["image_size"])
        cached_base_size = int(data["base_size"])

        if (
            cached_size == pool_size
            and cached_seed == int(seed)
            and cached_image_size == IMAGE_SIZE
            and cached_base_size == len(X_base)
        ):
            X_pool = data["X_pool"]
            y_pool = data["y_pool"].astype(np.int64, copy=False)
            print(
                f"Широкий обучающий пул загружен из кэша: "
                f"{cache_path} -> {len(y_pool)} изображений"
            )
            return X_pool, y_pool

        print(
            "Кэш широкого пула не соответствует текущим параметрам "
            "и будет пересоздан."
        )

    rng = np.random.default_rng(seed)
    X_pool = np.empty(
        (pool_size, X_base.shape[1]),
        dtype=np.uint8,
    )
    y_pool = np.empty(pool_size, dtype=np.uint8)

    n_base = len(X_base)
    X_pool[:n_base] = np.rint(
        np.clip(X_base, 0, 1) * 255
    ).astype(np.uint8)
    y_pool[:n_base] = y_base.astype(np.uint8)

    remaining = pool_size - n_base
    print(
        f"Создание широкого обучающего пула: {pool_size} изображений "
        f"({n_base} исходных + {remaining} аугментированных)"
    )

    for out_idx in range(n_base, pool_size):
        source_idx = int(rng.integers(0, n_base))
        base = X_base[source_idx].reshape(
            IMAGE_SIZE, IMAGE_SIZE
        )
        aug = augment_image_array_extended(base, rng)
        X_pool[out_idx] = np.rint(
            aug.reshape(-1) * 255
        ).astype(np.uint8)
        y_pool[out_idx] = np.uint8(
            y_base[source_idx]
        )

        completed = out_idx + 1
        if completed % 5000 == 0 or completed == pool_size:
            print(
                f"Формирование пула: "
                f"{completed}/{pool_size}"
            )

    perm = rng.permutation(pool_size)
    X_pool = X_pool[perm]
    y_pool = y_pool[perm]

    # Без сжатия: формирование и последующая загрузка существенно быстрее.
    np.savez(
        cache_path,
        X_pool=X_pool,
        y_pool=y_pool,
        pool_size=np.asarray(
            pool_size, dtype=np.int64
        ),
        seed=np.asarray(seed, dtype=np.int64),
        image_size=np.asarray(
            IMAGE_SIZE, dtype=np.int64
        ),
        base_size=np.asarray(
            n_base, dtype=np.int64
        ),
    )
    size_mb = cache_path.stat().st_size / (1024 ** 2)
    print(
        f"Широкий обучающий пул сохранён: "
        f"{cache_path} ({size_mb:.1f} МБ)"
    )
    return X_pool, y_pool


def stratified_subtrain_validation_split(
    y: np.ndarray,
    validation_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Стратифицированно делит уже выделенную TRAIN-часть на subtrain/validation.

    Для каждого класса гарантируется минимум один объект в validation
    и минимум один объект остается в subtrain.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio должен быть между 0 и 1.")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for class_id in range(NUM_CLASSES):
        idx = np.where(y == class_id)[0].copy()
        rng.shuffle(idx)
        n = len(idx)

        if n < 2:
            raise ValueError(
                f"Для класса {ROMAJI[class_id]} нужно минимум 2 объекта "
                "для внутреннего train/validation split."
            )

        n_val = int(round(n * validation_ratio))
        n_val = min(max(n_val, 1), n - 1)

        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _balanced_source_indices(
    y: np.ndarray,
    total_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Выбирает индексы примерно равномерно по всем 46 классам."""
    total_samples = int(total_samples)
    if total_samples <= 0:
        return np.empty(0, dtype=np.int64)

    per_class = total_samples // NUM_CLASSES
    remainder = total_samples % NUM_CLASSES

    selected: list[int] = []
    for class_id in range(NUM_CLASSES):
        class_idx = np.where(y == class_id)[0]
        if len(class_idx) == 0:
            raise ValueError(
                f"В источнике отсутствует класс {ROMAJI[class_id]}."
            )

        n_take = per_class + (1 if class_id < remainder else 0)
        chosen = rng.choice(
            class_idx,
            size=n_take,
            replace=True,
        )
        selected.extend(chosen.tolist())

    selected_arr = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_arr)
    return selected_arr


def build_or_load_domain_balanced_pool(
    X_kaggle: np.ndarray,
    y_kaggle: np.ndarray,
    X_custom: np.ndarray,
    y_custom: np.ndarray,
    pool_size: int,
    custom_fraction: float,
    cache_path: Path,
    seed: int = 42,
    rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Строит широкий пул с фиксированным балансом доменов и классов.

    Возвращает:
      X_pool uint8,
      y_pool uint8,
      domain_pool uint8: 0=исходный набор, 1=пользовательский почерк.

    Внутри каждого домена классы выбираются примерно равномерно.
    Каждый выбранный объект проходит независимую расширенную аугментацию.
    Небольшая доля исходных изображений сохраняется без искажений.
    """
    if not 0.0 < custom_fraction < 1.0:
        raise ValueError("custom_fraction должен быть между 0 и 1.")

    pool_size = int(pool_size)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    n_custom = int(round(pool_size * custom_fraction))
    n_kaggle = pool_size - n_custom

    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=False)
        ok = (
            int(data["pool_size"]) == pool_size
            and abs(float(data["custom_fraction"]) - custom_fraction) < 1e-12
            and int(data["seed"]) == int(seed)
            and int(data["image_size"]) == IMAGE_SIZE
            and int(data["kaggle_base_size"]) == len(X_kaggle)
            and int(data["custom_base_size"]) == len(X_custom)
        )
        if ok:
            X_pool = data["X_pool"]
            y_pool = data["y_pool"]
            domain_pool = data["domain_pool"]
            print(
                f"Сбалансированный обучающий пул загружен: "
                f"{cache_path} -> {len(y_pool)} изображений"
            )
            return X_pool, y_pool, domain_pool

        print(
            "Кэш сбалансированного пула не соответствует параметрам "
            "и будет пересоздан."
        )

    rng = np.random.default_rng(seed)

    kaggle_idx = _balanced_source_indices(
        y_kaggle,
        n_kaggle,
        rng,
    )
    custom_idx = _balanced_source_indices(
        y_custom,
        n_custom,
        rng,
    )

    X_pool = np.empty(
        (pool_size, X_kaggle.shape[1]),
        dtype=np.uint8,
    )
    y_pool = np.empty(pool_size, dtype=np.uint8)
    domain_pool = np.empty(pool_size, dtype=np.uint8)

    # Примерно 10% выбранных объектов оставляем почти исходными.
    original_probability = 0.10

    cursor = 0
    for domain_id, X_src, y_src, selected in (
        (0, X_kaggle, y_kaggle, kaggle_idx),
        (1, X_custom, y_custom, custom_idx),
    ):
        for source_idx in selected:
            base = X_src[int(source_idx)].reshape(
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
            if rng.random() < original_probability:
                arr = base
            else:
                arr = augment_image_array_extended(
                    base,
                    rng,
                )

            X_pool[cursor] = np.rint(
                np.clip(arr.reshape(-1), 0, 1) * 255
            ).astype(np.uint8)
            y_pool[cursor] = np.uint8(
                y_src[int(source_idx)]
            )
            domain_pool[cursor] = np.uint8(domain_id)
            cursor += 1

            if cursor % 5000 == 0 or cursor == pool_size:
                print(
                    f"Формирование сбалансированного пула: "
                    f"{cursor}/{pool_size}"
                )

    perm = rng.permutation(pool_size)
    X_pool = X_pool[perm]
    y_pool = y_pool[perm]
    domain_pool = domain_pool[perm]

    np.savez(
        cache_path,
        X_pool=X_pool,
        y_pool=y_pool,
        domain_pool=domain_pool,
        pool_size=np.asarray(pool_size, dtype=np.int64),
        custom_fraction=np.asarray(custom_fraction, dtype=np.float64),
        seed=np.asarray(seed, dtype=np.int64),
        image_size=np.asarray(IMAGE_SIZE, dtype=np.int64),
        kaggle_base_size=np.asarray(len(X_kaggle), dtype=np.int64),
        custom_base_size=np.asarray(len(X_custom), dtype=np.int64),
    )

    size_mb = cache_path.stat().st_size / (1024 ** 2)
    print(
        f"Сбалансированный пул сохранён: {cache_path} "
        f"({size_mb:.1f} МБ)"
    )
    print(
        f"Состав пула: исходный домен={n_kaggle}, "
        f"пользовательский домен={n_custom}"
    )
    return X_pool, y_pool, domain_pool


def build_finetune_pool(
    X_kaggle: np.ndarray,
    y_kaggle: np.ndarray,
    X_custom: np.ndarray,
    y_custom: np.ndarray,
    pool_size: int,
    custom_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Создает небольшой in-memory пул для финальной адаптации."""
    rng = np.random.default_rng(seed)
    n_custom = int(round(pool_size * custom_fraction))
    n_kaggle = pool_size - n_custom

    kaggle_idx = _balanced_source_indices(
        y_kaggle,
        n_kaggle,
        rng,
    )
    custom_idx = _balanced_source_indices(
        y_custom,
        n_custom,
        rng,
    )

    X_out = np.empty(
        (pool_size, X_kaggle.shape[1]),
        dtype=np.uint8,
    )
    y_out = np.empty(pool_size, dtype=np.uint8)

    cursor = 0
    for X_src, y_src, selected in (
        (X_kaggle, y_kaggle, kaggle_idx),
        (X_custom, y_custom, custom_idx),
    ):
        for source_idx in selected:
            base = X_src[int(source_idx)].reshape(
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
            # Fine-tuning мягче: 35% оставляем практически исходными.
            if rng.random() < 0.35:
                arr = base
            else:
                arr = augment_image_array_extended(
                    base,
                    rng,
                )
            X_out[cursor] = np.rint(
                np.clip(arr.reshape(-1), 0, 1) * 255
            ).astype(np.uint8)
            y_out[cursor] = np.uint8(
                y_src[int(source_idx)]
            )
            cursor += 1

    perm = rng.permutation(pool_size)
    return X_out[perm], y_out[perm]


def augment_image_array_hard_negative(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Мягкая аугментация для визуально близких проблемных классов.

    Здесь искажения намеренно слабее основной аугментации:
    - rotation: ±5°
    - shift: ±2 px на итоговом 32x32
    - scale/stretch: примерно 0.95..1.05
    - shear: ±3°
    - умеренное изменение толщины
    - brightness: 0.90..1.00
    - редкий очень слабый blur
    """
    base = Image.fromarray(
        np.uint8(np.clip(image, 0, 1) * 255),
        mode="L",
    )

    work_size = IMAGE_SIZE * 2
    pil = base.resize(
        (work_size, work_size),
        Image.Resampling.BICUBIC,
    )

    # Умеренное изменение толщины. Экстремальное thinning исключено.
    if rng.random() < 0.50:
        mode = int(
            rng.choice(
                np.asarray([0, 1, 2]),
                p=np.asarray([0.20, 0.45, 0.35]),
            )
        )
        if mode == 0:
            pil = pil.filter(ImageFilter.MinFilter(size=3))
        elif mode == 2:
            pil = pil.filter(ImageFilter.MaxFilter(size=3))

    global_scale = float(rng.uniform(0.95, 1.05))
    stretch_x = float(rng.uniform(0.95, 1.05))
    stretch_y = float(rng.uniform(0.95, 1.05))

    sx = global_scale * stretch_x
    sy = global_scale * stretch_y

    new_w = max(8, int(round(work_size * sx)))
    new_h = max(8, int(round(work_size * sy)))

    resized = pil.resize(
        (new_w, new_h),
        Image.Resampling.BICUBIC,
    )

    canvas = Image.new(
        "L",
        (work_size, work_size),
        0,
    )
    canvas.paste(
        resized,
        (
            (work_size - new_w) // 2,
            (work_size - new_h) // 2,
        ),
    )
    pil = canvas

    if rng.random() < 0.40:
        shear_degrees = float(
            rng.uniform(-3.0, 3.0)
        )
        shear = np.tan(
            np.deg2rad(shear_degrees)
        )
        cy = (work_size - 1) / 2.0
        pil = pil.transform(
            (work_size, work_size),
            Image.Transform.AFFINE,
            (
                1.0,
                -shear,
                shear * cy,
                0.0,
                1.0,
                0.0,
            ),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    angle = float(
        rng.uniform(-5.0, 5.0)
    )
    pil = pil.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )

    # ±4 px на 64x64 ~= ±2 px на итоговом 32x32.
    dx = int(rng.integers(-4, 5))
    dy = int(rng.integers(-4, 5))

    shifted = Image.new(
        "L",
        (work_size, work_size),
        0,
    )
    shifted.paste(pil, (dx, dy))
    pil = shifted

    if rng.random() < 0.10:
        pil = pil.filter(
            ImageFilter.GaussianBlur(
                radius=float(
                    rng.uniform(0.10, 0.35)
                )
            )
        )

    if rng.random() < 0.18:
        pil = ImageEnhance.Brightness(
            pil
        ).enhance(
            float(
                rng.uniform(0.90, 1.00)
            )
        )

    pil = pil.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    return np.clip(
        np.asarray(
            pil,
            dtype=np.float32,
        ) / 255.0,
        0.0,
        1.0,
    )


def _balanced_indices_for_classes(
    y: np.ndarray,
    total_samples: int,
    class_ids: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Выбирает total_samples с replacement равномерно по заданным классам."""
    if total_samples <= 0:
        return np.empty(0, dtype=np.int64)
    if not class_ids:
        raise ValueError("class_ids не может быть пустым.")

    per_class = total_samples // len(class_ids)
    remainder = total_samples % len(class_ids)
    out: list[int] = []

    for pos, class_id in enumerate(class_ids):
        available = np.where(y == class_id)[0]
        if len(available) == 0:
            raise ValueError(
                f"В источнике отсутствует класс "
                f"{ROMAJI[class_id]} ({KANA[class_id]})."
            )

        n_take = per_class + (
            1 if pos < remainder else 0
        )
        chosen = rng.choice(
            available,
            size=n_take,
            replace=True,
        )
        out.extend(chosen.tolist())

    result = np.asarray(
        out,
        dtype=np.int64,
    )
    rng.shuffle(result)
    return result


def build_hard_negative_micro_pool(
    X_kaggle: np.ndarray,
    y_kaggle: np.ndarray,
    X_custom: np.ndarray,
    y_custom: np.ndarray,
    pool_size: int,
    hard_fraction: float,
    hard_class_ids: list[int],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Формирует небольшой targeted-пул для micro-tuning.

    70% (по умолчанию) — обычный balanced train:
        50% Kaggle / 50% custom, все 46 классов.
    30% — hard-negative часть:
        50% Kaggle / 50% custom, только проблемные классы.

    Возвращает:
        X_pool_uint8,
        y_pool_uint8,
        sample_type: 0=обычный, 1=hard-negative.
    """
    pool_size = int(pool_size)
    if pool_size < 100:
        raise ValueError("pool_size слишком мал.")
    if not 0.0 < hard_fraction < 1.0:
        raise ValueError(
            "hard_fraction должен быть между 0 и 1."
        )

    rng = np.random.default_rng(seed)

    n_hard = int(
        round(pool_size * hard_fraction)
    )
    n_regular = pool_size - n_hard

    # Обе части держим 50/50 по доменам, чтобы не разрушать
    # найденный в v1.7.1 оптимальный domain balance.
    regular_kaggle = n_regular // 2
    regular_custom = n_regular - regular_kaggle
    hard_kaggle = n_hard // 2
    hard_custom = n_hard - hard_kaggle

    all_classes = list(range(NUM_CLASSES))

    regular_k_idx = _balanced_indices_for_classes(
        y_kaggle,
        regular_kaggle,
        all_classes,
        rng,
    )
    regular_c_idx = _balanced_indices_for_classes(
        y_custom,
        regular_custom,
        all_classes,
        rng,
    )
    hard_k_idx = _balanced_indices_for_classes(
        y_kaggle,
        hard_kaggle,
        hard_class_ids,
        rng,
    )
    hard_c_idx = _balanced_indices_for_classes(
        y_custom,
        hard_custom,
        hard_class_ids,
        rng,
    )

    X_out = np.empty(
        (pool_size, X_kaggle.shape[1]),
        dtype=np.uint8,
    )
    y_out = np.empty(
        pool_size,
        dtype=np.uint8,
    )
    sample_type = np.empty(
        pool_size,
        dtype=np.uint8,
    )

    cursor = 0

    def add_regular(
        X_src: np.ndarray,
        y_src: np.ndarray,
        indices: np.ndarray,
        clean_probability: float,
    ) -> None:
        nonlocal cursor
        for source_idx in indices:
            source_idx = int(source_idx)
            base = X_src[source_idx].reshape(
                IMAGE_SIZE,
                IMAGE_SIZE,
            )

            if rng.random() < clean_probability:
                arr = base
            else:
                # Для 70% обычной части сохраняем ту же расширенную
                # аугментацию, на которой была получена лучшая h64.
                arr = augment_image_array_extended(
                    base,
                    rng,
                )

            X_out[cursor] = np.rint(
                np.clip(
                    arr.reshape(-1),
                    0,
                    1,
                ) * 255
            ).astype(np.uint8)
            y_out[cursor] = np.uint8(
                y_src[source_idx]
            )
            sample_type[cursor] = 0
            cursor += 1

    def add_hard(
        X_src: np.ndarray,
        y_src: np.ndarray,
        indices: np.ndarray,
        clean_probability: float,
    ) -> None:
        nonlocal cursor
        for source_idx in indices:
            source_idx = int(source_idx)
            base = X_src[source_idx].reshape(
                IMAGE_SIZE,
                IMAGE_SIZE,
            )

            if rng.random() < clean_probability:
                arr = base
            else:
                arr = augment_image_array_hard_negative(
                    base,
                    rng,
                )

            X_out[cursor] = np.rint(
                np.clip(
                    arr.reshape(-1),
                    0,
                    1,
                ) * 255
            ).astype(np.uint8)
            y_out[cursor] = np.uint8(
                y_src[source_idx]
            )
            sample_type[cursor] = 1
            cursor += 1

    # В обычной части близко повторяем v1.7.1: около 10% clean.
    add_regular(
        X_kaggle,
        y_kaggle,
        regular_k_idx,
        clean_probability=0.10,
    )
    add_regular(
        X_custom,
        y_custom,
        regular_c_idx,
        clean_probability=0.10,
    )

    # Hard-классы оставляем чуть чаще чистыми: реальные различия
    # между близкими символами важнее агрессивной вариативности.
    add_hard(
        X_kaggle,
        y_kaggle,
        hard_k_idx,
        clean_probability=0.20,
    )
    add_hard(
        X_custom,
        y_custom,
        hard_c_idx,
        clean_probability=0.25,
    )

    if cursor != pool_size:
        raise RuntimeError(
            f"Ошибка формирования micro-pool: "
            f"{cursor} != {pool_size}"
        )

    perm = rng.permutation(pool_size)
    return (
        X_out[perm],
        y_out[perm],
        sample_type[perm],
    )

def discover_dataset(root: Path) -> list[tuple[Path, int]]:
    """Ищет изображения и сопоставляет папки с 46 классами.

    Поддерживаются и romaji-папки из датасета (например aa, ii, uu, ee, oo, shi, tsu, nn),
    и папки, названные непосредственно символами хираганы.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Папка датасета не найдена: {root.resolve()}")

    class_dirs: dict[int, Path] = {}
    unknown_dirs: list[str] = []

    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        key = path.name.strip().lower()
        if key not in ALIAS_TO_CLASS_ID:
            unknown_dirs.append(path.name)
            continue
        class_id = ALIAS_TO_CLASS_ID[key]
        if class_id in class_dirs:
            raise ValueError(
                f"Один класс найден дважды: {class_dirs[class_id].name!r} и {path.name!r}."
            )
        class_dirs[class_id] = path

    missing = [ROMAJI[i] for i in range(NUM_CLASSES) if i not in class_dirs]
    if missing:
        raise ValueError(
            "Не найдены папки для всех 46 символов. Отсутствуют: " + ", ".join(missing)
        )
    if unknown_dirs:
        print("Предупреждение: проигнорированы неизвестные папки:", ", ".join(unknown_dirs))

    items: list[tuple[Path, int]] = []
    counts = []
    for class_id in range(NUM_CLASSES):
        class_dir = class_dirs[class_id]
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise ValueError(f"В папке {class_dir} нет изображений.")
        counts.append(len(files))
        items.extend((path, class_id) for path in files)

    print(f"Классов: {NUM_CLASSES}")
    print(f"Изображений: {len(items)}")
    print(f"Изображений на класс: min={min(counts)}, max={max(counts)}")
    return items


def load_processed_dataset(
    root: Path,
    cache_path: Path | None = None,
    rebuild_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Загружает и preprocess'ит весь датасет. Результат можно кэшировать в NPZ."""
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        paths = data["paths"].astype(str).tolist()
        if X.shape[1] != IMAGE_SIZE * IMAGE_SIZE:
            raise ValueError("Кэш создан для другого размера изображения. Используй --rebuild-cache.")
        print(f"Кэш набора данных загружен: {cache_path} -> X={X.shape}")
        return X, y, paths

    items = discover_dataset(root)
    X = np.empty((len(items), IMAGE_SIZE * IMAGE_SIZE), dtype=np.float32)
    y = np.empty(len(items), dtype=np.int64)
    paths: list[str] = []

    for i, (path, class_id) in enumerate(items):
        with Image.open(path) as image:
            processed = preprocess_image(image)
        X[i] = processed.reshape(-1)
        y[i] = class_id
        paths.append(str(path))
        if (i + 1) % 500 == 0 or i + 1 == len(items):
            print(f"Предварительная обработка: {i + 1}/{len(items)}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, paths=np.asarray(paths, dtype=object))
        print(f"Кэш набора данных сохранён: {cache_path}")

    return X, y, paths


def stratified_train_val_test_split(
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ручной stratified split по каждому классу."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_id in range(NUM_CLASSES):
        idx = np.where(y == class_id)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            raise ValueError(f"Для класса {ROMAJI[class_id]} нужно минимум 3 объекта.")

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx



def stratified_train_test_split(
    y: np.ndarray,
    train_ratio: float = 0.60,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split TRAIN/TEST внутри каждого класса.

    Используется для небольшого пользовательского датасета.
    Validation намеренно не выделяется: для early stopping остается
    исходная Kaggle validation-выборка.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio должен быть между 0 и 1.")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for class_id in range(NUM_CLASSES):
        idx = np.where(y == class_id)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        if n < 2:
            raise ValueError(
                f"Для custom-класса {ROMAJI[class_id]} нужно минимум 2 объекта."
            )

        n_train = int(round(n * train_ratio))
        n_train = min(max(n_train, 1), n - 1)

        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def save_train_test_manifest(
    root: Path,
    paths: list[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_path: Path,
) -> None:
    """Сохраняет фиксированный custom split относительными путями."""
    root = Path(root).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_by_index: dict[int, str] = {}
    for idx in train_idx:
        split_by_index[int(idx)] = "train"
    for idx in test_idx:
        split_by_index[int(idx)] = "test"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_id", "romaji", "hiragana", "relative_path"])
        for idx in range(len(y)):
            class_id = int(y[idx])
            path = Path(paths[idx]).resolve()
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = Path(path.name)
            writer.writerow([
                split_by_index[idx],
                class_id,
                ROMAJI[class_id],
                KANA[class_id],
                str(relative),
            ])

def augment_training_set(
    X: np.ndarray,
    y: np.ndarray,
    copies: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    total = len(X) * (copies + 1)
    X_out = np.empty((total, X.shape[1]), dtype=np.float32)
    y_out = np.empty(total, dtype=np.int64)

    cursor = 0
    for i in range(len(X)):
        base = X[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
        X_out[cursor] = X[i]
        y_out[cursor] = y[i]
        cursor += 1
        for _ in range(copies):
            aug = augment_image_array(base, rng)
            X_out[cursor] = aug.reshape(-1)
            y_out[cursor] = y[i]
            cursor += 1

    perm = rng.permutation(total)
    return X_out[perm], y_out[perm]


def save_split_manifest(
    paths: list[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_by_index: dict[int, str] = {}
    for idx in train_idx:
        split_by_index[int(idx)] = "train"
    for idx in val_idx:
        split_by_index[int(idx)] = "validation"
    for idx in test_idx:
        split_by_index[int(idx)] = "test"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_id", "romaji", "hiragana", "path"])
        for idx in range(len(y)):
            class_id = int(y[idx])
            writer.writerow([
                split_by_index[idx], class_id, ROMAJI[class_id], KANA[class_id], paths[idx]
            ])
