from __future__ import annotations

import argparse
from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from config import ALIAS_TO_CLASS_ID, IMAGE_SIZE, KANA, ROMAJI
from data_utils import preprocess_image
from inference_utils import normalize_live_canvas
from neural_network import load_model

CANVAS_SIZE = 400
DEFAULT_BRUSH_SIZE = 18.0
DEFAULT_CUSTOM_DATASET = Path('custom_dataset')
DEFAULT_MODEL = Path('models/best_model_final.pkl')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Финальная система распознавания хираганы'
    )
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--image', type=Path, default=None)
    parser.add_argument('--confidence-threshold', type=float, default=0.60)
    parser.add_argument(
        '--custom-data',
        type=Path,
        default=DEFAULT_CUSTOM_DATASET,
    )
    return parser.parse_args()


def resolve_class_labels(metadata: dict | None) -> tuple[list[str], list[str]]:
    metadata = metadata or {}
    romaji = metadata.get('class_romaji')
    kana = metadata.get('class_hiragana')
    if romaji is None or kana is None:
        return list(ROMAJI), list(KANA)
    romaji = list(romaji)
    kana = list(kana)
    if len(romaji) != len(ROMAJI) or len(kana) != len(KANA):
        return list(ROMAJI), list(KANA)
    return romaji, kana


def predict_processed(nn, metadata: dict, processed: np.ndarray):
    X = np.asarray(processed, dtype=np.float32).reshape(
        1, IMAGE_SIZE * IMAGE_SIZE
    )
    probs = nn.predict_proba(X)[0]
    romaji, kana = resolve_class_labels(metadata)
    if len(probs) != len(romaji):
        raise ValueError(
            f'Несовместимая модель: выходов={len(probs)}, ожидается={len(romaji)}.'
        )
    top3 = np.argsort(probs)[::-1][:3]
    return [
        {
            'class_id': int(i),
            'romaji': romaji[i],
            'kana': kana[i],
            'probability': float(probs[i]),
        }
        for i in top3
    ]


def predict_image(nn, metadata: dict, image: Image.Image):
    processed = preprocess_image(image)
    return predict_processed(nn, metadata, processed), processed


def cli_predict(nn, metadata, image_path: Path):
    with Image.open(image_path) as image:
        top3, _ = predict_image(nn, metadata, image)
    print(f'Изображение: {image_path}')
    for place, row in enumerate(top3, 1):
        print(
            f"{place}. {row['kana']} ({row['romaji']}) — "
            f"{row['probability'] * 100:.2f}%"
        )


class HiraganaApp:
    def __init__(
        self,
        root,
        nn,
        metadata,
        confidence_threshold: float,
        custom_data_dir: Path,
    ):
        self.root = root
        self.nn = nn
        self.metadata = metadata
        self.confidence_threshold = confidence_threshold
        self.custom_data_dir = custom_data_dir
        self.input_source = 'canvas'

        root.title('Нейросетевое распознавание хираганы')
        root.resizable(True, True)

        outer = tk.Frame(root)
        outer.pack(fill='both', expand=True)
        scroll = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(outer, orient='vertical', command=scroll.yview)
        scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        scroll.pack(side='left', fill='both', expand=True)

        ui = tk.Frame(scroll)
        win = scroll.create_window((0, 0), window=ui, anchor='nw')
        ui.bind('<Configure>', lambda _e: scroll.configure(scrollregion=scroll.bbox('all')))
        scroll.bind('<Configure>', lambda e: scroll.itemconfigure(win, width=e.width))

        def wheel(event):
            if getattr(event, 'delta', 0):
                scroll.yview_scroll(-1 if event.delta > 0 else 1, 'units')
            return 'break'
        scroll.bind_all('<MouseWheel>', wheel)

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.geometry(f"{min(820, screen_w - 100)}x{min(900, screen_h - 100)}")

        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.drawer = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.loaded_canvas_image = None
        self.preview_image = None
        self.brush_size = tk.DoubleVar(value=DEFAULT_BRUSH_SIZE)
        self.brush_size_text = tk.StringVar(value=f'{DEFAULT_BRUSH_SIZE:.1f}')

        tk.Label(
            ui,
            text=(
                '46 базовых символов хираганы. '
                'Дакутэн/хандакутэн и малые знаки не входят в классы модели.'
            ),
            anchor='w',
            justify='left',
            font=('Arial', 11),
        ).grid(row=0, column=0, columnspan=4, sticky='w', padx=12, pady=(10, 0))

        self.canvas = tk.Canvas(
            ui,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg='white',
            cursor='cross',
            highlightthickness=1,
        )
        self.canvas.grid(row=1, column=0, columnspan=4, padx=12, pady=12)
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

        tk.Button(ui, text='Распознать', command=self.recognize, width=16).grid(row=2, column=0, padx=5, pady=6)
        tk.Button(ui, text='Очистить', command=self.clear, width=16).grid(row=2, column=1, padx=5, pady=6)
        tk.Button(ui, text='Загрузить JPG/PNG', command=self.load_image, width=16).grid(row=2, column=2, padx=5, pady=6)
        tk.Button(ui, text='Сохранить пример', command=self.save_labeled_sample, width=16).grid(row=2, column=3, padx=5, pady=6)

        brush_frame = tk.Frame(ui)
        brush_frame.grid(row=3, column=0, columnspan=4, sticky='ew', padx=12)
        tk.Label(brush_frame, text='Толщина кисти:').pack(side='left')

        def update_brush(value):
            self.brush_size_text.set(f'{float(value):.1f}')

        ttk.Scale(
            brush_frame,
            from_=8,
            to=28,
            orient='horizontal',
            variable=self.brush_size,
            length=220,
            command=update_brush,
        ).pack(side='left', padx=(8, 10))
        tk.Label(brush_frame, textvariable=self.brush_size_text, width=4).pack(side='left')

        self.result_label = tk.Label(
            ui,
            text='Нарисуй один символ хираганы.',
            justify='left',
            anchor='w',
            font=('Arial', 15),
        )
        self.result_label.grid(row=4, column=0, columnspan=4, sticky='w', padx=12, pady=(9, 4))

        self.norm_label = tk.Label(
            ui,
            text='',
            justify='left',
            anchor='w',
            font=('Arial', 10),
        )
        self.norm_label.grid(row=5, column=0, columnspan=4, sticky='w', padx=12, pady=(0, 3))

        tk.Label(
            ui,
            text=(
                'Проценты — относительные оценки внутри 46 известных классов; '
                'они не являются OOD-проверкой.'
            ),
            justify='left',
            anchor='w',
            wraplength=700,
            font=('Arial', 10),
        ).grid(row=6, column=0, columnspan=4, sticky='w', padx=12, pady=(0, 5))

        tk.Label(
            ui,
            text='Фактический вход модели после preprocessing (32×32):',
            font=('Arial', 10),
        ).grid(row=7, column=0, columnspan=4, pady=(4, 2))
        self.preview_label = tk.Label(ui)
        self.preview_label.grid(row=8, column=0, columnspan=4, pady=(0, 12))

        for col in range(4):
            ui.grid_columnconfigure(col, weight=1, uniform='controls')

    def start_draw(self, event):
        self.input_source = 'canvas'
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.last_x is None:
            self.start_draw(event)
            return
        self.input_source = 'canvas'
        width = int(round(self.brush_size.get()))
        self.canvas.create_line(
            self.last_x, self.last_y, event.x, event.y,
            fill='black', width=width, capstyle=tk.ROUND, smooth=True,
        )
        self.drawer.line(
            (self.last_x, self.last_y, event.x, event.y),
            fill=0, width=width,
        )
        self.last_x, self.last_y = event.x, event.y

    def stop_draw(self, _event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.drawer = ImageDraw.Draw(self.image)
        self.input_source = 'canvas'
        self.loaded_canvas_image = None
        self.result_label.config(text='Нарисуй один символ хираганы.')
        self.norm_label.config(text='')
        self.preview_label.config(image='')
        self.preview_image = None

    def recognize(self):
        try:
            processed = preprocess_image(self.image)
            delta = 0.0
            if self.input_source == 'canvas':
                processed, delta = normalize_live_canvas(
                    processed,
                    self.brush_size.get(),
                )
            top3 = predict_processed(self.nn, self.metadata, processed)
        except Exception as exc:
            messagebox.showerror('Ошибка', str(exc))
            return

        best = top3[0]
        if best['probability'] < self.confidence_threshold:
            header = f"Низкая уверенность модели. Лучший класс: {best['kana']} ({best['romaji']})"
        else:
            header = f"Результат: {best['kana']} ({best['romaji']})"

        lines = [
            f"{place}. {row['kana']} ({row['romaji']}) — {row['probability'] * 100:.2f}%"
            for place, row in enumerate(top3, 1)
        ]
        self.result_label.config(text=header + '\n' + '\n'.join(lines))

        if self.input_source == 'canvas' and delta > 0:
            self.norm_label.config(
                text=(
                    f'Адаптивная нормализация толщины: +{delta:.2f} px '
                    f'(кисть {self.brush_size.get():.1f}).'
                )
            )
        elif self.input_source == 'canvas':
            self.norm_label.config(
                text='Нормализация толщины не требуется для текущей кисти.'
            )
        else:
            self.norm_label.config(
                text='Загруженное изображение: толщина штриха автоматически не изменяется.'
            )

        preview = Image.fromarray(
            np.uint8(np.clip(processed, 0, 1) * 255), mode='L'
        ).resize((180, 180), Image.Resampling.NEAREST)
        self.preview_image = ImageTk.PhotoImage(preview)
        self.preview_label.config(image=self.preview_image)

    def load_image(self):
        path = filedialog.askopenfilename(
            title='Выбери изображение',
            filetypes=[
                ('Изображения', '*.jpg *.jpeg *.png *.bmp'),
                ('Все файлы', '*.*'),
            ],
        )
        if not path:
            return
        try:
            with Image.open(path) as source:
                loaded = source.convert('L')
            display = loaded.copy()
            display.thumbnail((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)
            background = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
            x = (CANVAS_SIZE - display.width) // 2
            y = (CANVAS_SIZE - display.height) // 2
            background.paste(display, (x, y))
            self.image = background
            self.drawer = ImageDraw.Draw(self.image)
            self.input_source = 'file'
            self.loaded_canvas_image = ImageTk.PhotoImage(background)
            self.canvas.delete('all')
            self.canvas.create_image(0, 0, anchor='nw', image=self.loaded_canvas_image)
            self.recognize()
        except Exception as exc:
            messagebox.showerror('Ошибка', str(exc))

    def save_labeled_sample(self):
        try:
            preprocess_image(self.image)
        except Exception as exc:
            messagebox.showerror('Ошибка', str(exc))
            return
        label = simpledialog.askstring(
            'Истинный класс',
            "Введи ромадзи или символ хираганы, например 'te' или 'て':",
            parent=self.root,
        )
        if label is None:
            return
        key = label.strip().lower()
        if key not in ALIAS_TO_CLASS_ID:
            messagebox.showerror('Неизвестный класс', 'Допустимы только 46 базовых символов хираганы.')
            return
        class_id = ALIAS_TO_CLASS_ID[key]
        romaji = ROMAJI[class_id]
        kana = KANA[class_id]
        target_dir = self.custom_data_dir / romaji
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime('%Y%m%d_%H%M%S_%f.png')
        path = target_dir / filename
        self.image.save(path)
        messagebox.showinfo('Пример сохранён', f'{kana} ({romaji})\n{path}')


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f'Модель не найдена: {args.model.resolve()}\n'
            'Ожидается models/best_model_final.pkl.'
        )
    nn, metadata = load_model(args.model)
    if args.image is not None:
        cli_predict(nn, metadata, args.image)
        return
    root = tk.Tk()
    HiraganaApp(root, nn, metadata, args.confidence_threshold, args.custom_data)
    root.mainloop()


if __name__ == '__main__':
    main()
