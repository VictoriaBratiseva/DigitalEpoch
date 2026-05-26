#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор синтетического датасета для YOLO: обнаружение revision cloud / HOLD cloud в PDF.

Что делает скрипт:
- принимает папку с PDF или изображениями без синтетических облачков;
- рендерит PDF-страницы в изображения;
- добавляет на изображения 0..N инженерных revision-cloud контуров;
- размещает слово HOLD фиксированного размера внутри или рядом с облачком, либо оставляет облачко без HOLD;
- сохраняет изображения и YOLO-разметку bbox облачков;
- дополнительно может сохранять preview с зелёными bbox для ручной проверки.

Установка:
    python -m pip install pillow pymupdf

Пример запуска:
    python generate_hold_cloud_dataset_v2.py --input-dir ./base_pages --output-dir ./hold_dataset --samples 50 --min-clouds 1 --max-clouds 5 --preview

Важно:
- bbox в YOLO-разметке относится к самому облачку, а не к тексту HOLD;
- размер HOLD задаётся параметром --hold-font-size и НЕ масштабируется вместе с облачком;
- форма облачка сделана похожей на инженерные revision cloud: волнистый контур; по умолчанию включён стиль wavy, а также немного увеличивается bbox, чтобы облачко целиком помещалось в разметку и preview.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_HOLD_MODES = {"inside", "right", "left", "top", "bottom", "absent"}
SUPPORTED_CLOUD_COLORS = {"red", "dark", "mixed"}
SUPPORTED_CLOUD_STYLES = {"sample", "wavy", "revision"}


@dataclass
class BasePage:
    image: Image.Image
    source_file: str
    page_index: Optional[int]


@dataclass
class ObjectMeta:
    class_id: int
    bbox_px: Tuple[int, int, int, int]
    bbox_yolo: Tuple[float, float, float, float]
    hold_mode: str
    text: str
    cloud_color: str


@dataclass
class SampleMeta:
    image_file: str
    label_file: str
    split: str
    source_file: str
    source_page_index: Optional[int]
    width: int
    height: int
    objects: List[ObjectMeta]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генератор YOLO-датасета с инженерными revision-cloud облачками HOLD."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Папка с исходными PDF/изображениями.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Папка, куда сохранить датасет.")
    parser.add_argument("--samples", type=int, default=50, help="Сколько изображений сгенерировать всего.")
    parser.add_argument("--min-clouds", type=int, default=0, help="Минимальное число облачков на изображении.")
    parser.add_argument("--max-clouds", type=int, default=5, help="Максимальное число облачков на изображении. По заданию лучше 3-5.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI для рендера PDF-страниц.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Доля validation-выборки.")
    parser.add_argument("--seed", type=int, default=None, help="Seed для воспроизводимости.")
    parser.add_argument("--class-name", default="hold_cloud", help="Название класса YOLO.")
    parser.add_argument("--class-id", type=int, default=0, help="ID класса в YOLO-разметке.")
    parser.add_argument(
        "--hold-modes",
        default="inside,right,left,top,bottom,absent",
        help="Позиции HOLD через запятую: inside,right,left,top,bottom,absent.",
    )
    parser.add_argument(
        "--label-only-with-hold",
        action="store_true",
        help="Если включить, облачка без HOLD будут отрицательными примерами и НЕ попадут в разметку.",
    )
    parser.add_argument(
        "--hold-font-size",
        type=int,
        default=26,
        help="Фиксированный размер шрифта HOLD в пикселях. Не зависит от размера облачка.",
    )
    parser.add_argument(
        "--hold-color",
        choices=["red", "dark", "mixed"],
        default="mixed",
        help="Цвет текста HOLD: red, dark или mixed.",
    )
    parser.add_argument(
        "--cloud-color",
        choices=sorted(SUPPORTED_CLOUD_COLORS),
        default="mixed",
        help="Цвет облачка: red, dark или mixed.",
    )
    parser.add_argument(
        "--cloud-line-width",
        type=int,
        default=4,
        help="Толщина линии облачка в пикселях. Лучше 2-5 для инженерного вида.",
    )
    parser.add_argument(
        "--cloud-bump-radius",
        type=int,
        default=14,
        help="Радиус мелких дуг по периметру облачка. Чем меньше, тем мельче волна.",
    )
    parser.add_argument(
        "--cloud-style",
        choices=sorted(SUPPORTED_CLOUD_STYLES),
        default="sample",
        help="Форма облачка: sample — как на чертеже-образце, wavy — более свободная волна, revision — регулярные дуги по периметру.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=6,
        help="Небольшой запас вокруг bbox облачка в пикселях, чтобы облачко целиком помещалось в разметку.",
    )
    parser.add_argument(
        "--preview-bbox-padding",
        type=int,
        default=8,
        help="Дополнительное увеличение bbox только на preview-картинках.",
    )
    parser.add_argument(
        "--min-cloud-width-ratio",
        type=float,
        default=0.08,
        help="Минимальная ширина облачка как доля ширины изображения.",
    )
    parser.add_argument(
        "--max-cloud-width-ratio",
        type=float,
        default=0.30,
        help="Максимальная ширина облачка как доля ширины изображения.",
    )
    parser.add_argument(
        "--min-cloud-height-ratio",
        type=float,
        default=0.035,
        help="Минимальная высота облачка как доля высоты изображения.",
    )
    parser.add_argument(
        "--max-cloud-height-ratio",
        type=float,
        default=0.13,
        help="Максимальная высота облачка как доля высоты изображения.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Дополнительно сохранить preview-картинки с зелёными bbox для проверки.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Дополнительно сохранить каждую сгенерированную картинку как raster-PDF для демонстрации.",
    )
    parser.add_argument("--jpg-quality-min", type=int, default=74, help="Минимальное качество JPEG.")
    parser.add_argument("--jpg-quality-max", type=int, default=96, help="Максимальное качество JPEG.")
    return parser.parse_args()


def split_hold_modes(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def validate_args(args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        errors.append(f"input-dir не существует или не является папкой: {args.input_dir}")
    if args.samples <= 0:
        errors.append("samples должен быть больше 0")
    if args.min_clouds < 0:
        errors.append("min-clouds не может быть меньше 0")
    if args.max_clouds < args.min_clouds:
        errors.append("max-clouds должен быть >= min-clouds")
    if args.max_clouds > 5:
        errors.append("max-clouds лучше не делать больше 5 по заданию")
    if args.dpi <= 0:
        errors.append("dpi должен быть больше 0")
    if not 0 <= args.val_ratio < 1:
        errors.append("val-ratio должен быть в диапазоне [0, 1)")
    if args.hold_font_size <= 0:
        errors.append("hold-font-size должен быть больше 0")
    if args.cloud_line_width <= 0:
        errors.append("cloud-line-width должен быть больше 0")
    if args.cloud_bump_radius <= 0:
        errors.append("cloud-bump-radius должен быть больше 0")
    if args.bbox_padding < 0:
        errors.append("bbox-padding не может быть меньше 0")
    if args.preview_bbox_padding < 0:
        errors.append("preview-bbox-padding не может быть меньше 0")
    if args.min_cloud_width_ratio <= 0 or args.max_cloud_width_ratio <= 0:
        errors.append("cloud width ratio должен быть больше 0")
    if args.min_cloud_height_ratio <= 0 or args.max_cloud_height_ratio <= 0:
        errors.append("cloud height ratio должен быть больше 0")
    if args.max_cloud_width_ratio < args.min_cloud_width_ratio:
        errors.append("max-cloud-width-ratio должен быть >= min-cloud-width-ratio")
    if args.max_cloud_height_ratio < args.min_cloud_height_ratio:
        errors.append("max-cloud-height-ratio должен быть >= min-cloud-height-ratio")
    if not 1 <= args.jpg_quality_min <= 100 or not 1 <= args.jpg_quality_max <= 100:
        errors.append("jpg quality должен быть от 1 до 100")
    if args.jpg_quality_min > args.jpg_quality_max:
        errors.append("jpg-quality-min должен быть <= jpg-quality-max")

    modes = split_hold_modes(args.hold_modes)
    unknown = set(modes) - SUPPORTED_HOLD_MODES
    if unknown:
        errors.append(f"Неизвестные hold-modes: {sorted(unknown)}. Можно: {sorted(SUPPORTED_HOLD_MODES)}")
    if not modes:
        errors.append("Нужно указать хотя бы один hold-mode")

    return errors


def find_input_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    allowed = SUPPORTED_IMAGE_EXTS | SUPPORTED_PDF_EXTS
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            files.append(path)
    return files


def render_pdf_pages(pdf_path: Path, dpi: int) -> List[BasePage]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("Для обработки PDF установите PyMuPDF: python -m pip install pymupdf") from exc

    pages: List[BasePage] = []
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pages.append(BasePage(image=image, source_file=pdf_path.name, page_index=page_index))

    doc.close()
    return pages


def load_image_page(image_path: Path) -> BasePage:
    image = Image.open(image_path).convert("RGB")
    return BasePage(image=image, source_file=image_path.name, page_index=None)


def load_base_pages(input_dir: Path, dpi: int) -> List[BasePage]:
    files = find_input_files(input_dir)
    if not files:
        raise RuntimeError(f"В папке {input_dir} не найдено PDF или изображений.")

    pages: List[BasePage] = []
    for file_path in files:
        ext = file_path.suffix.lower()
        if ext in SUPPORTED_PDF_EXTS:
            pages.extend(render_pdf_pages(file_path, dpi=dpi))
        elif ext in SUPPORTED_IMAGE_EXTS:
            pages.append(load_image_page(file_path))

    if not pages:
        raise RuntimeError("Не удалось загрузить ни одной страницы/картинки.")
    return pages


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_split(rng: random.Random, val_ratio: float) -> str:
    return "val" if rng.random() < val_ratio else "train"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def random_cloud_bbox(
    image_size: Tuple[int, int],
    rng: random.Random,
    existing: Sequence[Tuple[int, int, int, int]],
    args: argparse.Namespace,
) -> Optional[Tuple[int, int, int, int]]:
    width, height = image_size
    if width < 180 or height < 140:
        return None

    margin_x = int(width * 0.035)
    margin_y = int(height * 0.035)

    min_w = max(100, int(width * args.min_cloud_width_ratio))
    max_w = max(min_w + 1, int(width * args.max_cloud_width_ratio))
    min_h = max(45, int(height * args.min_cloud_height_ratio))
    max_h = max(min_h + 1, int(height * args.max_cloud_height_ratio))

    max_w = min(max_w, width - 2 * margin_x)
    max_h = min(max_h, height - 2 * margin_y)

    for _ in range(100):
        # В инженерных чертежах облачка часто вытянутые. Поэтому чаще генерируем широкий прямоугольник.
        if rng.random() < 0.72:
            bw = rng.randint(min_w, max_w)
            target_h = int(bw * rng.uniform(0.16, 0.36))
            bh = int(clamp(target_h, min_h, max_h))
        else:
            bw = rng.randint(min_w, max_w)
            bh = rng.randint(min_h, max_h)

        bw = min(bw, width - 2 * margin_x)
        bh = min(bh, height - 2 * margin_y)
        if bw <= 70 or bh <= 35:
            continue

        x1 = rng.randint(margin_x, max(margin_x, width - margin_x - bw))
        y1 = rng.randint(margin_y, max(margin_y, height - margin_y - bh))
        bbox = (x1, y1, x1 + bw, y1 + bh)

        if all(bbox_iou(bbox, old) < 0.10 for old in existing):
            return bbox

    return None


def wavy_rect_points(
    inner_bbox: Tuple[int, int, int, int],
    amp: float,
    step: int,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    x1, y1, x2, y2 = inner_bbox
    points: List[Tuple[float, float]] = []
    phase_top = rng.uniform(0, math.tau)
    phase_right = rng.uniform(0, math.tau)
    phase_bottom = rng.uniform(0, math.tau)
    phase_left = rng.uniform(0, math.tau)

    def side_count(length: float) -> int:
        return max(8, int(length / max(step, 1)))

    top_n = side_count(x2 - x1)
    right_n = side_count(y2 - y1)
    bottom_n = side_count(x2 - x1)
    left_n = side_count(y2 - y1)

    for i in range(top_n + 1):
        t = i / top_n
        x = x1 + (x2 - x1) * t
        y = y1 - amp * math.sin(t * math.tau * 3.0 + phase_top)
        points.append((x, y))

    for i in range(1, right_n + 1):
        t = i / right_n
        y = y1 + (y2 - y1) * t
        x = x2 + amp * math.sin(t * math.tau * 3.0 + phase_right)
        points.append((x, y))

    for i in range(1, bottom_n + 1):
        t = i / bottom_n
        x = x2 - (x2 - x1) * t
        y = y2 + amp * math.sin(t * math.tau * 3.0 + phase_bottom)
        points.append((x, y))

    for i in range(1, left_n + 1):
        t = i / left_n
        y = y2 - (y2 - y1) * t
        x = x1 - amp * math.sin(t * math.tau * 3.0 + phase_left)
        points.append((x, y))

    return points


def paste_rgba(base_rgb: Image.Image, overlay_rgba: Image.Image) -> None:
    composed = Image.alpha_composite(base_rgb.convert("RGBA"), overlay_rgba).convert("RGB")
    base_rgb.paste(composed)


def expand_bbox(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int], pad: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    W, H = image_size
    return (
        int(clamp(x1 - pad, 0, W)),
        int(clamp(y1 - pad, 0, H)),
        int(clamp(x2 + pad, 0, W)),
        int(clamp(y2 + pad, 0, H)),
    )


def choose_cloud_color(mode: str, rng: random.Random) -> Tuple[Tuple[int, int, int, int], str]:
    if mode == "red":
        return rng.choice([
            (215, 0, 0, 235),
            (235, 20, 20, 225),
            (190, 0, 0, 240),
        ]), "red"
    if mode == "dark":
        return rng.choice([
            (30, 30, 30, 220),
            (55, 55, 55, 220),
            (80, 80, 80, 205),
        ]), "dark"
    if rng.random() < 0.63:
        return choose_cloud_color("red", rng)
    return choose_cloud_color("dark", rng)


def arc_points_top(x0: float, x1: float, y: float, samples: int) -> List[Tuple[float, float]]:
    cx = (x0 + x1) / 2.0
    r = (x1 - x0) / 2.0
    return [(cx + r * math.cos(t), y - r * math.sin(t)) for t in linspace(math.pi, 0.0, samples)]


def arc_points_right(x: float, y0: float, y1: float, samples: int) -> List[Tuple[float, float]]:
    cy = (y0 + y1) / 2.0
    r = (y1 - y0) / 2.0
    return [(x + r * math.cos(t), cy + r * math.sin(t)) for t in linspace(-math.pi / 2.0, math.pi / 2.0, samples)]


def arc_points_bottom(x0: float, x1: float, y: float, samples: int) -> List[Tuple[float, float]]:
    # Идём справа налево.
    cx = (x0 + x1) / 2.0
    r = (x1 - x0) / 2.0
    return [(cx + r * math.cos(t), y + r * math.sin(t)) for t in linspace(0.0, math.pi, samples)]


def arc_points_left(x: float, y0: float, y1: float, samples: int) -> List[Tuple[float, float]]:
    # Идём снизу вверх.
    cy = (y0 + y1) / 2.0
    r = (y1 - y0) / 2.0
    return [(x - r * math.cos(t), cy + r * math.sin(t)) for t in linspace(math.pi / 2.0, -math.pi / 2.0, samples)]


def linspace(start: float, stop: float, count: int) -> List[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def build_revision_cloud_path(
    bbox: Tuple[int, int, int, int],
    bump_radius: int,
) -> List[Tuple[float, float]]:
    """Строит путь из последовательности полуокружностей вокруг прямоугольника.

    bbox — внешний bbox всего облачка. Волны касаются границ bbox, а не выходят за него.
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    r = max(4, min(bump_radius, int(min(bw, bh) * 0.25)))

    # Центровая прямоугольная рамка. Дуги выходят наружу и доходят до внешнего bbox.
    left = x1 + r
    right = x2 - r
    top = y1 + r
    bottom = y2 - r

    if right <= left or bottom <= top:
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

    def side_segments(length: float) -> int:
        approx_segment = max(2 * r, 8)
        return max(2, int(round(length / approx_segment)))

    top_n = side_segments(right - left)
    right_n = side_segments(bottom - top)
    bottom_n = top_n
    left_n = right_n

    # Подгоняем сегменты под длину стороны, чтобы все дуги ровно замыкались.
    points: List[Tuple[float, float]] = []
    samples_per_arc = 9

    # top: left -> right
    for i in range(top_n):
        sx0 = left + (right - left) * i / top_n
        sx1 = left + (right - left) * (i + 1) / top_n
        arc = arc_points_top(sx0, sx1, top, samples_per_arc)
        if points:
            arc = arc[1:]
        points.extend(arc)

    # right: top -> bottom
    for i in range(right_n):
        sy0 = top + (bottom - top) * i / right_n
        sy1 = top + (bottom - top) * (i + 1) / right_n
        arc = arc_points_right(right, sy0, sy1, samples_per_arc)
        if points:
            arc = arc[1:]
        points.extend(arc)

    # bottom: right -> left
    for i in range(bottom_n):
        sx1 = right - (right - left) * i / bottom_n
        sx0 = right - (right - left) * (i + 1) / bottom_n
        arc = arc_points_bottom(sx0, sx1, bottom, samples_per_arc)
        if points:
            arc = arc[1:]
        points.extend(arc)

    # left: bottom -> top
    for i in range(left_n):
        sy1 = bottom - (bottom - top) * i / left_n
        sy0 = bottom - (bottom - top) * (i + 1) / left_n
        arc = arc_points_left(left, sy0, sy1, samples_per_arc)
        if points:
            arc = arc[1:]
        points.extend(arc)

    if points:
        points.append(points[0])
    return points


def draw_cloud(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    rng: random.Random,
    args: argparse.Namespace,
) -> Tuple[str, Tuple[int, int, int, int]]:
    """Рисует облачко и возвращает его цвет и итоговый bbox для разметки.

    По умолчанию используется стиль sample, приближённый к форме на чертеже-образце. Альтернативно можно включить style=wavy или style=revision.
    """
    color, color_name = choose_cloud_color(args.cloud_color, rng)
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    if args.cloud_style == "sample":
        # Стиль, приближённый к облачку на предоставленном образце: частые одинаковые полукруглые
        # дуги, тонкая линия, аккуратный прямоугольный контур без "рыхлой" синусоиды.
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        line_width = max(1, int(round(args.cloud_line_width * rng.uniform(0.78, 1.00))))
        target_d = max(10, int(round((args.cloud_bump_radius * 2) * rng.uniform(0.92, 1.06))))
        # Для примера с чертежа дуги мелкие и частые, поэтому при больших облачках увеличиваем число дуг.
        top_n = max(3, int(round(bw / max(target_d, 1))))
        side_n = max(2, int(round(bh / max(target_d, 1))))
        dx = bw / top_n
        dy = bh / side_n
        r_top = dx / 2.0
        r_side = dy / 2.0

        # Внешний bbox оставляем как опорный, дуги касаются его границ.
        left = x1 + r_side
        right = x2 - r_side
        top = y1 + r_top
        bottom = y2 - r_top

        # Верх/низ: плотные одинаковые полукружия.
        for i in range(top_n):
            sx0 = x1 + i * dx
            sx1 = x1 + (i + 1) * dx
            rr = (sx1 - sx0) / 2.0
            draw.arc((sx0, top - rr, sx1, top + rr), start=180, end=360, fill=color, width=line_width)
            draw.arc((sx0, bottom - rr, sx1, bottom + rr), start=0, end=180, fill=color, width=line_width)

        # Левая/правая стороны.
        for i in range(side_n):
            sy0 = y1 + i * dy
            sy1 = y1 + (i + 1) * dy
            rr = (sy1 - sy0) / 2.0
            draw.arc((left - rr, sy0, left + rr, sy1), start=90, end=270, fill=color, width=line_width)
            draw.arc((right - rr, sy0, right + rr, sy1), start=270, end=450, fill=color, width=line_width)

        # Микро-замыкание на углах.
        cr = min(r_top, r_side)
        draw.arc((left - cr, top - cr, left + cr, top + cr), 180, 270, fill=color, width=line_width)
        draw.arc((right - cr, top - cr, right + cr, top + cr), 270, 360, fill=color, width=line_width)
        draw.arc((right - cr, bottom - cr, right + cr, bottom + cr), 0, 90, fill=color, width=line_width)
        draw.arc((left - cr, bottom - cr, left + cr, bottom + cr), 90, 180, fill=color, width=line_width)

        paste_rgba(image, overlay)
        label_bbox = expand_bbox(bbox, image.size, max(args.bbox_padding, line_width + 3))
        return color_name, label_bbox

    if args.cloud_style == "revision":
        line_width = max(1, int(round(args.cloud_line_width * rng.uniform(0.85, 1.12))))
        base_radius = max(4, int(round(args.cloud_bump_radius * rng.uniform(0.90, 1.10))))

        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        base_radius = max(4, min(base_radius, int(min(bw, bh) * 0.28)))

        left = x1 + base_radius
        right = x2 - base_radius
        top = y1 + base_radius
        bottom = y2 - base_radius

        if right <= left or bottom <= top:
            draw.rectangle(bbox, outline=color, width=line_width)
            paste_rgba(image, overlay)
            label_bbox = expand_bbox(bbox, image.size, max(args.bbox_padding, line_width + 2))
            return color_name, label_bbox

        def side_segments(length: float) -> int:
            approx_segment = max(2 * base_radius, 8)
            return max(2, int(round(length / approx_segment)))

        top_n = side_segments(right - left)
        side_n = side_segments(bottom - top)

        for i in range(top_n):
            sx0 = left + (right - left) * i / top_n
            sx1 = left + (right - left) * (i + 1) / top_n
            r = (sx1 - sx0) / 2.0
            draw.arc((sx0, top - r, sx1, top + r), start=180, end=360, fill=color, width=line_width)
        for i in range(side_n):
            sy0 = top + (bottom - top) * i / side_n
            sy1 = top + (bottom - top) * (i + 1) / side_n
            r = (sy1 - sy0) / 2.0
            draw.arc((right - r, sy0, right + r, sy1), start=270, end=450, fill=color, width=line_width)
        for i in range(top_n):
            sx0 = left + (right - left) * i / top_n
            sx1 = left + (right - left) * (i + 1) / top_n
            r = (sx1 - sx0) / 2.0
            draw.arc((sx0, bottom - r, sx1, bottom + r), start=0, end=180, fill=color, width=line_width)
        for i in range(side_n):
            sy0 = top + (bottom - top) * i / side_n
            sy1 = top + (bottom - top) * (i + 1) / side_n
            r = (sy1 - sy0) / 2.0
            draw.arc((left - r, sy0, left + r, sy1), start=90, end=270, fill=color, width=line_width)

        corner_r = base_radius
        draw.arc((left - corner_r, top - corner_r, left + corner_r, top + corner_r), 180, 270, fill=color, width=line_width)
        draw.arc((right - corner_r, top - corner_r, right + corner_r, top + corner_r), 270, 360, fill=color, width=line_width)
        draw.arc((right - corner_r, bottom - corner_r, right + corner_r, bottom + corner_r), 0, 90, fill=color, width=line_width)
        draw.arc((left - corner_r, bottom - corner_r, left + corner_r, bottom + corner_r), 90, 180, fill=color, width=line_width)

        paste_rgba(image, overlay)
        label_bbox = expand_bbox(bbox, image.size, max(args.bbox_padding, line_width + 2))
        return color_name, label_bbox

    # style == 'wavy'
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    line_width = max(2, int(round(args.cloud_line_width * rng.uniform(0.9, 1.35))))
    amp = max(5, min(bw, bh) * rng.uniform(0.055, 0.095))
    step = max(8, int((args.cloud_bump_radius * 2) * rng.uniform(0.85, 1.20)))
    inset = int(amp + line_width + 3)

    inner = (x1 + inset, y1 + inset, x2 - inset, y2 - inset)
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        inner = (x1 + line_width, y1 + line_width, x2 - line_width, y2 - line_width)

    points = wavy_rect_points(inner, amp=amp, step=step, rng=rng)
    points.append(points[0])
    draw.line(points, fill=color, width=line_width)

    if rng.random() < 0.22:
        offset = rng.choice([-1, 1])
        shifted = [(px + offset, py + offset) for px, py in points]
        draw.line(shifted, fill=(color[0], color[1], color[2], 90), width=max(1, line_width // 2))

    paste_rgba(image, overlay)
    extra_pad = max(args.bbox_padding, int(math.ceil(line_width / 2 + 2)))
    label_bbox = expand_bbox(bbox, image.size, extra_pad)
    return color_name, label_bbox

    def side_segments(length: float) -> int:
        approx_segment = max(2 * base_radius, 8)
        return max(2, int(round(length / approx_segment)))

    top_n = side_segments(right - left)
    side_n = side_segments(bottom - top)

    # Верхняя сторона: дуги наружу вверх.
    for i in range(top_n):
        sx0 = left + (right - left) * i / top_n
        sx1 = left + (right - left) * (i + 1) / top_n
        r = (sx1 - sx0) / 2.0
        draw.arc((sx0, top - r, sx1, top + r), start=180, end=360, fill=color, width=line_width)

    # Правая сторона: дуги наружу вправо.
    for i in range(side_n):
        sy0 = top + (bottom - top) * i / side_n
        sy1 = top + (bottom - top) * (i + 1) / side_n
        r = (sy1 - sy0) / 2.0
        draw.arc((right - r, sy0, right + r, sy1), start=270, end=450, fill=color, width=line_width)

    # Нижняя сторона: дуги наружу вниз.
    for i in range(top_n):
        sx0 = left + (right - left) * i / top_n
        sx1 = left + (right - left) * (i + 1) / top_n
        r = (sx1 - sx0) / 2.0
        draw.arc((sx0, bottom - r, sx1, bottom + r), start=0, end=180, fill=color, width=line_width)

    # Левая сторона: дуги наружу влево.
    for i in range(side_n):
        sy0 = top + (bottom - top) * i / side_n
        sy1 = top + (bottom - top) * (i + 1) / side_n
        r = (sy1 - sy0) / 2.0
        draw.arc((left - r, sy0, left + r, sy1), start=90, end=270, fill=color, width=line_width)

    # Небольшие дуги на углах закрывают возможные микроразрывы между сторонами.
    corner_r = base_radius
    draw.arc((left - corner_r, top - corner_r, left + corner_r, top + corner_r), 180, 270, fill=color, width=line_width)
    draw.arc((right - corner_r, top - corner_r, right + corner_r, top + corner_r), 270, 360, fill=color, width=line_width)
    draw.arc((right - corner_r, bottom - corner_r, right + corner_r, bottom + corner_r), 0, 90, fill=color, width=line_width)
    draw.arc((left - corner_r, bottom - corner_r, left + corner_r, bottom + corner_r), 90, 180, fill=color, width=line_width)

    paste_rgba(image, overlay)
    return color_name


def get_font(font_size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            if path and Path(path).exists():
                return ImageFont.truetype(path, font_size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=0)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def choose_text_color(mode: str, cloud_color_name: str, rng: random.Random) -> Tuple[int, int, int]:
    if mode == "red":
        return rng.choice([(185, 0, 0), (215, 0, 0), (230, 20, 20)])
    if mode == "dark":
        return rng.choice([(20, 20, 20), (45, 45, 45), (70, 70, 70)])
    # mixed: чаще делаем цвет как у облачка, но иногда чёрный текст рядом с красным облачком.
    if cloud_color_name == "red" and rng.random() < 0.70:
        return choose_text_color("red", cloud_color_name, rng)
    return choose_text_color("dark", cloud_color_name, rng)


def place_text_near_bbox(
    image_size: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    mode: str,
    text_wh: Tuple[int, int],
    rng: random.Random,
) -> Optional[Tuple[int, int]]:
    W, H = image_size
    x1, y1, x2, y2 = bbox
    tw, th = text_wh
    gap = rng.randint(5, 18)

    def clip_xy(x: float, y: float) -> Tuple[int, int]:
        return int(clamp(x, 0, max(0, W - tw))), int(clamp(y, 0, max(0, H - th)))

    if mode == "inside":
        # Размер шрифта НЕ уменьшаем. Если HOLD не помещается внутри маленького облачка, переносим рядом.
        if tw + 12 <= (x2 - x1) and th + 8 <= (y2 - y1):
            x = rng.uniform(x1 + 6, max(x1 + 6, x2 - tw - 6))
            y = rng.uniform(y1 + 4, max(y1 + 4, y2 - th - 4))
            return clip_xy(x, y)
        return place_text_near_bbox(image_size, bbox, "right", text_wh, rng)

    if mode == "right":
        x = x2 + gap
        y = (y1 + y2) / 2 - th / 2 + rng.randint(-8, 8)
        if x + tw <= W:
            return clip_xy(x, y)
        return place_text_near_bbox(image_size, bbox, "left", text_wh, rng)

    if mode == "left":
        x = x1 - gap - tw
        y = (y1 + y2) / 2 - th / 2 + rng.randint(-8, 8)
        if x >= 0:
            return clip_xy(x, y)
        return place_text_near_bbox(image_size, bbox, "right", text_wh, rng)

    if mode == "top":
        x = (x1 + x2) / 2 - tw / 2 + rng.randint(-14, 14)
        y = y1 - gap - th
        if y >= 0:
            return clip_xy(x, y)
        return place_text_near_bbox(image_size, bbox, "bottom", text_wh, rng)

    if mode == "bottom":
        x = (x1 + x2) / 2 - tw / 2 + rng.randint(-14, 14)
        y = y2 + gap
        if y + th <= H:
            return clip_xy(x, y)
        return place_text_near_bbox(image_size, bbox, "top", text_wh, rng)

    return None


def draw_hold_text(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    mode: str,
    rng: random.Random,
    args: argparse.Namespace,
    cloud_color_name: str,
    text: str = "HOLD",
) -> str:
    if mode == "absent":
        return mode

    font = get_font(args.hold_font_size, bold=False)
    tw, th = text_size(text, font)
    xy = place_text_near_bbox(image.size, bbox, mode, (tw, th), rng)
    if xy is None:
        return "absent"

    draw = ImageDraw.Draw(image)
    fill = choose_text_color(args.hold_color, cloud_color_name, rng)

    # Небольшая белая обводка помогает читать HOLD поверх линий чертежа.
    # Она не меняет bbox облачка, так как текст не является объектом разметки.
    draw.text(xy, text, font=font, fill=fill, stroke_width=1, stroke_fill=(255, 255, 255))
    return mode


def bbox_to_yolo(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    W, H = image_size
    x1, y1, x2, y2 = bbox
    x1 = clamp(x1, 0, W)
    y1 = clamp(y1, 0, H)
    x2 = clamp(x2, 0, W)
    y2 = clamp(y2, 0, H)

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return xc / W, yc / H, bw / W, bh / H


def write_yolo_label(label_path: Path, objects: Sequence[ObjectMeta]) -> None:
    lines = []
    for obj in objects:
        xc, yc, bw, bh = obj.bbox_yolo
        lines.append(f"{obj.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def apply_scan_like_distortions(image: Image.Image, rng: random.Random) -> Image.Image:
    # Сохраняем искажения умеренными: задача — сделать облачко похожим на PDF/скан, а не испортить чертёж.
    if rng.random() < 0.55:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.96, 1.04))
    if rng.random() < 0.55:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.96, 1.08))
    if rng.random() < 0.23:
        noise = Image.effect_noise(image.size, rng.uniform(3, 8)).convert("RGB")
        image = Image.blend(image, noise, rng.uniform(0.010, 0.028))
    if rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.10, 0.35)))
    return image.convert("RGB")


def draw_preview_bbox(image: Image.Image, objects: Sequence[ObjectMeta], extra_pad: int = 8) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for obj in objects:
        x1, y1, x2, y2 = expand_bbox(obj.bbox_px, preview.size, extra_pad)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 180, 0), width=4)
        draw.text((x1, max(0, y1 - 22)), f"class {obj.class_id}", fill=(0, 120, 0))
    return preview


def generate_one_sample(
    base_page: BasePage,
    args: argparse.Namespace,
    rng: random.Random,
    hold_modes: Sequence[str],
) -> Tuple[Image.Image, List[ObjectMeta]]:
    image = base_page.image.copy().convert("RGB")
    objects: List[ObjectMeta] = []
    existing_bboxes: List[Tuple[int, int, int, int]] = []

    cloud_count = rng.randint(args.min_clouds, args.max_clouds)
    for _ in range(cloud_count):
        bbox = random_cloud_bbox(image.size, rng, existing_bboxes, args)
        if bbox is None:
            continue

        cloud_color_name, label_bbox = draw_cloud(image, bbox, rng, args)
        requested_hold_mode = rng.choice(list(hold_modes))
        actual_hold_mode = draw_hold_text(image, bbox, requested_hold_mode, rng, args, cloud_color_name)

        should_label = not (args.label_only_with_hold and actual_hold_mode == "absent")
        if should_label:
            yolo_bbox = bbox_to_yolo(label_bbox, image.size)
            objects.append(
                ObjectMeta(
                    class_id=args.class_id,
                    bbox_px=label_bbox,
                    bbox_yolo=yolo_bbox,
                    hold_mode=actual_hold_mode,
                    text="HOLD" if actual_hold_mode != "absent" else "",
                    cloud_color=cloud_color_name,
                )
            )
        existing_bboxes.append(label_bbox)

    image = apply_scan_like_distortions(image, rng)
    return image, objects


def write_data_yaml(output_dir: Path, class_name: str) -> None:
    content = (
        f'path: "{output_dir.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    errors = validate_args(args)
    if errors:
        raise SystemExit("\n".join(f"Ошибка: {e}" for e in errors))

    rng = random.Random(args.seed)
    hold_modes = split_hold_modes(args.hold_modes)

    output_dir: Path = args.output_dir
    for split in ("train", "val"):
        safe_mkdir(output_dir / "images" / split)
        safe_mkdir(output_dir / "labels" / split)
    if args.preview:
        safe_mkdir(output_dir / "previews")
    if args.save_pdf:
        safe_mkdir(output_dir / "pdf")

    print(f"Загружаю исходные страницы из: {args.input_dir}")
    base_pages = load_base_pages(args.input_dir, dpi=args.dpi)
    print(f"Найдено базовых страниц/изображений: {len(base_pages)}")

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for sample_idx in range(args.samples):
            base_page = rng.choice(base_pages)
            split = choose_split(rng, args.val_ratio)
            image, objects = generate_one_sample(base_page, args, rng, hold_modes)

            stem_source = Path(base_page.source_file).stem.replace(" ", "_").replace("[", "").replace("]", "")
            page_part = f"p{base_page.page_index + 1:03d}" if base_page.page_index is not None else "img"
            sample_name = f"{sample_idx + 1:05d}_{stem_source}_{page_part}"

            image_path = output_dir / "images" / split / f"{sample_name}.jpg"
            label_path = output_dir / "labels" / split / f"{sample_name}.txt"

            quality = rng.randint(args.jpg_quality_min, args.jpg_quality_max)
            image.save(image_path, format="JPEG", quality=quality, optimize=True)
            write_yolo_label(label_path, objects)

            if args.preview:
                preview = draw_preview_bbox(image, objects, extra_pad=args.preview_bbox_padding)
                preview.save(output_dir / "previews" / f"{sample_name}_preview.jpg", format="JPEG", quality=92)

            if args.save_pdf:
                image.save(output_dir / "pdf" / f"{sample_name}.pdf", "PDF", resolution=args.dpi)

            sample_meta = SampleMeta(
                image_file=str(image_path.relative_to(output_dir)),
                label_file=str(label_path.relative_to(output_dir)),
                split=split,
                source_file=base_page.source_file,
                source_page_index=base_page.page_index,
                width=image.width,
                height=image.height,
                objects=objects,
            )
            meta_file.write(json.dumps(asdict(sample_meta), ensure_ascii=False) + "\n")

    write_data_yaml(output_dir, args.class_name)
    print(f"Готово: {args.samples} изображений сохранено в {output_dir}")
    print(f"YOLO config: {output_dir / 'data.yaml'}")
    print(f"Метаданные для проверки: {metadata_path}")


if __name__ == "__main__":
    main()
