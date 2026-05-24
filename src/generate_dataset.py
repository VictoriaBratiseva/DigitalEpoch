#!/usr/bin/env python3
"""Generate synthetic YOLO dataset for HOLD/revision-cloud detection.

Input: folder with clean PDF documents or images.
Output: YOLO dataset structure with images, labels, data.yaml, metadata.jsonl and previews.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont, ImageFilter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}
CLASS_ID = 0
CLASS_NAME = "hold_cloud"


@dataclass
class CloudAnnotation:
    class_id: int
    bbox_px: Tuple[int, int, int, int]
    hold_mode: str
    color: Tuple[int, int, int]
    line_width: int
    bump_radius: float
    wave_jitter: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic HOLD cloud YOLO dataset")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder with clean PDFs/images")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output dataset folder")
    parser.add_argument("--samples", type=int, default=300, help="Number of generated images")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--min-clouds", type=int, default=0, help="Minimum clouds per image")
    parser.add_argument("--max-clouds", type=int, default=5, help="Maximum clouds per image")
    parser.add_argument("--dpi", type=int, default=160, help="PDF render DPI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hold-font-size", type=int, default=28, help="Fixed HOLD font size in pixels")
    parser.add_argument("--hold-modes", default="inside,right,left,top,bottom,absent", help="Comma-separated: inside,right,left,top,bottom,absent")
    parser.add_argument("--cloud-color", default="mixed", choices=["mixed", "red", "dark"], help="Cloud line color mode")
    parser.add_argument("--preview", action="store_true", help="Save label preview images with bbox")
    parser.add_argument("--label-only-with-hold", action="store_true", help="Do not label clouds without HOLD")
    parser.add_argument("--clean-output", action="store_true", help="Delete output folder before generation")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="Output JPEG quality")
    return parser.parse_args()


def collect_sources(input_dir: Path) -> List[Path]:
    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS | PDF_EXTS:
            files.append(p)
    return sorted(files)


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def load_random_background(sources: List[Path], dpi: int) -> Tuple[Image.Image, dict]:
    src = random.choice(sources)
    if src.suffix.lower() in PDF_EXTS:
        doc = fitz.open(src)
        page_count = doc.page_count
        page_index = random.randrange(page_count)
        doc.close()
        img = render_pdf_page(src, page_index, dpi)
        meta = {"source": str(src), "page_index": page_index + 1, "type": "pdf"}
    else:
        img = Image.open(src).convert("RGB")
        meta = {"source": str(src), "page_index": None, "type": "image"}
    return img, meta


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def choose_cloud_color(mode: str) -> Tuple[int, int, int]:
    if mode == "red":
        return random.choice([(190, 25, 25), (210, 0, 0), (165, 35, 35)])
    if mode == "dark":
        return random.choice([(0, 0, 0), (35, 35, 35), (70, 70, 70)])
    return random.choice([(190, 25, 25), (210, 0, 0), (0, 0, 0), (45, 45, 45)])


def wavy_rect_points(x1: int, y1: int, x2: int, y2: int, bump: float, jitter: float) -> List[Tuple[float, float]]:
    """Create a revision-cloud-like wavy rectangle polyline.

    Each cloud gets its own bump radius and jitter, so several clouds on one
    image can have visibly different arc sizes/frequencies.
    """
    points: List[Tuple[float, float]] = []

    def add_edge(start, end, normal):
        sx, sy = start
        ex, ey = end
        nx, ny = normal
        length = math.hypot(ex - sx, ey - sy)
        steps = max(5, int(length / max(6, bump * random.uniform(0.75, 1.35))))
        for i in range(steps + 1):
            t = i / steps
            bx = sx + (ex - sx) * t
            by = sy + (ey - sy) * t
            # semi-regular scallops; random phase and amplitude imitate hand/CAD differences
            phase = 2 * math.pi * t * steps
            amp = bump * random.uniform(0.45, 0.95)
            wave = abs(math.sin(phase)) * amp
            jx = random.uniform(-jitter, jitter)
            jy = random.uniform(-jitter, jitter)
            points.append((bx + nx * wave + jx, by + ny * wave + jy))

    add_edge((x1, y1), (x2, y1), (0, -1))
    add_edge((x2, y1), (x2, y2), (1, 0))
    add_edge((x2, y2), (x1, y2), (0, 1))
    add_edge((x1, y2), (x1, y1), (-1, 0))
    points.append(points[0])
    return points


def draw_revision_cloud(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], color, line_width: int, bump: float, jitter: float):
    x1, y1, x2, y2 = bbox
    pts = wavy_rect_points(x1, y1, x2, y2, bump=bump, jitter=jitter)
    draw.line(pts, fill=color, width=line_width, joint="curve")


def random_bbox(img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    # Engineering revision clouds are often elongated, but not always.
    w = random.randint(max(90, img_w // 18), max(140, img_w // 5))
    h = random.randint(max(45, img_h // 35), max(80, img_h // 9))
    if random.random() < 0.7:
        w = int(w * random.uniform(1.2, 2.2))
        h = int(h * random.uniform(0.65, 1.05))
    w = min(w, img_w - 30)
    h = min(h, img_h - 30)
    x1 = random.randint(15, max(16, img_w - w - 15))
    y1 = random.randint(15, max(16, img_h - h - 15))
    return x1, y1, x1 + w, y1 + h


def yolo_line(bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def place_hold_text(draw: ImageDraw.ImageDraw, bbox, img_size, mode: str, font, color):
    if mode == "absent":
        return
    img_w, img_h = img_size
    x1, y1, x2, y2 = bbox
    tw, th = text_size(draw, "HOLD", font)
    pad = 8
    if mode == "inside":
        x = x1 + max(3, (x2 - x1 - tw) // 2) + random.randint(-8, 8)
        y = y1 + max(3, (y2 - y1 - th) // 2) + random.randint(-6, 6)
    elif mode == "right":
        x = x2 + pad
        y = y1 + random.randint(0, max(1, y2 - y1 - th))
    elif mode == "left":
        x = x1 - tw - pad
        y = y1 + random.randint(0, max(1, y2 - y1 - th))
    elif mode == "top":
        x = x1 + random.randint(0, max(1, x2 - x1 - tw))
        y = y1 - th - pad
    elif mode == "bottom":
        x = x1 + random.randint(0, max(1, x2 - x1 - tw))
        y = y2 + pad
    else:
        return
    # Keep text within image boundaries where possible.
    x = max(2, min(img_w - tw - 2, x))
    y = max(2, min(img_h - th - 2, y))
    draw.text((x, y), "HOLD", font=font, fill=color)


def add_noise(img: Image.Image) -> Image.Image:
    # Mild blur/noise imitation; intentionally conservative for engineering drawings.
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.45)))
    return img


def draw_preview(img: Image.Image, anns: List[CloudAnnotation], out_path: Path):
    preview = img.copy()
    d = ImageDraw.Draw(preview)
    for ann in anns:
        d.rectangle(ann.bbox_px, outline=(0, 180, 0), width=max(3, ann.line_width))
    preview.save(out_path, quality=92)


def prepare_dirs(out: Path, clean: bool):
    if clean and out.exists():
        shutil.rmtree(out)
    for rel in ["images/train", "images/val", "labels/train", "labels/val", "previews"]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {args.input_dir}")
    sources = collect_sources(args.input_dir)
    if not sources:
        raise SystemExit("No PDF/image files found in input folder")

    prepare_dirs(args.output_dir, args.clean_output)
    hold_modes = [m.strip() for m in args.hold_modes.split(",") if m.strip()]
    font = get_font(args.hold_font_size)

    metadata_path = args.output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for i in range(1, args.samples + 1):
            img, src_meta = load_random_background(sources, args.dpi)
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            img_w, img_h = img.size

            cloud_count = random.randint(args.min_clouds, args.max_clouds)
            anns: List[CloudAnnotation] = []
            label_lines: List[str] = []

            for _ in range(cloud_count):
                bbox = random_bbox(img_w, img_h)
                color = choose_cloud_color(args.cloud_color)
                line_width = random.randint(2, 5)
                # Key improvement: bump radius varies independently for every cloud.
                # Sometimes small clouds get large arcs and large clouds get fine arcs.
                min_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
                bump_radius = random.uniform(8, max(10, min(34, min_side * random.uniform(0.18, 0.55))))
                if random.random() < 0.25:
                    bump_radius *= random.uniform(1.25, 1.85)
                bump_radius = max(7, min(42, bump_radius))
                jitter = random.uniform(0.5, 3.0)
                hold_mode = random.choice(hold_modes)

                draw_revision_cloud(draw, bbox, color=color, line_width=line_width, bump=bump_radius, jitter=jitter)
                place_hold_text(draw, bbox, img.size, hold_mode, font, color)

                ann = CloudAnnotation(
                    class_id=CLASS_ID,
                    bbox_px=bbox,
                    hold_mode=hold_mode,
                    color=color,
                    line_width=line_width,
                    bump_radius=round(float(bump_radius), 2),
                    wave_jitter=round(float(jitter), 2),
                )
                anns.append(ann)
                if not (args.label_only_with_hold and hold_mode == "absent"):
                    label_lines.append(yolo_line(bbox, img_w, img_h))

            img = add_noise(img)
            split = "train" if random.random() < args.train_ratio else "val"
            base_name = f"{i:05d}"
            img_path = args.output_dir / "images" / split / f"{base_name}.jpg"
            label_path = args.output_dir / "labels" / split / f"{base_name}.txt"
            img.save(img_path, quality=args.jpeg_quality)
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            if args.preview:
                preview_path = args.output_dir / "previews" / f"{base_name}_preview.jpg"
                draw_preview(img, anns, preview_path)

            meta_f.write(json.dumps({
                "image": str(img_path),
                "label": str(label_path),
                "split": split,
                "source": src_meta,
                "clouds_total": cloud_count,
                "clouds_labeled": len(label_lines),
                "annotations": [asdict(a) for a in anns],
            }, ensure_ascii=False) + "\n")

    data_yaml = (
        f"path: {args.output_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )
    (args.output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"Dataset generated: {args.output_dir}")
    print(f"Images: {args.samples}")
    print(f"Class: {CLASS_ID} -> {CLASS_NAME}")


if __name__ == "__main__":
    main()
