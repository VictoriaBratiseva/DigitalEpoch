#!/usr/bin/env python3
"""Create quick visual previews for a YOLO detection dataset.

The script draws bounding boxes from YOLO .txt labels onto corresponding images.
It is useful for checking generated markup without opening labelImg.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO labels")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Dataset root with images/ and labels/")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for preview images")
    parser.add_argument("--split", default="all", choices=["train", "val", "all"], help="Dataset split to visualize")
    parser.add_argument("--line-width", type=int, default=4, help="BBox line width")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of previews, 0 means all")
    parser.add_argument("--draw-empty", action="store_true", help="Also save previews for images without labels")
    return parser.parse_args()


def find_images(dataset_dir: Path, split: str) -> List[Path]:
    splits = ["train", "val"] if split == "all" else [split]
    images: List[Path] = []
    for sp in splits:
        img_dir = dataset_dir / "images" / sp
        for ext in IMG_EXTS:
            images.extend(img_dir.glob(f"*{ext}"))
            images.extend(img_dir.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def label_path_for_image(dataset_dir: Path, image_path: Path) -> Path:
    # Replace /images/<split>/<name>.jpg with /labels/<split>/<name>.txt
    parts = image_path.parts
    idx = parts.index("images")
    split = parts[idx + 1]
    return dataset_dir / "labels" / split / f"{image_path.stem}.txt"


def yolo_to_xyxy(line: str, img_w: int, img_h: int) -> Tuple[int, int, int, int] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    _, cx, cy, bw, bh = parts
    cx, cy, bw, bh = map(float, [cx, cy, bw, bh])
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(args.dataset_dir, args.split)
    if args.max_images > 0:
        images = images[: args.max_images]

    saved = 0
    for image_path in images:
        label_path = label_path_for_image(args.dataset_dir, image_path)
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        label_lines = []
        if label_path.exists():
            label_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        if not label_lines and not args.draw_empty:
            continue

        for line in label_lines:
            bbox = yolo_to_xyxy(line, img_w, img_h)
            if bbox is not None:
                draw.rectangle(bbox, outline=(0, 180, 0), width=args.line_width)

        rel_name = f"{image_path.parent.name}_{image_path.stem}_labels.jpg"
        img.save(args.output_dir / rel_name, quality=92)
        saved += 1

    print(f"Found images: {len(images)}")
    print(f"Saved previews: {saved}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
