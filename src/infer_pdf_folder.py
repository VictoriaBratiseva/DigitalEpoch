#!/usr/bin/env python3
"""Run trained YOLO model on a folder with PDF documents.

The script supports multi-page PDFs. Each page is rendered to an image, passed to
YOLO, and saved with detected cloud boxes. Visualization contains boxes only:
no class names and no confidence labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect HOLD/revision clouds in PDF folder")
    parser.add_argument("--pdf-dir", required=True, type=Path, help="Folder with PDF files")
    parser.add_argument("--model", required=True, type=Path, help="Path to trained best.pt")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for visualized results")
    parser.add_argument("--dpi", type=int, default=160, help="PDF render DPI")
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--line-width", type=int, default=4, help="Visualization box line width")
    parser.add_argument("--save-rendered-pages", action="store_true", help="Also save raw rendered pages")
    return parser.parse_args()


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def main() -> None:
    args = parse_args()
    if not args.pdf_dir.exists():
        raise SystemExit(f"PDF folder does not exist: {args.pdf_dir}")
    if not args.model.exists():
        raise SystemExit(f"Model file does not exist: {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = args.output_dir / "rendered_pages"
    visual_dir = args.output_dir / "visualized"
    visual_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rendered_pages:
        rendered_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    pdf_files = sorted(args.pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise SystemExit("No PDF files found")

    rows: List[Dict] = []

    for pdf_path in pdf_files:
        doc = fitz.open(pdf_path)
        for page_index in range(doc.page_count):
            page = doc[page_index]
            image = render_page(page, args.dpi)
            page_name = f"{pdf_path.stem}_p{page_index + 1:03d}"

            if args.save_rendered_pages:
                image.save(rendered_dir / f"{page_name}.jpg", quality=92)

            results = model.predict(
                source=image,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )
            result = results[0]

            # labels=False and conf=False are important: curator asked for clean visualization.
            plotted = result.plot(labels=False, conf=False, line_width=args.line_width)
            out_img = Image.fromarray(plotted[..., ::-1])  # BGR to RGB
            out_path = visual_dir / f"{page_name}_detected.jpg"
            out_img.save(out_path, quality=92)

            page_w_pt = float(page.rect.width)
            page_h_pt = float(page.rect.height)
            img_w, img_h = image.size

            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                conf = float(box.conf[0]) if box.conf is not None else None
                cls = int(box.cls[0]) if box.cls is not None else 0

                # Approximate conversion from rendered image pixels back to PDF points.
                pdf_x1 = x1 / img_w * page_w_pt
                pdf_y1 = y1 / img_h * page_h_pt
                pdf_x2 = x2 / img_w * page_w_pt
                pdf_y2 = y2 / img_h * page_h_pt

                rows.append({
                    "pdf": str(pdf_path),
                    "page": page_index + 1,
                    "class_id": cls,
                    "confidence": conf,
                    "image_x1": x1,
                    "image_y1": y1,
                    "image_x2": x2,
                    "image_y2": y2,
                    "pdf_x1_pt": pdf_x1,
                    "pdf_y1_pt": pdf_y1,
                    "pdf_x2_pt": pdf_x2,
                    "pdf_y2_pt": pdf_y2,
                    "visualization": str(out_path),
                })

        doc.close()

    csv_path = args.output_dir / "detections.csv"
    json_path = args.output_dir / "detections.json"
    fieldnames = list(rows[0].keys()) if rows else [
        "pdf", "page", "class_id", "confidence", "image_x1", "image_y1", "image_x2", "image_y2",
        "pdf_x1_pt", "pdf_y1_pt", "pdf_x2_pt", "pdf_y2_pt", "visualization"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed PDFs: {len(pdf_files)}")
    print(f"Detections: {len(rows)}")
    print(f"Visualizations: {visual_dir}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
