# text_extraction/image_extraction_worker.py

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import pytesseract

from .extraction_utils import ImageOCRUtils, OCRUtils, SubprocessUtils

def _parse_config(config_json_arg: str) -> dict:
    config = json.loads(config_json_arg)
    if not isinstance(config, dict):
        raise ValueError("--config-json must decode to a JSON object")
    return config


def _extract_text(path: Path, config: dict) -> str:
    tesseract_cmd = config.get("tesseract_cmd")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    lang = str(config.get("lang", "eng"))
    psm = int(config.get("psm", 3))
    oem = int(config.get("oem", 3))
    preprocess = bool(config.get("preprocess", True))
    max_side = int(config.get("max_side", 3000))
    default_image_dpi = int(config.get("default_image_dpi", 300))

    images = ImageOCRUtils.load_images(path=path, max_side=max_side)
    texts = []
    for image in images:
        image = ImageOCRUtils.ensure_longside_bottom(image)
        image = ImageOCRUtils.inject_dpi(image, default_image_dpi)
        image = ImageOCRUtils.detect_and_correct_orientation(image)
        if preprocess:
            image = ImageOCRUtils.preprocess_light(image)

        text = pytesseract.image_to_string(
            image=image,
            lang=lang,
            config=OCRUtils.config_str(f"--psm {psm}", f"--oem {oem}"),
        )
        texts.append(text)

    return "\n".join(texts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run image OCR in an isolated subprocess")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--config-json", required=True, help="OCR config JSON")
    parser.add_argument("--mem-mb", type=int, default=None, help="Optional memory cap in MB")

    args = parser.parse_args()

    try:
        SubprocessUtils.apply_memory_limit(args.mem_mb, stderr=sys.stderr)

        config = _parse_config(args.config_json)
        text = _extract_text(path=Path(args.input), config=config)
        print(text, end="")
        return 0
    except Exception as exc:
        print(f"image ocr worker failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
