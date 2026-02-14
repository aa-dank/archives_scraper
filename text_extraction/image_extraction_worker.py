import argparse
import json
from pathlib import Path
import re
import sys
from typing import Optional

from PIL import Image, ImageOps, ImageSequence
import pytesseract


def _apply_memory_limit(mem_mb: Optional[int]) -> None:
    if mem_mb is None:
        return

    if not sys.platform.startswith("linux"):
        print("warning: --mem-mb is only supported on Linux; ignoring", file=sys.stderr)
        return

    try:
        import resource

        limit_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception as exc:
        print(f"warning: failed to set memory limit: {exc}", file=sys.stderr)


def _parse_config(config_json_arg: str) -> dict:
    config = json.loads(config_json_arg)
    if not isinstance(config, dict):
        raise ValueError("--config-json must decode to a JSON object")
    return config


def _config_str(*parts: str) -> str:
    return " ".join(part for part in parts if part)


def _load_images(path: Path, max_side: int) -> list[Image.Image]:
    images = []
    with Image.open(path) as im:
        try:
            for frame in ImageSequence.Iterator(im):
                images.append(frame.convert("RGB"))
        except Exception:
            images.append(im.convert("RGB"))

    out = []
    for image in images:
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
        out.append(image)
    return out


def _preprocess_light(pil_img: Image.Image) -> Image.Image:
    image = ImageOps.grayscale(pil_img)
    return image.point(lambda pixel: 255 if pixel > 200 else 0)


def _ensure_longside_bottom(pil_img: Image.Image) -> Image.Image:
    width, height = pil_img.size
    short_side, long_side = sorted((width, height))
    ratio = short_side / long_side
    letter_ratio = 8.5 / 11
    if abs(ratio - letter_ratio) > 0.05 and height > width:
        return pil_img.rotate(90, expand=True)
    return pil_img


def _detect_and_correct_orientation(pil_img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(pil_img)
    except pytesseract.TesseractError:
        return pil_img

    rotation_match = re.search(r"Rotate: (\d+)", osd)
    if rotation_match:
        angle = int(rotation_match.group(1))
        if angle != 0:
            pil_img = pil_img.rotate(360 - angle, expand=True)
    return pil_img


def _inject_dpi(pil_img: Image.Image, dpi: int) -> Image.Image:
    existing = pil_img.info.get("dpi", (0, 0))[0]
    if not existing:
        pil_img.info["dpi"] = (dpi, dpi)
    return pil_img


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

    images = _load_images(path=path, max_side=max_side)
    texts = []
    for image in images:
        image = _ensure_longside_bottom(image)
        image = _inject_dpi(image, default_image_dpi)
        image = _detect_and_correct_orientation(image)
        if preprocess:
            image = _preprocess_light(image)

        text = pytesseract.image_to_string(
            image=image,
            lang=lang,
            config=_config_str(f"--psm {psm}", f"--oem {oem}"),
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
        _apply_memory_limit(args.mem_mb)

        config = _parse_config(args.config_json)
        text = _extract_text(path=Path(args.input), config=config)
        print(text, end="")
        return 0
    except Exception as exc:
        print(f"image ocr worker failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
