import argparse
import json
import sys
from typing import Optional

import ocrmypdf


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


def _parse_params(params_json_arg: str) -> dict:
    if params_json_arg.startswith("@"):
        params_path = params_json_arg[1:]
        with open(params_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return json.loads(params_json_arg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCRmyPDF in an isolated subprocess")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", required=True, help="Path to output OCR PDF")
    parser.add_argument("--params-json", required=True, help="OCR params JSON string or @/path/to/file.json")
    parser.add_argument("--mem-mb", type=int, default=None, help="Optional memory cap in MB")

    args = parser.parse_args()

    try:
        ocr_params = _parse_params(args.params_json)
        if not isinstance(ocr_params, dict):
            raise ValueError("--params-json must decode to a JSON object")

        _apply_memory_limit(args.mem_mb)
        ocrmypdf.ocr(input_file=args.input, output_file=args.output, **ocr_params)
        return 0
    except Exception as exc:
        print(f"ocr worker failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())