#!/usr/bin/env python3
"""
Clean special tokens from reasoning data.

Rules (per team notes):
- Remove `<think>` and `</think>` tokens (keep inner content)
- Replace `<answer> ... </answer>` with `Answer: ...`
- Be a bit forgiving about whitespace adjacent to tags

Supports:
- Plain text (line-by-line)
- JSONL with a selected text field (default: `text`)
- gzip-compressed inputs/outputs by .gz extension

Usage examples:
1) Stream a single JSONL from S3 and write locally:
   aws s3 cp s3://bucket/path/documents/file.jsonl - | \
     python scripts/clean_special_tokens.py --mode jsonl --json-key text --input - --output cleaned.jsonl

2) Process a local directory recursively and mirror outputs:
   python scripts/clean_special_tokens.py \
     --input /data/verifiable/gpt-41/documents \
     --output /data/verifiable/gpt-41/cleaned-documents

3) Stream a gzip JSONL and write gzip JSONL:
   aws s3 cp s3://bucket/file.jsonl.gz - | \
     python scripts/clean_special_tokens.py --mode jsonl --json-key text --input - --output - \
     | gzip > cleaned.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, TextIO


OPEN_ANSWER_PATTERN = re.compile(r"\s*<answer>\s*", flags=re.IGNORECASE)
CLOSE_ANSWER_PATTERN = re.compile(r"\s*</answer>\s*", flags=re.IGNORECASE)
OPEN_THINK_PATTERN = re.compile(r"<think>", flags=re.IGNORECASE)
CLOSE_THINK_PATTERN = re.compile(r"</think>", flags=re.IGNORECASE)


def clean_text(raw_text: str, normalize_spaces: bool = False) -> str:
    """Apply cleaning rules to a single text string.

    - Remove <think> and </think>
    - Replace <answer> ... </answer> with 'Answer: ...'
    - Optionally normalize repeated spaces (not newlines)
    """
    if not raw_text:
        return raw_text

    text = raw_text

    # Remove think tags only (keep content)
    text = OPEN_THINK_PATTERN.sub("", text)
    text = CLOSE_THINK_PATTERN.sub("", text)

    # Replace answer tags with a readable prefix; trim immediate surrounding whitespace
    text = OPEN_ANSWER_PATTERN.sub(" Answer: ", text)
    text = CLOSE_ANSWER_PATTERN.sub("", text)

    if normalize_spaces:
        # Collapse runs of spaces/tabs to a single space, but keep newlines as-is
        # This is conservative: it won't touch newlines, only intra-line spacing
        text = re.sub(r"[ \t]{2,}", " ", text)

    # Trim edges but do not strip internal newlines
    return text.strip()


@dataclass
class Args:
    input: str
    output: str
    mode: str
    json_key: str
    fallback_keys: list[str]
    normalize_spaces: bool


def detect_mode_from_path(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
        return "jsonl"
    return "text"


def open_input_stream(path: str) -> TextIO:
    if path == "-":
        return sys.stdin
    if path.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, mode="rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def open_output_stream(path: str) -> TextIO:
    if path == "-":
        return sys.stdout
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, mode="wb"), encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def iter_files_recursive(input_path: str) -> Iterator[tuple[str, str]]:
    """Yield (in_path, rel_path) for all files under input_path.

    If input_path is a file, yields that file with rel_path being its basename.
    """
    p = Path(input_path)
    if p.is_file():
        yield (str(p), p.name)
        return
    for root, _dirs, files in os.walk(p):
        for fname in files:
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(p))
            yield (str(fpath), rel)


def process_text_stream(in_fp: TextIO, out_fp: TextIO, normalize_spaces: bool) -> None:
    for line in in_fp:
        cleaned = clean_text(line.rstrip("\n"), normalize_spaces=normalize_spaces)
        out_fp.write(cleaned + "\n")


def process_jsonl_stream(
    in_fp: TextIO,
    out_fp: TextIO,
    json_key: str,
    fallback_keys: list[str],
    normalize_spaces: bool,
) -> None:
    for raw_line in in_fp:
        line = raw_line.strip()
        if not line:
            out_fp.write("\n")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Not valid JSON on this line; treat as plain text
            cleaned = clean_text(line, normalize_spaces=normalize_spaces)
            out_fp.write(cleaned + "\n")
            continue

        key_to_use: Optional[str] = None
        if json_key and json_key in obj and isinstance(obj[json_key], str):
            key_to_use = json_key
        else:
            for k in fallback_keys:
                if k in obj and isinstance(obj[k], str):
                    key_to_use = k
                    break

        if key_to_use is None:
            # Nothing to clean; pass through
            out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
            continue

        obj[key_to_use] = clean_text(obj[key_to_use], normalize_spaces=normalize_spaces)
        out_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_single_file(in_path: str, out_path: str, mode: str, args: Args) -> None:
    actual_mode = mode if mode != "auto" else detect_mode_from_path(in_path)
    with open_input_stream(in_path) as in_fp, open_output_stream(out_path) as out_fp:
        if actual_mode == "jsonl":
            process_jsonl_stream(
                in_fp=in_fp,
                out_fp=out_fp,
                json_key=args.json_key,
                fallback_keys=args.fallback_keys,
                normalize_spaces=args.normalize_spaces,
            )
        else:
            process_text_stream(
                in_fp=in_fp,
                out_fp=out_fp,
                normalize_spaces=args.normalize_spaces,
            )


def run_directory(input_dir: str, output_dir: str, args: Args) -> None:
    for in_path, rel_path in iter_files_recursive(input_dir):
        out_path = str(Path(output_dir) / rel_path)
        mode = args.mode if args.mode != "auto" else detect_mode_from_path(in_path)
        process_single_file(in_path, out_path, mode, args)


def parse_args(argv: Optional[list[str]] = None) -> Args:
    parser = argparse.ArgumentParser(description="Remove <think> tags and convert <answer> blocks")
    parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory path (use '-' for stdin)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file or directory path (use '-' for stdout)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "jsonl", "text"],
        default="auto",
        help="How to parse inputs. Default auto by file extension",
    )
    parser.add_argument(
        "--json-key",
        default="text",
        help="Primary JSON key to clean when in jsonl mode (default: text)",
    )
    parser.add_argument(
        "--fallback-keys",
        default="content,document",
        help="Comma-separated fallback keys for jsonl mode if primary is absent",
    )
    parser.add_argument(
        "--normalize-spaces",
        action="store_true",
        help="Collapse runs of spaces/tabs within lines",
    )

    ns = parser.parse_args(argv)
    return Args(
        input=ns.input,
        output=ns.output,
        mode=ns.mode,
        json_key=ns.json_key,
        fallback_keys=[k.strip() for k in (ns.fallback_keys or "").split(",") if k.strip()],
        normalize_spaces=bool(ns.normalize_spaces),
    )


def main() -> None:
    args = parse_args()

    in_path = args.input
    out_path = args.output

    in_is_dir = (in_path != "-") and Path(in_path).is_dir()
    out_is_dir = (out_path != "-") and (out_path.endswith(os.sep) or Path(out_path).suffix == "" or Path(out_path).is_dir())

    if in_is_dir and not out_is_dir:
        raise SystemExit("When input is a directory, output must be a directory path.")

    if in_is_dir:
        run_directory(in_path, out_path, args)
        return

    # File/stream to file/stream
    process_single_file(in_path, out_path, args.mode, args)


if __name__ == "__main__":
    main()


