#!/usr/bin/env python3
"""
Script to compare continuation fields between two JSONL files.
Prints lines where the continuation differs between the files.
"""

import json
import argparse
from pathlib import Path
import re
from typing import List, Dict, Any, Optional


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of parsed JSON objects."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON on line {line_num} in {filepath}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    return data


def get_continuation(record: Dict[str, Any]) -> Optional[str]:
    """Extract continuation from model_output field."""
    try:
        model_output = record.get('model_output', [])
        if model_output and len(model_output) > 0:
            return model_output[0].get('continuation')
    except (KeyError, IndexError, TypeError):
        pass
    return None


def get_f1(record: Dict[str, Any]) -> Optional[float]:
    """Extract f1 score from metrics field."""
    try:
        metrics = record.get('metrics', {})
        return metrics.get('f1')
    except (KeyError, TypeError):
        pass
    return None


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and replacing whitespace sequences with single space."""
    if text is None:
        return None
    # Lowercase and replace all whitespace sequences with single space
    normalized = re.sub(r'\s+', ' ', text.lower())
    return normalized.strip()  # Remove leading/trailing whitespace


def compare_continuations(
    data1: List[Dict],
    data2: List[Dict],
    label1: str = "File 1",
    label2: str = "File 2",
    input_data: List[Dict] | None = None,
    verbose: bool = False,
    ignore_equal_f1: bool = False,
    norm_before_compare: bool = False
) -> None:
    """Compare continuations by line number."""
    max_len = max(len(data1), len(data2))
    differences_found = 0
    skipped_equal_f1 = 0
    skipped_normalized = 0

    # Track F1 scores for printed differences
    printed_f1_scores_1 = []
    printed_f1_scores_2 = []

    print("Comparing continuations by line number...")
    if norm_before_compare:
        print("(Normalizing: lowercase + single spaces)")
    print("=" * 60)

    for i in range(max_len):
        record1 = data1[i] if i < len(data1) else None
        record2 = data2[i] if i < len(data2) else None

        record1_doc_id = record1.get('doc_id') if record1 else None
        record2_doc_id = record2.get('doc_id') if record2 else None

        cont1 = get_continuation(record1) if record1 else None
        cont2 = get_continuation(record2) if record2 else None

        f1_1 = get_f1(record1) if record1 else None
        f1_2 = get_f1(record2) if record2 else None

        # Skip if ignore_equal_f1 is enabled and F1 scores are equal
        if ignore_equal_f1 and f1_1 == f1_2:
            skipped_equal_f1 += 1
            continue

        # Compare original continuations
        original_different = cont1 != cont2

        # actually not supporting unaligned records
        assert cont1 is not None and cont2 is not None
        assert f1_1 is not None and f1_2 is not None
        assert record1_doc_id is not None and record2_doc_id is not None and record1_doc_id == record2_doc_id, \
            "doc_id mismatch between records"

        # If normalizing, check if they're the same after normalization
        if norm_before_compare and original_different:
            norm_cont1 = normalize_text(cont1)
            norm_cont2 = normalize_text(cont2)
            if norm_cont1 == norm_cont2:
                skipped_normalized += 1
                continue

        if input_data:
            query = input_data[i].get('doc', {}).get('query')
            label = input_data[i].get('label')
            doc_id = input_data[i].get('doc_id')
            assert query is not None, "input file provided but query is missing"
            assert label is not None, "input file provided but label is missing"
            assert doc_id is not None, "input file provided but doc_id is missing"
            assert doc_id == record1_doc_id, "doc_id mismatch between input file and records"
        else:
            query = label = doc_id = None

        label1_padding = (" " * d if (d := (len(label2) - len(label1))) > 0 else "")
        label2_padding = (" " * d if (d := (len(label1) - len(label2))) > 0 else "")
        query_padding = (" " * d if (d := (max(len(label2), len(label1)) - len("Query:"))) > 0 else "")

        if original_different:
            differences_found += 1
            print(f"Doc ID: {doc_id}")

            if query is not None and label is not None:
                print(f"    Query:{query_padding}      {repr(query)}")
                print(f"    Gold label:{query_padding} {repr(label)}")
                print()

            print(f"    {label1}:{label1_padding}     {repr(cont1)}")
            print(f"    {label2}:{label2_padding}     {repr(cont2)}")
            print(f"    {label1} F1:{label1_padding}  {float(f1_1) * 100:6.2f}")
            print(f"    {label2} F1:{label2_padding}  {float(f1_2) * 100:6.2f}")

            # Collect F1 scores for average calculation
            if f1_1 is not None:
                printed_f1_scores_1.append(f1_1)
            if f1_2 is not None:
                printed_f1_scores_2.append(f1_2)

            if verbose and record1 and record2:
                print(f"    {label1} ID:{label1_padding}  {record1.get('doc_id', 'N/A')}")
                print(f"    {label2} ID:{label2_padding}  {record2.get('doc_id', 'N/A')}")
            print("" if input_data is None else "\n")

    print(f"Total differences found: {differences_found}")
    if ignore_equal_f1:
        print(f"Lines skipped due to equal F1 scores: {skipped_equal_f1}")
    if norm_before_compare:
        print(f"Lines skipped due to normalization (case/whitespace only): {skipped_normalized}")

    # Calculate and display average F1 scores for printed differences
    if printed_f1_scores_1:
        avg_f1_1 = sum(printed_f1_scores_1) / len(printed_f1_scores_1)
        print(f"Average F1 score ({label1}, printed differences): {avg_f1_1:.4f}")
    else:
        print(f"Average F1 score ({label1}, printed differences): N/A")

    if printed_f1_scores_2:
        avg_f1_2 = sum(printed_f1_scores_2) / len(printed_f1_scores_2)
        print(f"Average F1 score ({label2}, printed differences): {avg_f1_2:.4f}")
    else:
        print(f"Average F1 score ({label2}, printed differences): N/A")


def main():
    parser = argparse.ArgumentParser(
        description="Compare continuation fields between two JSONL files by line number"
    )
    parser.add_argument("file1", help="First JSONL file")
    parser.add_argument("file2", help="Second JSONL file")
    parser.add_argument(
        "--ignore-equal-f1",
        action="store_true",
        help="Skip lines where F1 scores are equal between both files"
    )
    parser.add_argument(
        "--norm-before-compare",
        action="store_true",
        help="Normalize text before comparison (lowercase + single spaces)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show additional information like doc_id"
    )
    parser.add_argument(
        "--input-file",
        default=None,
        type=Path,
        help="Input file with query and labels"
    )

    args = parser.parse_args()

    # Load both files
    print(f"Loading {args.file1}...")
    data1 = load_jsonl(args.file1)
    print(f"Loaded {len(data1)} records from {args.file1}")

    print(f"Loading {args.file2}...")
    data2 = load_jsonl(args.file2)
    print(f"Loaded {len(data2)} records from {args.file2}")

    if args.input_file:
        input_data = load_jsonl(args.input_file)
        print(f"Loaded {len(input_data)} records from {args.input_file}")
    else:
        input_data = None

    if not data1 or not data2:
        print("Error: Could not load data from one or both files.")
        return

    print()

    label1 = args.file1.split("/")[-1].split(".")[0]
    label2 = args.file2.split("/")[-1].split(".")[0]

    # Compare continuations
    compare_continuations(
        data1=data1,
        data2=data2,
        label1=label1,
        label2=label2,
        input_data=input_data,
        verbose=args.verbose,
        ignore_equal_f1=args.ignore_equal_f1,
        norm_before_compare=args.norm_before_compare
    )


if __name__ == "__main__":
    main()
