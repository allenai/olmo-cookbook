# Test the extract_contaminated_items.py script

You can test the script with:

```bash
# Show help
python3 scripts/extract_contaminated_items.py --help

# Extract contaminated items and save to a file
python3 scripts/extract_contaminated_items.py ~/Downloads/ai2-decon-reports/ian-variants-with-oe-eval/7-24/ contaminated_items.txt --stats
```

## Expected Output Format

The script will output tuples in the format:
```
(10, ["winogrande:mc"])
(42, ["gsm8k:cot"])
(20, ["arc_challenge:mc", "arc_easy:mc"])
```

## Features

- Recursively finds all `.jsonl` files in the directory tree
- Parses each JSON line to extract `oe-eval-doc-id` and `oe-eval-task`
- Skips lines where `oe-eval-task` is null
- Groups multiple task names by document ID
- Outputs sorted results for consistent output
- Provides statistics when `--stats` flag is used
- Outputs the list of tuples to a specified file
