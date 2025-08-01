# Filter Feature Documentation

The filtering feature allows you to recalculate results over a subset of item-level results instead of using the default aggregate metrics. This enables more detailed analysis by including or excluding specific document items from specific tasks.

## Usage

Use one of the following flags with the `olmo-cookbook-eval results` command:

- `--filter-include <path>`: Only include items matching the filter tuples
- `--filter-exclude <path>`: Exclude items matching the filter tuples  

These options are mutually exclusive.

## Filter File Format

The filter file should contain tuples in the format:
```
(doc_id, ["task_name1", "task_name2"])
```

Where:
- `doc_id` is an integer document ID
- Task names are strings in a list

### Example filter file:

```
(0, ["arc_challenge:mc"])
(1, ["arc_challenge:mc"])
(5, ["arc_challenge:mc", "hellaswag:mc"])
```

This would filter documents with IDs 0, 1, and 5 for the specified tasks.

## Output

When using the default table format, the command will:

1. Display the filtered results table
2. Show a filter report with counts of filtered vs total items for each task

Example filter report:
```
📊 Filter Report (included items):
--------------------------------------------------
  arc_challenge:mc: 3/1119 items (0.3%)
  hellaswag:mc: 1/10042 items (0.0%)
```

## Implementation Details

The feature:
- Fetches prediction-level data from the datalake 
- Applies filtering based on document IDs and task names
- Recalculates primary metrics as averages over filtered items
- Maintains compatibility with existing result display formats
- Supports all existing output formats (table, JSON, CSV)

## Example Commands

```bash
# Include only specific items
olmo-cookbook-eval results --dashboard my-dashboard --tasks "arc_challenge:mc" --filter-include my_filter.txt

# Exclude specific items  
olmo-cookbook-eval results --dashboard my-dashboard --tasks "arc_challenge:mc" --filter-exclude problematic_items.txt

# Use with multiple tasks and models
olmo-cookbook-eval results --dashboard my-dashboard --tasks "arc_challenge:mc" "hellaswag:mc" --models "model1" --filter-include subset.txt
```
