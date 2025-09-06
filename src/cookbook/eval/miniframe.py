import re
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Generator, Iterable, NamedTuple

from rich.console import Console
from rich.table import Table


class Row(NamedTuple):
    name: str
    columns: list[str]
    values: list[float | None]

    def missing(self) -> list[str]:
        return [col for col, val in zip(self.columns, self.values) if val is None]


@dataclass
class MiniFrame:
    """A pandas-like lightweight table"""

    title: str

    # columns are keys, rows are values
    _data: OrderedDict[str, OrderedDict[str, float | None]] = field(default_factory=OrderedDict)

    def add(self, row: str, col: str, val: float | None, overwrite: bool = True):
        """Add a value to the table at the specified row and column.

        Args:
            row: The row name (a model name, e.g. "olmo2-microanneal-all-dressed-v20-10ddb0f7_step4769-hf")
            col: The column name (a task/metric identifier, e.g. "arc_challenge:mc::xlarge")
            val: The value for the model on the task/metric; None if the value is not available
            overwrite: If True, overwrite existing values. If False, only add if value is not set or None.
        """
        column = self._data.setdefault(col, OrderedDict())
        if overwrite or row not in column or column[row] is None:
            column[row] = val

    def add_many(self, *elements: tuple[str, str, float | None], overwrite: bool = True):
        """Add multiple values to the table at the specified row and column.

        Args:
            elements: A sequence of tuples, each containing a row name, a column name, and a value.
            overwrite: If True, overwrite existing values. If False, only add if value is not set or None.
        """
        for row, col, val in elements:
            self.add(row=row, col=col, val=val, overwrite=overwrite)

    def sort(
        self,
        by_col: str | None = None,
        by_avg: bool = False,
        by_name: bool = False,
        reverse: bool = False,
    ) -> "MiniFrame":
        """Sort the rows of the table based on specified criteria.

        Args:
            by_col: Sort rows by values in a specific column. Rows with None values
                are treated as having -inf value for sorting purposes.
            by_avg: Sort rows by the average of all their values across columns.
                None values are treated as 0 when computing averages.
            by_name: Sort rows alphabetically by row name.
            reverse: If True, sort in descending order. If False, sort in ascending order.

        Returns:
            A new MiniFrame with rows sorted according to the specified criteria.

        Raises:
            AssertionError: If multiple sorting criteria are provided simultaneously.
            ValueError: If no sorting criteria are provided.

        Note:
            Only one sorting criterion (by_col, by_avg, or by_name) can be specified at a time.
        """

        # model names to sort
        all_keys = {row for col in self._data for row in self._data[col]}

        key_fn: Callable[[str], float | str]

        if by_col:
            # sort by values in a column; make sure we don't provide both by_avg and by_name
            assert by_avg is False and by_name is False, "Cannot provide both by_col and by_avg or by_name"

            # make sure the column exists
            assert by_col in self._data, f"Column {by_col} not found"

            # we get values from the column; if the value is None, we use -inf as the key
            def sort_by_col_fn(row: str) -> float:
                return (self._data[by_col].get(row) or float("-inf"))
            key_fn = sort_by_col_fn

        elif by_avg:
            # sort by average of all values; make sure we don't provide both by_col and by_name
            assert by_col is None and by_name is False, "Cannot provide both by_avg and by_col or by_name"

            # we get the average of all values; if the value is None, we use 0 as the value
            def sort_by_avg_fn(row: str) -> float:
                return sum(self._data[col].get(row) or 0 for col in self._data) / len(self._data)
            key_fn = sort_by_avg_fn

        elif by_name:
            # we sort alphabetically by column name
            assert by_col is None and by_avg is False, "Cannot provide both by_name and by_col or by_avg"

            def sort_by_name_fn(col: str) -> str:
                return col
            key_fn = sort_by_name_fn

        else:
            # I don't recognize the sorting criteria
            raise ValueError("No sorting criteria provided")

        # actually sort the keys
        sorted_keys = sorted(all_keys, key=key_fn, reverse=reverse)

        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((row, col, self._data[col].get(row)) for col in self.columns for row in sorted_keys))
        return new_frame

    def _make_fn(self, vals: tuple[str | re.Pattern, ...], reverse: bool) -> Callable[[str], bool]:
        def fn(x: str, _vals: tuple[str | re.Pattern, ...], reverse: bool) -> bool:
            for val in _vals:
                if isinstance(val, str):
                    if x == val:
                        return not reverse
                elif isinstance(val, re.Pattern):
                    if val.search(x) is not None:
                        return not reverse
                else:
                    raise ValueError(f"Invalid column filter: {val}")
            return reverse

        return partial(fn, _vals=vals, reverse=reverse)

    def drop_cols(self, *col: str | re.Pattern) -> "MiniFrame":
        fn = self._make_fn(vals=col, reverse=True)
        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((r, c, self._data[c][r]) for c in self._data for r in self._data[c] if fn(c)))
        return new_frame

    def keep_cols(self, *col: str | re.Pattern) -> "MiniFrame":
        fn = self._make_fn(vals=col, reverse=False)
        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((r, c, self._data[c][r]) for c in self._data for r in self._data[c] if fn(c)))
        return new_frame

    def drop_rows(self, *row: str | re.Pattern) -> "MiniFrame":
        fn = self._make_fn(vals=row, reverse=True)
        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((r, c, self._data[c][r]) for c in self._data for r in self._data[c] if fn(r)))
        return new_frame

    def keep_rows(self, *row: str | re.Pattern) -> "MiniFrame":
        fn = self._make_fn(vals=row, reverse=False)
        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((r, c, self._data[c][r]) for c in self._data for r in self._data[c] if fn(r)))
        return new_frame

    @property
    def columns(self) -> Generator[str, None, None]:
        for col in self._data:
            yield col

    @property
    def rows(self) -> Generator[Row, None, None]:
        seen = set()
        columns = list(self.columns)
        for col in columns:
            for row in self._data[col]:
                if row in seen:
                    continue
                yield Row(name=row, columns=columns, values=[self._data[c].get(row, None) for c in columns])
                seen.add(row)

    def drop_empty(self) -> "MiniFrame":
        has_empty_column = {col for col in self._data if any(v is None for v in self._data[col].values())}
        return self.drop_cols(*has_empty_column)

    def show(self):
        console = Console()
        table = Table(title=self.title, min_width=len(self.title) + 4)

        table.add_column("")  # this is the column for the row name
        for col in self.columns:
            # we add column and center the text
            table.add_column(col, justify="center")

        for row in self.rows:
            formatted_values = [f"{v * 100:.2f}" if v is not None else "-" for v in row.values]
            table.add_row(row.name, *formatted_values)

        console.print(table)

    def to_json(self) -> str:
        return json.dumps(self._data, sort_keys=True)

    def to_csv(self) -> str:
        # Header row with just column names, sorted using the original table order
        # columns = sorted(self.columns)
        columns = list(self.columns)
        header = ",".join(["name"] + columns)
        # Data rows with formatted values
        rows = []
        for row in self.rows:
            formatted_values = [
                f"{v * 100:.2f}" if v is not None else "-"
                for v in [row.values[list(self.columns).index(col)] for col in columns]
            ]
            rows.append(",".join([row.name] + formatted_values))
        return "\n".join([header] + rows)

    def __getitem__(self, item: Iterable[str]) -> float | None:
        try:
            col, row = item
        except ValueError:
            raise ValueError(f"Invalid item: {item}")

        return self._data[col].get(row, None)

    def __setitem__(self, item: Iterable[str], val: float | None):
        try:
            col, row = item
        except ValueError:
            raise ValueError(f"Invalid item: {item}")
        return self.add(row=row, col=col, val=val)

    def __contains__(self, col: str) -> bool:
        return col in self._data

    def __len__(self) -> int:
        return len(self._data)

    def shape(self) -> tuple[int, int]:
        if len(self._data) == 0:
            return 0, 0
        return len(self._data), len(max(self._data.values(), key=len))

    def copy(self) -> "MiniFrame":
        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((row, col, val) for col in self._data for row, val in self._data[col].items()))
        return new_frame

    def __add__(self, other: "MiniFrame") -> "MiniFrame":
        new_frame = MiniFrame(title=self.title)
        # first add all columns from this frame
        new_frame.add_many(*((row, col, val) for col in self._data for row, val in self._data[col].items()))
        # then add all columns from the other frame
        new_frame.add_many(*((row, col, val) for col in other._data for row, val in other._data[col].items()))
        return new_frame

    def __radd__(self, other: "MiniFrame") -> "MiniFrame":
        return other + self
