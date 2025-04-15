from collections import OrderedDict
from dataclasses import field, dataclass
from functools import partial
import re
from typing import Generator, Callable, Iterable
from rich.table import Table
from rich.console import Console
from cookbook.constants import SHORT_NAMES


@dataclass
class MiniFrame:
    """A pandas-like lightweight table"""
    title: str

    # columns are keys, rows are values
    _data: OrderedDict[str, OrderedDict[str, float | None]] = field(default_factory=OrderedDict)

    def add(self, row: str, col: str, val: float | None):
        self._data.setdefault(col, OrderedDict())[row] = val

    def add_many(self, *elements: tuple[str, str, float | None]):
        for row, col, val in elements:
            self.add(row=row, col=col, val=val)

    def sort(self, col: str, reverse: bool = False) -> "MiniFrame":
        all_keys = {row for col in self._data for row in self._data[col]}
        sorted_keys = sorted(all_keys, key=lambda row: self._data[col].get(row) or float("-inf"), reverse=reverse)
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
    def rows(self) -> Generator[tuple[str, list[float | None]], None, None]:
        seen = set()
        for col in self._data:
            for row in self._data[col]:
                if row in seen:
                    continue
                yield row, [self._data[col].get(row, None) for col in self._data]
                seen.add(row)

    def drop_empty(self) -> "MiniFrame":
        has_empty_column = {col for col in self._data if any(v is None for v in self._data[col].values())}
        return self.drop_cols(*has_empty_column)

    def show(self):
        console = Console()
        table = Table(title=self.title)

        table.add_column("") # this is the column for the row name
        for col in self.columns:
            # we shorten the column name if it is in the SHORT_NAMES dict
            for pattern, replacement in SHORT_NAMES.items():
                col = re.sub(pattern, replacement, col)

            # we add column and center the text
            table.add_column(col, justify="center")

        for row, values in self.rows:
            formatted_values = [f"{v * 100:.2f}" if v else "-" for v in values]
            table.add_row(row, *formatted_values)

        console.print(table)

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

    def __add__(self, other: "MiniFrame") -> "MiniFrame":
        new_frame = MiniFrame(title=self.title)
        # first add all columns from this frame
        new_frame.add_many(*((row, col, val) for col in self._data for row, val in self._data[col].items()))
        # then add all columns from the other frame
        new_frame.add_many(*((row, col, val) for col in other._data for row, val in other._data[col].items()))
        return new_frame

    def __radd__(self, other: "MiniFrame") -> "MiniFrame":
        return other + self
