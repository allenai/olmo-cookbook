from collections import OrderedDict
from dataclasses import field, dataclass
import re
from typing import Generator, Callable
from rich.table import Table
from rich.console import Console


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
        sorted_keys = [
            k for k, _ in sorted(self._data[col].items(), key=lambda x: x[1] or float("-inf"), reverse=reverse)
        ]
        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((row, col, self._data[col][row]) for col in self._data for row in sorted_keys))
        return new_frame

    def _make_fn(self, val: str | re.Pattern | list[str], reverse: bool, strict: bool) -> Callable[[str], bool]:
        if isinstance(val, str):
            fn = lambda x: (x == val) if strict else (x in val)
        elif isinstance(val, re.Pattern):
            fn = lambda x: val.search(x) is not None
        elif isinstance(val, list):
            fn = lambda x: x in val if strict else any(c in x for c in val)     # pyright: ignore
        else:
            raise ValueError(f"Invalid column filter: {val}")

        fn_ = lambda x: not fn(x) if reverse else fn(x)
        return fn_

    def drop_cols(self, col: str | re.Pattern | list[str]) -> "MiniFrame":
        fn = self._make_fn(val=col, reverse=True, strict=False)
        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((r, c, self._data[c][r]) for c in self._data for r in self._data[c] if fn(c)))
        return new_frame

    def keep_cols(self, col: str | re.Pattern | list[str]) -> "MiniFrame":
        fn = self._make_fn(val=col, reverse=False, strict=False)
        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((r, c, self._data[c][r]) for c in self._data for r in self._data[c] if fn(c)))
        return new_frame

    def drop_rows(self, row: str | re.Pattern | list[str]) -> "MiniFrame":
        fn = self._make_fn(val=row, reverse=True, strict=False)

        new_frame = MiniFrame(title=self.title)
        new_frame.add_many(*((r, c, self._data[c][r]) for c in self._data for r in self._data[c] if fn(r)))
        return new_frame

    def keep_rows(self, row: str | re.Pattern | list[str]) -> "MiniFrame":
        fn = self._make_fn(val=row, reverse=False, strict=False)
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
        return self.drop_cols(list(has_empty_column))

    def show(self):
        console = Console()
        table = Table(title=self.title)

        table.add_column("")
        for col in self.columns:
            table.add_column(col, justify="center")

        for row, values in self.rows:
            formatted_values = [f"{v * 100:.2f}" if v else "-" for v in values]
            table.add_row(row, *formatted_values)

        console.print(table)

    def __contains__(self, col: str) -> bool:
        return col in self._data

    def __len__(self) -> int:
        return len(self._data)

    def shape(self) -> tuple[int, int]:
        if len(self._data) == 0:
            return 0, 0
        return len(self._data), len(max(self._data.values(), key=len))
