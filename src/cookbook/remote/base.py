from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Generic, TypeVar, Union, TypeAlias, Self, Literal
import json
from urllib.parse import urlparse

from cookbook.constants import WEKA_MOUNTS

JSON_VALID_TYPES: TypeAlias = Union[str, int, float, bool, list, dict]


C = TypeVar("C")


@dataclass(frozen=True)
class BaseAuthentication(Generic[C]):
    """Base class for all remote authentication classes."""

    @classmethod
    def from_dict(cls, obj: dict[str, JSON_VALID_TYPES]) -> "Self":
        """Convert a dictionary to a BaseAuthentication instance."""
        return cls(**obj)

    @classmethod
    def _check_dict_types(cls, obj: dict[str, JSON_VALID_TYPES]) -> None:
        """Check if the dictionary contains only valid types."""
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key type: {key!r} (expected str)")
            if not isinstance(value, JSON_VALID_TYPES):
                raise ValueError(f"Invalid value type: {value!r} (expected {JSON_VALID_TYPES})")
            if isinstance(value, dict):
                cls._check_dict_types(value)

    def to_dict(self) -> dict[str, JSON_VALID_TYPES]:
        """Convert a BaseAuthentication instance to a dictionary."""
        self._check_dict_types(obj := asdict(self))
        return obj

    @classmethod
    def from_json(cls, obj: str) -> "Self":
        """Convert a JSON string to a BaseAuthentication instance."""
        obj = json.loads(obj)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid JSON object: {obj}")
        return cls.from_dict(obj)


    def to_json(self) -> str:
        """Convert a BaseAuthentication instance to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def make(cls) -> "Self":
        """Create a new credentials instance to be used for remote operations."""
        raise NotImplementedError("Subclasses must implement this method")

    def apply(self, *args: Any, **kwargs: Any) -> C:
        """Apply the credentials so that it can be used for remote operations."""
        raise NotImplementedError("Subclasses must implement this method")



class AuthenticationError(RuntimeError):
    """Error raised when authentication fails."""
    ...


@dataclass(frozen=True)
class LocatedPath:
    prot: Literal["gs", "s3", "weka"]
    bucket: str
    prefix: str

    @classmethod
    def from_str(cls, path: str | Path) -> "Self":
        parsed_path = urlparse(str(path))
        if parsed_path.scheme.startswith("s3"):
            return cls(prot="s3", bucket=parsed_path.netloc, prefix=parsed_path.path.lstrip("/"))
        elif parsed_path.scheme in ("gs", "gcs"):
            return cls(prot="gs", bucket=parsed_path.netloc, prefix=parsed_path.path.lstrip("/"))
        elif parsed_path.scheme == "weka":
            return cls(prot="weka", bucket=parsed_path.netloc, prefix=parsed_path.path.lstrip("/"))
        elif parsed_path.scheme == "":
            _, *path_parts = Path(parsed_path.path).parts

            # maintain any trailing slashes
            suffix = "/" if str(path).endswith("/") else ""

            if path_parts[0] == "weka" and path_parts[1] in WEKA_MOUNTS:
                return cls(prot="weka", bucket=path_parts[1], prefix="/".join(path_parts[2:]) + suffix)
            elif path_parts[0] in WEKA_MOUNTS:
                return cls(prot="weka", bucket=path_parts[0], prefix="/".join(path_parts[1:]) + suffix)
            else:
                raise ValueError(f"Invalid path: {path}")




        # first_part, *rest_parts = Path(parsed_path.path).parts
        # if first_part in WEKA_MOUNTS:
        #     return cls(prot="weka", bucket=first_part, prefix="/".join(rest_parts))e
        # elif first_part == "weka" and len(rest_parts) >= 1 and rest_parts[0] in WEKA_MOUNTS:
        #     return cls(prot="weka", bucket=rest_parts[0], prefix="/".join(rest_parts[1:]))

        # print(first_part, rest_parts)

        raise ValueError(f"Invalid path: {path}")

    def to_str(self) -> str:
        """Convert a LocatedPath instance to a string."""
        return f"{self.prot}://{self.bucket}/{self.prefix}"

    @property
    def full(self) -> Path:
        return Path(f"/{self.bucket.strip('/')}/{self.prefix.lstrip('/')}")

    def __str__(self) -> str:
        return self.to_str()
