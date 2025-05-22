import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Optional, TypeAlias, TypeVar, Union
from urllib.parse import urlparse

from typing_extensions import Self

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
    prot: Literal["gs", "s3", "weka", "file"]
    path: str

    @classmethod
    def weka_path(cls, path: str | Path) -> Optional["Self"]:
        parsed = urlparse(str(path))
        termination = "/" if str(path).endswith("/") else ""

        if parsed.scheme == "weka":
            if parsed.netloc not in WEKA_MOUNTS:
                raise ValueError(f"Invalid Weka bucket: {parsed.netloc}")
            return cls(prot="weka", path=f"/{parsed.netloc.strip('/')}/{parsed.path.lstrip('/')}{termination}")

        # the first part is usually '/'
        _, *parts = Path(path).parts

        if parts[0] in WEKA_MOUNTS:
            return cls(prot="weka", path="/" + "/".join(parts).strip("/") + termination)
        elif parts[0] == "weka" and parts[1] in WEKA_MOUNTS:
            return cls(prot="weka", path="/" + "/".join(parts[1:]).strip("/") + termination)

        return None

    @classmethod
    def local_path(cls, path: str | Path) -> Optional["Self"]:
        parsed = urlparse(str(path))
        termination = "/" if str(path).endswith("/") else ""
        if parsed.scheme == "file":
            return cls(prot="file", path=f"/{parsed.netloc.strip('/')}/{parsed.path.lstrip('/')}{termination}")

        if parsed.scheme == "":
            return cls(prot="file", path=str(path))

        return None

    @classmethod
    def s3_path(cls, path: str | Path) -> Optional["Self"]:
        parsed = urlparse(str(path))
        if parsed.scheme.startswith("s3"):
            return cls(prot="s3", path=parsed.netloc.strip("/") + "/" + parsed.path.lstrip("/"))
        return None

    @classmethod
    def gcs_path(cls, path: str | Path) -> Optional["Self"]:
        parsed = urlparse(str(path))
        if parsed.scheme in ("gs", "gcs"):
            return cls(prot="gs", path=parsed.netloc.strip("/") + "/" + parsed.path.lstrip("/"))
        return None

    @classmethod
    def from_str(cls, path: str | Path) -> "Self":
        if p := cls.weka_path(path):
            return p
        elif p := cls.local_path(path):
            return p
        elif p := cls.s3_path(path):
            return p
        elif p := cls.gcs_path(path):
            return p
        raise ValueError(f"Invalid path: {path}")

    @property
    def local(self) -> Path:
        if self.prot in ("weka", "file"):
            return Path(self.path)

        raise ValueError(f"Path is not local: {self.path}")

    @property
    def remote(self) -> str:
        if self.prot == "file":
            raise ValueError(f"Path is not remote: {self.path}")
        return f"{self.prot}://{self.path.lstrip('/')}"

    @property
    def bucket(self) -> str:
        remote = self.remote
        url = urlparse(remote)
        return url.netloc

    @property
    def prefix(self) -> str:
        remote = self.remote
        url = urlparse(remote)
        return url.path.lstrip("/")
