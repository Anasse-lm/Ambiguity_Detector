"""YAML loading and project-root resolution for config-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """
    Locate repository root by finding configs/train.yaml.

    Searches from `start` (default: cwd) upward, then falls back to a path
    relative to this module (src/req_ambiguity/utils -> project root).
    """
    candidates: list[Path] = []
    if start is not None:
        p = start.resolve()
        candidates.append(p)
        candidates.extend(p.parents)
    else:
        cwd = Path.cwd().resolve()
        candidates.append(cwd)
        candidates.extend(cwd.parents)

    marker = Path("configs") / "train.yaml"
    for base in candidates:
        if (base / marker).is_file():
            return base

    pkg_root = Path(__file__).resolve().parents[3]
    if (pkg_root / marker).is_file():
        return pkg_root

    raise FileNotFoundError(
        "Could not locate project root (missing configs/train.yaml). "
        "Run commands from the repository root or pass --project-root."
    )


def load_yaml(path: Path | str, *, root: Path | None = None) -> dict[str, Any]:
    """Load a YAML file. Relative paths resolve against `root` (default: project root)."""
    p = Path(path)
    if not p.is_absolute():
        base = root or find_project_root()
        p = (base / p).resolve()
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping at root of YAML, got {type(data)} from {p}")
    return dict(data)


def resolve_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a path; if relative, join to project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    base = root or find_project_root()
    return (base / p).resolve()
