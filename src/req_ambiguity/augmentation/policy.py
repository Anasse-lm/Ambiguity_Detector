"""Load thesis-style placeholders.yaml into label -> token lists."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from req_ambiguity.utils.config import load_yaml, resolve_path


_METADATA_KEYS = frozenset({"vocabulary_version", "total_placeholders", "naming_convention"})


def load_placeholder_policy(
    placeholders_yaml: Path | str,
    label_cols: list[str],
    *,
    project_root: Path | None = None,
) -> tuple[dict[str, list[str]], frozenset[str]]:
    """
    Parse configs/placeholders.yaml (ambiguity sections with `- name: <TBD_...>`).

    Returns:
        label_to_tokens: each label maps to distinct placeholder tokens (order preserved).
        legal_tokens: flat set of all allowed tokens for compliance checks.
    """
    raw = load_yaml(placeholders_yaml, root=project_root)
    label_to_tokens: dict[str, list[str]] = {c: [] for c in label_cols}

    for key, value in raw.items():
        if key in _METADATA_KEYS:
            continue
        if key not in label_cols:
            continue
        if not isinstance(value, list):
            raise ValueError(f"Expected list under {key!r} in placeholders YAML")
        tokens: list[str] = []
        seen_label: set[str] = set()
        for item in value:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.startswith("<TBD_"):
                continue
            if name in seen_label:
                continue
            seen_label.add(name)
            tokens.append(name)
        label_to_tokens[key] = tokens

    missing = [c for c in label_cols if not label_to_tokens[c]]
    if missing:
        raise ValueError(
            "No placeholders defined in YAML for label(s): "
            + ", ".join(missing)
            + ". Check configs/placeholders.yaml sections match label_cols."
        )

    legal = frozenset(tok for toks in label_to_tokens.values() for tok in toks)
    return label_to_tokens, legal


def resolve_placeholders_path(cfg: Mapping[str, Any], *, project_root: Path) -> Path:
    """Resolve placeholders file path from train config."""
    refs = cfg.get("config_refs") or {}
    rel = refs.get("placeholders")
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("train.yaml must set config_refs.placeholders")
    return resolve_path(rel, root=project_root)
