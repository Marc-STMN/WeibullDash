"""
Project-wide version resolution.

Priority:
1. ``WEIBULL_TOOL_VERSION`` environment variable
2. nearest Git tag via ``git describe``
3. static fallback when Git metadata is unavailable
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_FALLBACK_VERSION = "0+unknown"
_TAG_PATTERN = re.compile(
    r"^(?:v)?(?P<base>\d+\.\d+\.\d+)(?:-(?P<distance>\d+)-g(?P<sha>[0-9a-f]+))?(?P<dirty>-dirty)?$"
)


def _normalize_git_describe(raw_value: str) -> str | None:
    value = (raw_value or "").strip()
    if not value:
        return None

    match = _TAG_PATTERN.fullmatch(value)
    if match:
        base = match.group("base")
        distance = match.group("distance")
        sha = match.group("sha")
        dirty = match.group("dirty")

        if distance and sha:
            version = f"{base}+{distance}.g{sha}"
        else:
            version = base

        if dirty:
            version = f"{version}.dirty"
        return version

    short_hash = re.fullmatch(r"(?P<sha>[0-9a-f]{7,})(?P<dirty>-dirty)?", value)
    if short_hash:
        version = f"0+g{short_hash.group('sha')}"
        if short_hash.group("dirty"):
            version = f"{version}.dirty"
        return version

    return value


def _version_from_git() -> str | None:
    repo_dir = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    return _normalize_git_describe(result.stdout)


def _resolve_version() -> str:
    env_version = os.getenv("WEIBULL_TOOL_VERSION", "").strip()
    if env_version:
        return env_version

    git_version = _version_from_git()
    if git_version:
        return git_version

    return _FALLBACK_VERSION


def get_version() -> str:
    return _resolve_version()


__version__ = _resolve_version()


__all__ = ["__version__", "_normalize_git_describe", "get_version"]
