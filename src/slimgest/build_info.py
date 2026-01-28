"""
Build metadata helpers.

CI creates `slimgest._build_info` at build time so the built wheel includes:
- build date (UTC)
- git short SHA
- git branch/ref
- resolved package version used for the build
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class BuildInfo:
    version: str
    build_date_utc: str
    build_timestamp_utc: str
    git_sha: str
    git_branch: str
    git_ref: str


def get_build_info() -> BuildInfo:
    """
    Return build metadata.

    In CI-built wheels, this will be populated from `slimgest._build_info`.
    In editable installs / source checkouts, fields fall back to "unknown"
    (except for version, which tries package metadata first).
    """

    version = "unknown"
    try:
        version = metadata.version("slimgest")
    except metadata.PackageNotFoundError:
        pass

    try:
        # Created during CI build; not present in source by default.
        from . import _build_info as bi  # type: ignore

        return BuildInfo(
            version=getattr(bi, "version", version),
            build_date_utc=getattr(bi, "build_date_utc", "unknown"),
            build_timestamp_utc=getattr(bi, "build_timestamp_utc", "unknown"),
            git_sha=getattr(bi, "git_sha", "unknown"),
            git_branch=getattr(bi, "git_branch", "unknown"),
            git_ref=getattr(bi, "git_ref", "unknown"),
        )
    except Exception:
        return BuildInfo(
            version=version,
            build_date_utc="unknown",
            build_timestamp_utc="unknown",
            git_sha="unknown",
            git_branch="unknown",
            git_ref="unknown",
        )


__all__ = ["BuildInfo", "get_build_info"]

