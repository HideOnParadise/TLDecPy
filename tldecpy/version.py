"""Package version metadata."""

from __future__ import annotations

from .schemas import VersionInfo

__version__ = "1.0.0"


def get_version_info() -> VersionInfo:
    """
    Return package and API version metadata.

    Returns
    -------
    VersionInfo
        Structured version payload containing package and API versions.
    """
    return VersionInfo(
        version=__version__,
        api_version=__version__,
        build="ga",
    )
