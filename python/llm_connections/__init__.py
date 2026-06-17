"""Shared LLM client library — config-driven, multi-provider.

Core (always available):
    LLMConnection, LLMResponse, LLMChunk

Optional (requires the slurm-manipulator nested submodule):
    SlurmSession, SessionHandle, connect_ssh

Importing SlurmSession when slurm-manipulator isn't installed raises a
clear ImportError pointing at the submodule init command.
"""

from .client import LLMConnection
from .providers import ProviderCatalog, ProviderInfo
from .response import LLMChunk, LLMResponse
from .ssh import connect_ssh

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-connections")
except (ImportError, PackageNotFoundError):
    # Running from a source checkout / sys.path injection (not pip-installed):
    # fall back to the in-tree version. Keep in sync with pyproject.toml.
    __version__ = "0.1.0"

# Top-level surface is deliberately small: only the STABLE provider-discovery
# data contract (ProviderInfo/ProviderCatalog) is re-exported here. The
# PROVISIONAL layers — convenience helpers (format_providers/describe_providers)
# and the seams (ProviderSource, Formatter, register_formatter, …) — live in
# `llm_connections.providers`. Importing them from there is the signal that
# they're convenience, not part of the minimal blessed API.
__all__ = [
    "__version__",
    "LLMConnection",
    "LLMResponse",
    "LLMChunk",
    "connect_ssh",
    # provider discovery — STABLE data contract
    "ProviderInfo",
    "ProviderCatalog",
    # SlurmSession / SessionHandle exposed via __getattr__ — lazy so the
    # optional slurm-manipulator dependency is only resolved on access.
]


def __getattr__(name: str):
    if name in ("SlurmSession", "SessionHandle"):
        from .slurm_session import SessionHandle, SlurmSession
        return {"SlurmSession": SlurmSession, "SessionHandle": SessionHandle}[name]
    raise AttributeError(f"module 'llm_connections' has no attribute {name!r}")
