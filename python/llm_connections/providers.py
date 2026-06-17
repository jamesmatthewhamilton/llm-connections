"""Provider discovery, modelling, and presentation.

This module is built in three layers, each with a *different* stability promise.
That separation is deliberate: it is what lets the library evolve for years
without breaking the (potentially many) downstream consumers that depend on it.

    ┌─ Presentation ─────────  PROVISIONAL. Convenience only. Output is NOT a
    │   Formatter + format_*    stable contract — if you need stable output,
    │                          consume the data model and render it yourself.
    ├─ Acquisition ──────────  PROVISIONAL, EXTENSIBLE. ProviderSource lets you
    │   ProviderSource          add new origins (env, remote, secrets) without
    │                          forking. Built-ins: FileSource, DictSource.
    └─ Data model ───────────  STABLE. ProviderInfo / ProviderCatalog are the
        ProviderInfo            contract: serializable, versioned, and evolved
        ProviderCatalog         ADDITIVELY ONLY (new optional fields; never
                                rename/remove; unknown fields are preserved).

Stability tiers
---------------
* STABLE      — ``ProviderInfo``, ``ProviderCatalog``, ``SCHEMA_VERSION``. Safe
                to depend on; changes follow SemVer with deprecation warnings.
* PROVISIONAL — the seams (``ProviderSource``, ``Formatter``, the registry, the
                built-in sources/formatters) and the convenience functions. May
                change while we learn the right shape; will not change casually.
* INTERNAL    — anything prefixed ``_``. No guarantees.

Extension points (documented seams; implementations intentionally deferred)
---------------------------------------------------------------------------
* New config origins: implement ``ProviderSource`` (e.g. an ``EnvSource`` or a
  remote/secrets-manager source) and pass it to ``ProviderCatalog.from_sources``.
* New output formats: implement ``Formatter`` and ``register_formatter("name", …)``.
  A future release can auto-discover third-party formatters/sources via
  ``importlib.metadata`` entry points without changing this module's contract.
"""

import json
import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

import yaml

from .config import load_providers
from .llm_providers import PROVIDER_NAMES

logger = logging.getLogger(__name__)

# Bump (additively) whenever the serialized shape of ProviderCatalog.to_dict()
# gains fields. Consumers branch on this; old payloads must keep deserializing.
SCHEMA_VERSION = 1

__all__ = [
    # ── STABLE: data contract ──────────────────────────────────────────────
    "ProviderInfo",
    "ProviderCatalog",
    "SCHEMA_VERSION",
    # ── PROVISIONAL: acquisition seam ──────────────────────────────────────
    "ProviderSource",
    "FileSource",
    "DictSource",
    # ── PROVISIONAL: presentation seam ─────────────────────────────────────
    "Formatter",
    "register_formatter",
    "get_formatter",
    "available_formats",
    # ── PROVISIONAL: convenience (presentation; output not a stable contract)─
    "format_providers",
    "describe_providers",
]


# ════════════════════════════════════════════════════════════════════════════
# Data model — STABLE contract
# ════════════════════════════════════════════════════════════════════════════

def _is_int(v: Any) -> bool:
    # bool is an int subclass; reject it so True/False aren't accepted as numbers.
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# Type/range rules for the KNOWN_OPTIONS vocabulary (see llm_providers/base.py).
# We validate the *shape* of recognised options; unknown keys are left untouched
# so a newer config still round-trips through an older library.
_OPTION_RULES: dict[str, tuple] = {
    "num_ctx": (lambda v: _is_int(v) and v > 0, "must be a positive integer"),
    "num_predict": (_is_int, "must be an integer"),
    "max_tokens": (lambda v: _is_int(v) and v > 0, "must be a positive integer"),
    "temperature": (lambda v: _is_number(v) and v >= 0, "must be a non-negative number"),
    "top_p": (lambda v: _is_number(v) and 0 <= v <= 1, "must be a number in [0, 1]"),
    "top_k": (lambda v: _is_int(v) and v >= 0, "must be a non-negative integer"),
    "repeat_penalty": (lambda v: _is_number(v) and v > 0, "must be a positive number"),
}


@dataclass(frozen=True)
class ProviderInfo:
    """One provider, as configured (never connected to).

    ``options`` holds every config key other than ``provider``/``model`` —
    including keys this library doesn't recognise. Preserving unknown keys is
    what makes the model forward-compatible: a newer config round-trips
    losslessly through an older library.
    """

    name: str
    provider: str
    model: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Make `options` genuinely read-only (frozen=True only blocks attribute
        # *rebinding*, not mutation of a dict it points at). Side effect: this
        # type is intentionally not hashable — it carries a mapping.
        if not isinstance(self.options, MappingProxyType):
            object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems with this entry (empty = ok).

        Non-raising by design: callers decide how to react (warn, skip, fail).
        Validates known structure only — unrecognised option keys are not flagged,
        preserving forward-compatibility.
        """
        issues: list[str] = []
        if not self.provider:
            issues.append("missing 'provider'")
        elif self.provider not in PROVIDER_NAMES:
            known = ", ".join(sorted(PROVIDER_NAMES))
            issues.append(f"unknown provider type '{self.provider}' (known: {known})")
        if not self.model:
            issues.append("missing 'model'")
        for key, (ok, msg) in _OPTION_RULES.items():
            if key in self.options and not ok(self.options[key]):
                issues.append(f"option '{key}' {msg}, got {self.options[key]!r}")
        return issues

    @classmethod
    def from_config(cls, name: str, cfg: Mapping[str, Any] | None) -> "ProviderInfo":
        """Build from a raw YAML provider entry (``{provider, model, …}``)."""
        rest = dict(cfg or {})
        provider = str(rest.pop("provider", "") or "")
        model = str(rest.pop("model", "") or "")
        return cls(name=str(name), provider=provider, model=model, options=rest)

    def to_dict(self) -> dict:
        """Serialize to the stable, machine-readable shape."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderInfo":
        """Inverse of :meth:`to_dict`. Tolerates missing optional fields."""
        return cls(
            name=str(data["name"]),
            provider=str(data.get("provider", "") or ""),
            model=str(data.get("model", "") or ""),
            options=dict(data.get("options") or {}),
        )


class ProviderCatalog:
    """An ordered collection of :class:`ProviderInfo`, merged from sources.

    Holds *config only* — constructing one never opens a connection, tunnel, or
    Slurm job, so it is safe on fast paths (``--help``, ``--list``). This is a
    deliberate counterpart to ``LLMConnection``'s registry, which holds *live*
    provider instances.
    """

    def __init__(self, providers: Mapping[str, ProviderInfo] | None = None):
        # Insertion order is meaningful (it's the merge/display order).
        self._providers: dict[str, ProviderInfo] = dict(providers or {})

    # ── construction ───────────────────────────────────────────────────────
    @classmethod
    def from_sources(cls, sources) -> "ProviderCatalog":
        """Merge providers from an ordered iterable of :class:`ProviderSource`.

        Later sources win on name collisions (so a project config can override
        a home config). A source that yields nothing is simply skipped.
        """
        merged: dict[str, ProviderInfo] = {}
        for src in sources:
            for name, cfg in src.load().items():
                merged[name] = ProviderInfo.from_config(name, cfg)
        return cls(merged)

    @classmethod
    def from_paths(cls, paths) -> "ProviderCatalog":
        """Convenience: merge a sequence of YAML config paths (later wins)."""
        return cls.from_sources([FileSource(p) for p in paths])

    # ── access ─────────────────────────────────────────────────────────────
    def __iter__(self) -> Iterator[ProviderInfo]:
        return iter(self._providers.values())

    def __len__(self) -> int:
        return len(self._providers)

    def __bool__(self) -> bool:
        return bool(self._providers)

    def get(self, name: str) -> ProviderInfo | None:
        return self._providers.get(name)

    def names(self) -> list[str]:
        return list(self._providers)

    def validate(self) -> dict[str, list[str]]:
        """Validate every entry; return ``{name: issues}`` for providers with
        problems (empty dict = all valid). See :meth:`ProviderInfo.validate`."""
        return {p.name: issues for p in self if (issues := p.validate())}

    # ── serialization (STABLE, versioned, additive-only) ───────────────────
    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "providers": [p.to_dict() for p in self],
        }

    @staticmethod
    def _migrate(data: Mapping[str, Any], from_version: int) -> Mapping[str, Any]:
        """Upgrade an older serialized payload to the current schema.

        Identity today (only v1 exists). When SCHEMA_VERSION is bumped, add the
        per-version transforms here so old payloads keep deserializing — this is
        the seam that makes SCHEMA_VERSION a real contract, not a decoration.
        """
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderCatalog":
        version = data.get("schema_version", SCHEMA_VERSION)
        if version > SCHEMA_VERSION:
            logger.warning(
                "provider catalog schema_version %s is newer than supported %s; "
                "parsing leniently (unknown fields ignored)", version, SCHEMA_VERSION,
            )
        elif version < SCHEMA_VERSION:
            data = cls._migrate(data, version)
        infos = [ProviderInfo.from_dict(d) for d in data.get("providers", [])]
        return cls({p.name: p for p in infos})

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # ── presentation (delegates through the Formatter seam) ────────────────
    def format(self, fmt: str = "table", **kwargs) -> str:
        """Render via a named formatter (``table``/``plain``/``json`` built in)."""
        return get_formatter(fmt, **kwargs).render(self)

    def __repr__(self) -> str:
        return f"ProviderCatalog({self.names()})"


# ════════════════════════════════════════════════════════════════════════════
# Acquisition seam — PROVISIONAL
# ════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ProviderSource(Protocol):
    """A place providers come from. Implement this to add new origins.

    Deferred built-ins a consumer (or future release) might add: an env-var
    source, a remote-HTTP source, a secrets-manager source. None require
    changes here — that's the point of the seam.
    """

    def load(self) -> Mapping[str, Mapping[str, Any]]:
        """Return a ``{name: raw_config}`` mapping. Empty if nothing found."""
        ...


class FileSource:
    """Providers from a single YAML file's ``llm-providers:`` key.

    A missing file or a file without that key yields ``{}`` rather than raising,
    so it is safe to list optional/expected-absent configs.
    """

    def __init__(self, path: str):
        self.path = path

    def load(self) -> Mapping[str, Mapping[str, Any]]:
        try:
            return load_providers(self.path)
        except FileNotFoundError:
            logger.debug("provider config not found: %s", self.path)
            return {}
        except ValueError as e:
            # No 'llm-providers:' key — a genuinely empty/expected-absent config.
            logger.debug("no providers in %s: %s", self.path, e)
            return {}
        except yaml.YAMLError as e:
            # Malformed YAML is a real problem, not an absence — surface it so a
            # broken config doesn't look identical to "no file".
            logger.warning("failed to parse provider config %s: %s", self.path, e)
            return {}


class DictSource:
    """Providers supplied programmatically — for tests and in-code registration."""

    def __init__(self, providers: Mapping[str, Mapping[str, Any]]):
        self._providers = dict(providers or {})

    def load(self) -> Mapping[str, Mapping[str, Any]]:
        return dict(self._providers)


# ════════════════════════════════════════════════════════════════════════════
# Presentation seam — PROVISIONAL (output is NOT a stable contract)
# ════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Formatter(Protocol):
    """Renders a :class:`ProviderCatalog` to a string. Output may change."""

    def render(self, catalog: "ProviderCatalog") -> str:
        ...


# Config keys surfaced inline by the table formatter, in this order, when present.
_DETAIL_KEYS = (
    "num_ctx", "temperature", "num_predict", "top_p", "top_k",
    "max_tokens", "base_url", "slurm_session",
)


class TableFormatter:
    """Aligned, human-readable columns: ``name  provider  model  details``."""

    def __init__(self, *, title: str | None = "Available LLM providers",
                 indent: str = "  "):
        self.title = title
        self.indent = indent

    def _details(self, info: ProviderInfo) -> str:
        parts = [f"{k}={info.options[k]}" for k in _DETAIL_KEYS
                 if info.options.get(k) is not None]
        if info.options.get("ssh_tunnel"):
            parts.append("ssh_tunnel")
        return f"   {'  '.join(parts)}" if parts else ""

    def render(self, catalog: "ProviderCatalog") -> str:
        if not catalog:
            body = f"{self.indent}(no providers configured)"
            return f"{self.title}:\n{body}" if self.title else body

        name_w = max(len(p.name) for p in catalog)
        type_w = max(len(p.provider or "?") for p in catalog)

        lines = []
        for p in catalog:
            lines.append(
                f"{self.indent}{p.name:<{name_w}}  "
                f"{(p.provider or '?'):<{type_w}}  "
                f"{p.model or '(no model)'}{self._details(p)}"
            )
        block = "\n".join(lines)
        return f"{self.title}:\n{block}" if self.title else block


class PlainFormatter:
    """One ``name<TAB>provider<TAB>model`` line per provider. Easy to ``cut``/grep."""

    def render(self, catalog: "ProviderCatalog") -> str:
        return "\n".join(f"{p.name}\t{p.provider}\t{p.model}" for p in catalog)


class JsonFormatter:
    """The catalog's stable serialized form as JSON. Machine-readable."""

    def __init__(self, *, indent: int = 2):
        self.indent = indent

    def render(self, catalog: "ProviderCatalog") -> str:
        return catalog.to_json(indent=self.indent)


# Name -> factory(**kwargs) -> Formatter. The registry is the extension point;
# a future release can populate it from importlib.metadata entry points.
_FORMATTERS: dict[str, Any] = {
    "table": TableFormatter,
    "plain": PlainFormatter,
    "json": JsonFormatter,
}


def register_formatter(name: str, factory) -> None:
    """Register a formatter factory under ``name`` (overrides any existing)."""
    _FORMATTERS[name] = factory


def get_formatter(name: str = "table", **kwargs) -> Formatter:
    """Instantiate a registered formatter, forwarding kwargs to its factory."""
    try:
        factory = _FORMATTERS[name]
    except KeyError:
        avail = ", ".join(sorted(_FORMATTERS)) or "(none)"
        raise ValueError(f"Unknown formatter '{name}'. Available: {avail}") from None
    return factory(**kwargs)


def available_formats() -> list[str]:
    """Names of all registered formatters."""
    return sorted(_FORMATTERS)


# ════════════════════════════════════════════════════════════════════════════
# Convenience functions — PROVISIONAL (thin wrappers; output is not a contract)
# ════════════════════════════════════════════════════════════════════════════

def format_providers(providers: Mapping[str, Mapping[str, Any]], *,
                     title: str | None = "Available LLM providers",
                     indent: str = "  ") -> str:
    """Format a raw ``{name: config}`` mapping as an aligned table.

    Kept for callers that already hold a raw providers mapping (e.g. the result
    of :func:`load_providers`). New code should prefer building a
    :class:`ProviderCatalog` and calling ``.format(...)`` / ``.to_json()``.
    """
    catalog = ProviderCatalog(
        {n: ProviderInfo.from_config(n, c) for n, c in (providers or {}).items()}
    )
    return catalog.format("table", title=title, indent=indent)


def describe_providers(yaml_path: str | None = None, *, fmt: str = "table",
                       **kwargs) -> str:
    """Read a YAML config (no connection) and return a formatted summary.

    Defaults to the library's standard ``~/.llm-connections/config.yaml`` and
    the ``table`` formatter; pass ``fmt="json"`` for machine output. A missing
    file or missing ``llm-providers:`` key yields the empty notice, so this is
    safe in a ``--help`` epilog.
    """
    if yaml_path is None:
        from .client import DEFAULT_CONFIG_PATH
        yaml_path = DEFAULT_CONFIG_PATH
    return ProviderCatalog.from_paths([yaml_path]).format(fmt, **kwargs)
