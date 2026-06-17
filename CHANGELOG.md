# Changelog

All notable changes to `llm-connections` are documented here. The project aims
to follow [Semantic Versioning](https://semver.org/). The **public API surface
and its stability tiers** are what SemVer applies to:

- **STABLE** — `LLMConnection`, `LLMResponse`, `LLMChunk`, `connect_ssh`, and the
  provider-discovery data contract `ProviderInfo` / `ProviderCatalog` (plus their
  serialized form, versioned by `SCHEMA_VERSION` and published as
  `llm_connections/schemas/provider_catalog.v{N}.json`). Breaking changes here
  require a major bump and a deprecation period.
- **PROVISIONAL** — the seams in `llm_connections.providers` (`ProviderSource`,
  `Formatter`, the formatter registry) and the convenience helpers
  (`format_providers`, `describe_providers`). May change in minor releases.
- **INTERNAL** — anything prefixed `_`. No guarantees.

## [0.1.0] — unreleased

First versioned release; establishes the packaging and stability contract.

### Added
- `pyproject.toml` (setuptools): pip-installable distribution with provider SDKs
  as extras (`[litellm]`, `[ollama]`, `[slurm]`) and a `[test]` extra.
- PEP 561 `py.typed` marker so downstream type-checkers consume the shipped types.
- `__version__` on the package (resolved via `importlib.metadata`).
- Provider data contract: `ProviderInfo` / `ProviderCatalog`, with additive,
  versioned `to_dict()` / `from_dict()` serialization and the published JSON
  Schema `schemas/provider_catalog.v1.json`.
- `ProviderInfo.validate()` / `ProviderCatalog.validate()` — non-raising,
  house-style structural validation (mirrors `slurm_config.py`).
- Acquisition seam (`ProviderSource`, `FileSource`, `DictSource`) and presentation
  seam (`Formatter`, `register_formatter`, built-in `table`/`plain`/`json`).
- CI workflow running the test suite (incl. golden + JSON Schema contract tests)
  and type-checking across Python 3.10–3.12.

### Notes
- `ProviderInfo.options` is read-only (`MappingProxyType`); the type is
  intentionally not hashable because it carries a mapping.
- `FileSource` logs (never silently swallows): debug for expected-absent configs,
  warning for malformed YAML.
