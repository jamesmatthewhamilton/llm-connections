"""Tests for the provider discovery/modelling/presentation layers.

Covers the three layers separately: the STABLE data contract (ProviderInfo /
ProviderCatalog serialization), the acquisition seam (sources + merge order),
and the presentation seam (formatters + the legacy convenience functions).
"""

import json

import pytest

from llm_connections import ProviderCatalog, ProviderInfo
from llm_connections.providers import (
    SCHEMA_VERSION,
    DictSource,
    FileSource,
    JsonFormatter,
    PlainFormatter,
    TableFormatter,
    available_formats,
    describe_providers,
    format_providers,
    get_formatter,
    register_formatter,
)

RAW = {
    "default": {"provider": "ollama", "model": "gpt-oss:120b",
                "num_ctx": 65536, "temperature": 0.0,
                "slurm_session": "pace-gpt-oss-120b"},
    "lmistral": {"provider": "ollama", "model": "mistral-nemo:latest",
                 "num_ctx": 8192, "temperature": 0.0},
}


# ── Data model: STABLE contract ─────────────────────────────────────────────

def test_from_config_splits_known_keys_and_keeps_rest_in_options():
    info = ProviderInfo.from_config("default", RAW["default"])
    assert info.name == "default"
    assert info.provider == "ollama"
    assert info.model == "gpt-oss:120b"
    # everything else is preserved verbatim in options
    assert info.options == {"num_ctx": 65536, "temperature": 0.0,
                            "slurm_session": "pace-gpt-oss-120b"}


def test_unknown_fields_are_preserved_round_trip():
    """Forward-compat: a key this library doesn't know must survive a round trip."""
    info = ProviderInfo.from_config(
        "future", {"provider": "ollama", "model": "m",
                   "some_field_invented_in_2030": {"nested": True}}
    )
    restored = ProviderInfo.from_dict(info.to_dict())
    assert restored.options["some_field_invented_in_2030"] == {"nested": True}
    assert restored == info


def test_provider_info_to_from_dict_is_lossless():
    info = ProviderInfo.from_config("default", RAW["default"])
    assert ProviderInfo.from_dict(info.to_dict()) == info


def test_missing_provider_and_model_default_to_empty():
    info = ProviderInfo.from_config("bare", {})
    assert info.provider == ""
    assert info.model == ""
    assert info.options == {}


def test_catalog_serialized_shape_is_stable():
    """Golden contract test: guards the serialized schema against accidental drift.

    If this fails, a field was renamed/removed/reshaped — that's a breaking
    change for downstream consumers and must be intentional + version-bumped.
    """
    catalog = ProviderCatalog.from_sources([DictSource(RAW)])
    assert catalog.to_dict() == {
        "schema_version": SCHEMA_VERSION,
        "providers": [
            {"name": "default", "provider": "ollama", "model": "gpt-oss:120b",
             "options": {"num_ctx": 65536, "temperature": 0.0,
                         "slurm_session": "pace-gpt-oss-120b"}},
            {"name": "lmistral", "provider": "ollama",
             "model": "mistral-nemo:latest",
             "options": {"num_ctx": 8192, "temperature": 0.0}},
        ],
    }


def test_catalog_dict_round_trip():
    catalog = ProviderCatalog.from_sources([DictSource(RAW)])
    restored = ProviderCatalog.from_dict(catalog.to_dict())
    assert restored.to_dict() == catalog.to_dict()


def test_catalog_access_helpers():
    catalog = ProviderCatalog.from_sources([DictSource(RAW)])
    assert len(catalog) == 2
    assert bool(catalog) is True
    assert catalog.names() == ["default", "lmistral"]
    assert catalog.get("default").model == "gpt-oss:120b"
    assert catalog.get("missing") is None
    assert not ProviderCatalog()


# ── Acquisition seam ────────────────────────────────────────────────────────

def test_from_sources_later_source_wins_on_collision():
    base = DictSource({"a": {"provider": "ollama", "model": "old"}})
    override = DictSource({"a": {"provider": "ollama", "model": "new"}})
    catalog = ProviderCatalog.from_sources([base, override])
    assert catalog.get("a").model == "new"


def test_filesource_missing_file_yields_empty():
    assert FileSource("/no/such/file.yaml").load() == {}


def test_from_paths_merges_files(tmp_path):
    home = tmp_path / "home.yaml"
    proj = tmp_path / "proj.yaml"
    home.write_text("llm-providers:\n  a:\n    provider: ollama\n    model: home\n")
    proj.write_text("llm-providers:\n  a:\n    provider: ollama\n    model: proj\n"
                    "  b:\n    provider: ollama\n    model: only-proj\n")
    catalog = ProviderCatalog.from_paths([str(home), str(proj)])
    assert catalog.get("a").model == "proj"          # project wins
    assert catalog.names() == ["a", "b"]


# ── Presentation seam ───────────────────────────────────────────────────────

def test_table_formatter_aligns_and_lists_details():
    out = ProviderCatalog.from_sources([DictSource(RAW)]).format("table")
    lines = out.splitlines()
    assert lines[0] == "Available LLM providers:"
    assert "default" in lines[1] and "gpt-oss:120b" in lines[1]
    assert "slurm_session=pace-gpt-oss-120b" in lines[1]
    # name column padded to width of the longest name ("lmistral")
    assert "default " in lines[1]


def test_table_formatter_empty_catalog_notice():
    out = ProviderCatalog().format("table")
    assert "(no providers configured)" in out


def test_table_formatter_no_title():
    out = TableFormatter(title=None).render(ProviderCatalog.from_sources([DictSource(RAW)]))
    assert not out.startswith("Available LLM providers")


def test_plain_formatter():
    out = PlainFormatter().render(ProviderCatalog.from_sources([DictSource(RAW)]))
    assert out.splitlines()[0] == "default\tollama\tgpt-oss:120b"


def test_json_formatter_is_machine_readable():
    out = ProviderCatalog.from_sources([DictSource(RAW)]).format("json")
    parsed = json.loads(out)
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["providers"][0]["name"] == "default"


def test_get_formatter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown formatter"):
        get_formatter("nope")


def test_register_formatter_extends_registry():
    class UpperFormatter:
        def render(self, catalog):
            return ",".join(p.name.upper() for p in catalog)

    register_formatter("upper", UpperFormatter)
    try:
        assert "upper" in available_formats()
        out = ProviderCatalog.from_sources([DictSource(RAW)]).format("upper")
        assert out == "DEFAULT,LMISTRAL"
    finally:
        from llm_connections.providers import _FORMATTERS
        _FORMATTERS.pop("upper", None)


# ── Convenience functions (backward-compatible) ─────────────────────────────

def test_format_providers_matches_catalog_table():
    legacy = format_providers(RAW, title="Available LLM providers")
    catalog = ProviderCatalog.from_sources([DictSource(RAW)]).format("table")
    assert legacy == catalog


def test_format_providers_empty():
    assert "(no providers configured)" in format_providers({})


def test_describe_providers_reads_file(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("llm-providers:\n  a:\n    provider: ollama\n    model: m\n")
    assert "a" in describe_providers(str(cfg))
    assert '"a"' in describe_providers(str(cfg), fmt="json")


def test_describe_providers_missing_file_is_safe(tmp_path):
    out = describe_providers(str(tmp_path / "nope.yaml"))
    assert "(no providers configured)" in out


# ── Validation (house-style, non-raising) ───────────────────────────────────

def test_validate_clean_config_has_no_issues():
    catalog = ProviderCatalog.from_sources([DictSource(RAW)])
    assert catalog.validate() == {}
    assert ProviderInfo.from_config("default", RAW["default"]).validate() == []


def test_validate_flags_unknown_provider_type():
    info = ProviderInfo.from_config("x", {"provider": "openai", "model": "gpt-4o"})
    issues = info.validate()
    assert any("unknown provider type 'openai'" in i for i in issues)


def test_validate_flags_missing_provider_and_model():
    info = ProviderInfo.from_config("x", {})
    issues = info.validate()
    assert "missing 'provider'" in issues
    assert "missing 'model'" in issues


def test_validate_flags_bad_option_types():
    info = ProviderInfo.from_config(
        "x", {"provider": "ollama", "model": "m",
              "num_ctx": -1, "temperature": "hot", "top_p": 2}
    )
    joined = " ".join(info.validate())
    assert "num_ctx" in joined and "temperature" in joined and "top_p" in joined


def test_validate_rejects_bool_as_number():
    # bool is an int subclass; it must not satisfy the int/number option rules.
    info = ProviderInfo.from_config("x", {"provider": "ollama", "model": "m",
                                          "num_ctx": True})
    assert any("num_ctx" in i for i in info.validate())


def test_validate_ignores_unknown_option_keys():
    """Forward-compat: an unrecognised option must NOT be a validation issue."""
    info = ProviderInfo.from_config(
        "x", {"provider": "ollama", "model": "m", "future_knob_2030": "anything"}
    )
    assert info.validate() == []


def test_catalog_validate_aggregates_only_bad_entries():
    raw = {"good": {"provider": "ollama", "model": "m"},
           "bad": {"provider": "ollama", "model": ""}}
    result = ProviderCatalog.from_sources([DictSource(raw)]).validate()
    assert list(result) == ["bad"]
    assert "missing 'model'" in result["bad"]


# ── Immutability ────────────────────────────────────────────────────────────

def test_options_are_read_only():
    info = ProviderInfo.from_config("default", RAW["default"])
    with pytest.raises(TypeError):
        info.options["num_ctx"] = 1  # MappingProxyType is read-only


def test_mutating_source_dict_does_not_leak_into_info():
    cfg = {"provider": "ollama", "model": "m", "num_ctx": 8192}
    info = ProviderInfo.from_config("x", cfg)
    cfg["num_ctx"] = 999
    assert info.options["num_ctx"] == 8192


def test_provider_info_is_intentionally_unhashable():
    info = ProviderInfo.from_config("default", RAW["default"])
    with pytest.raises(TypeError):
        hash(info)


# ── Version-aware deserialization ───────────────────────────────────────────

def test_from_dict_accepts_current_version():
    catalog = ProviderCatalog.from_sources([DictSource(RAW)])
    assert ProviderCatalog.from_dict(catalog.to_dict()).names() == catalog.names()


def test_from_dict_newer_version_warns_but_parses(caplog):
    data = {"schema_version": SCHEMA_VERSION + 1,
            "providers": [{"name": "a", "provider": "ollama", "model": "m",
                           "options": {}}]}
    with caplog.at_level("WARNING", logger="llm_connections.providers"):
        catalog = ProviderCatalog.from_dict(data)
    assert catalog.names() == ["a"]
    assert any("newer than supported" in r.message for r in caplog.records)


def test_from_dict_older_version_routes_through_migrate(monkeypatch):
    called = {}

    def fake_migrate(data, from_version):
        called["from_version"] = from_version
        return data

    monkeypatch.setattr(ProviderCatalog, "_migrate", staticmethod(fake_migrate))
    ProviderCatalog.from_dict({"schema_version": 0, "providers": []})
    assert called["from_version"] == 0


# ── FileSource error handling (no silent swallow) ───────────────────────────

def test_filesource_malformed_yaml_warns(tmp_path, caplog):
    bad = tmp_path / "bad.yaml"
    bad.write_text("llm-providers:\n  a: [unclosed\n")
    with caplog.at_level("WARNING", logger="llm_connections.providers"):
        result = FileSource(str(bad)).load()
    assert result == {}
    assert any("failed to parse" in r.message for r in caplog.records)


def test_filesource_missing_file_is_quiet(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="llm_connections.providers"):
        result = FileSource(str(tmp_path / "nope.yaml")).load()
    assert result == {}
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ── Published JSON Schema contract ──────────────────────────────────────────

def _load_published_schema():
    # Reads the schema as shipped package data — also asserts it's packaged.
    from importlib.resources import files
    text = (files("llm_connections") / "schemas"
            / "provider_catalog.v1.json").read_text()
    return json.loads(text)


def test_to_dict_validates_against_published_schema():
    jsonschema = pytest.importorskip("jsonschema")  # test-only dep
    schema = _load_published_schema()
    catalog = ProviderCatalog.from_sources([DictSource(RAW)])
    jsonschema.validate(instance=catalog.to_dict(), schema=schema)


def test_published_schema_version_matches_code():
    schema = _load_published_schema()
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_empty_catalog_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(instance=ProviderCatalog().to_dict(),
                        schema=_load_published_schema())
