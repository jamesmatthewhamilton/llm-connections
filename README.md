# llm-connections

I, Matthew Hamilton, made this for self enrichment.

A config-driven, multi-provider LLM client. Local-first, dependency-light, and quietly opinionated about where your prompts are allowed to go - nowhere you didn't ask.

It reads exactly one key from your YAML and ignores everything else.

## Install

```bash
pip install -e .              # core: stdlib + PyYAML
pip install -e ".[ollama]"    # local/remote Ollama
pip install -e ".[litellm]"   # OpenAI-compatible endpoints (vLLM, TGI, …)
pip install -e ".[slurm]"     # the sibling submodule, for HPC sessions
```

## Quickstart

```python
from llm_connections import LLMConnection

LLMConnection.load()                       # ~/.llm-connections/config.yaml
client = LLMConnection.get("lmistral")     # a provider name from your config

resp = client.chat([{"role": "user", "content": "Hello"}])
print(resp.text)

# Streaming, for the impatient:
for chunk in client.chat(messages, stream=True):
    print(chunk.text, end="")
```

## Configuration

One file, one key. Everything under `llm-providers:` becomes a named provider.

```yaml
llm-providers:
  lmistral:                     # fully local
    provider: ollama
    model: mistral-nemo:latest
    num_ctx: 8192
    temperature: 0.0

  cluster:                      # remote Ollama spun up via Slurm
    provider: ollama
    model: gpt-oss:120b
    num_ctx: 65536
    slurm_session: pace-gpt-oss-120b
```

If the file doesn't exist, a scaffold is written for you. You're welcome; replace
the `REPLACE_WITH_*` placeholders before being surprised that they don't work.

## Providers

| `provider:` | Talks to | Notes |
|-------------|----------|-------|
| `ollama`    | Ollama at `base_url` (default `http://localhost:11434`), or a remote node via SSH tunnel | Local unless you say otherwise |
| `litellm`   | Any OpenAI-compatible `/chat/completions` endpoint | Named `litellm` for historical reasons; does **not** import litellm. It's hand-rolled httpx. We're as confused as you are. |

## Listing providers

`ProviderCatalog` reads your config **without connecting to anything** — safe for
`--help`, `--version`, and other fast paths.

```python
from llm_connections import ProviderCatalog

cat = ProviderCatalog.from_paths(["~/.llm-connections/config.yaml"])
print(cat.format())          # aligned table for humans
print(cat.format("json"))    # machine-readable, schema-versioned
issues = cat.validate()      # {name: [problems]} — empty dict means you got lucky
```

## Privacy

Your prompts and tool calls go to **exactly one place: the endpoint in your
config.** No telemetry, no analytics, no phone-home, no "anonymous usage
statistics" that are neither.

- `ollama` at localhost → stays on your machine.
- `slurm_session` / `ssh_tunnel` / a remote `base_url` → leaves your machine (to
  *your* host, over SSH), because you explicitly told it to.
- `litellm` pointed at a cloud URL → goes to the cloud, which is the one way to
  reach the internet here, and entirely your decision.

## Stability

The package documents three tiers so you know what you're allowed to lean on:

- **STABLE** — `LLMConnection`, `LLMResponse`, `LLMChunk`, `connect_ssh`,
  `ProviderInfo`, `ProviderCatalog`, and the JSON-Schema'd serialized form. We
  promise not to break these without a major version bump.
- **PROVISIONAL** — the seams (`ProviderSource`, `Formatter`, the registry) and
  the convenience helpers. We reserve the right to change our minds.
- **INTERNAL** — anything starting with `_`. You were never supposed to see that.

See [`CHANGELOG.md`](CHANGELOG.md) for the rest of the fine print.

## Optional: Slurm Sessions

If `slurm-manipulator` (the nested submodule) is installed, `SlurmSession` can
launch an Ollama server on an HPC node and tunnel it back to localhost. If it
isn't installed, importing `SlurmSession` raises a clear error telling you so,
rather than a mysterious one telling you nothing.
