"""Ollama provider — local or remote LLM inference via Ollama."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .base import BaseProvider
from ..response import LLMResponse, LLMChunk

logger = logging.getLogger(__name__)


def _is_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """True if base_url + /api/tags responds with anything (even 4xx)."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags",
                                    timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def _is_localhost(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


def _start_local_ollama(base_url: str, wait: int = 10) -> None:
    """Spawn `ollama serve` detached, listening on base_url's host:port.

    Raises ConnectionError if ollama isn't installed or doesn't come up.
    """
    if not shutil.which("ollama"):
        raise ConnectionError(
            "'ollama' command not found. Install from https://ollama.com/download."
        )

    parsed = urlparse(base_url)
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{parsed.hostname}:{parsed.port or 11434}"

    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env, start_new_session=True,
    )

    for _ in range(wait * 2):  # poll every 0.5s
        if _is_reachable(base_url, timeout=1.0):
            return
        time.sleep(0.5)
    raise ConnectionError(
        f"Started 'ollama serve' (OLLAMA_HOST={env['OLLAMA_HOST']}) but "
        f"it didn't become reachable at {base_url} within {wait}s."
    )


def kill_local_server() -> bool:
    """pkill any local 'ollama serve' process.

    Aggressive — terminates *every* matching process on the host, regardless
    of who started it. Returns True if any were killed (pkill exit 0),
    False if there was nothing to kill (pkill exit 1).
    """
    try:
        result = subprocess.run(
            ["pkill", "-f", "ollama serve"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        # pkill missing — fall back to killall, then give up.
        try:
            result = subprocess.run(
                ["killall", "ollama"], capture_output=True, text=True,
            )
        except FileNotFoundError:
            logger.warning("Neither pkill nor killall found; cannot stop ollama serve.")
            return False
    return result.returncode == 0


def _ensure_local_ollama_running(base_url: str) -> None:
    """If Ollama at base_url isn't reachable, prompt the user (TTY only) to start it."""
    if _is_reachable(base_url):
        return

    msg = f"Ollama is not running at {base_url}."
    if not _is_localhost(base_url):
        raise ConnectionError(
            f"{msg} The endpoint isn't local, so 'ollama serve' won't help. "
            f"Check the remote host or fix base_url in your config."
        )
    if not sys.stdin.isatty():
        raise ConnectionError(f"{msg} Start it with: ollama serve")

    print(f"\n{msg}", file=sys.stderr)
    answer = input("Start it now? [Y/n]: ").strip().lower()
    if answer not in ("", "y", "yes"):
        raise ConnectionError("Ollama not running and user declined to start it.")
    _start_local_ollama(base_url)
    print(f"Ollama running at {base_url}.", file=sys.stderr)


def _fix_malformed_tool_json(raw: str) -> dict:
    """Try to fix and parse malformed tool call JSON from a model.

    Common issues:
    - Stray "] after string values (model confuses JSON with array syntax)
    - Raw newlines inside string values (should be \\n)

    Returns a parsed tool call dict {name, arguments} or None if unfixable.
    """
    if not raw:
        return None
    try:
        # Fix stray "] -> "
        fixed = re.sub(r'"\](\s*[,}])', r'"\1', raw)
        # Fix raw newlines inside JSON strings
        result = []
        in_string = False
        escape = False
        for ch in fixed:
            if escape:
                result.append(ch)
                escape = False
                continue
            if ch == '\\':
                escape = True
                result.append(ch)
                continue
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                continue
            if in_string and ch == '\n':
                result.append('\\n')
                continue
            result.append(ch)
        fixed = ''.join(result)

        obj = json.loads(fixed)
        # Could be {command: ..., timeout: ...} or {name: ..., arguments: ...}
        if "name" in obj:
            return {"name": obj["name"], "arguments": obj.get("arguments", {})}
        elif "command" in obj:
            return {"name": "bash_run", "arguments": obj}
        elif "path" in obj:
            return {"name": "file_read", "arguments": obj}
        return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


class OllamaProvider(BaseProvider):
    """Provider for Ollama (local or remote via SSH tunnel)."""

    MAX_RETRIES = 3

    def __init__(self, config: dict):
        super().__init__(config)
        self._tunnel_port = None
        self._tunnel_config = config.get("ssh_tunnel")

        self._ensured = False
        if self._tunnel_config:
            self._setup_tunnel()
            self._ensured = True  # tunnel already verified in _setup_tunnel
        else:
            self.base_url = config.get("base_url", "http://localhost:11434")

    def _setup_tunnel(self):
        """Establish SSH tunnel to remote Ollama server.

        Config needs: user, host, remote_host, remote_port.
        """
        from ..ssh import open_tunnel

        tc = self._tunnel_config
        required = ["user", "host", "remote_host", "remote_port"]
        missing = [k for k in required if k not in tc]
        if missing:
            raise ValueError(f"ssh_tunnel config missing: {', '.join(missing)}")

        self._tunnel_port = open_tunnel(
            ssh_user=tc["user"],
            ssh_host=tc["host"],
            remote_host=tc["remote_host"],
            remote_port=tc["remote_port"],
            local_port=tc.get("local_port", 0),
            ssh_password=tc.get("password"),
            verify_url="/api/tags",
            verify_timeout=tc.get("verify_timeout", 30),
        )
        self.base_url = f"http://localhost:{self._tunnel_port}"
        logger.info(f"Ollama tunneled to {self.base_url}")

    def close(self):
        """Close the SSH tunnel if one was opened."""
        if self._tunnel_port:
            from ..ssh import close_tunnel
            close_tunnel(self._tunnel_port)
            self._tunnel_port = None

    def _get_client(self):
        """Get an Ollama client pointed at our base_url."""
        # Deferred connectivity check: only runs when this provider is actually
        # used (not at startup when every provider is instantiated). Cached so it
        # prompts at most once per instance and never loops.
        if not self._ensured:
            self._ensured = True
            _ensure_local_ollama_running(self.base_url)
        import ollama
        return ollama.Client(host=self.base_url)

    def chat(self, messages: list, tools: list = None,
             stream: bool = False, **overrides) -> LLMResponse:
        client = self._get_client()

        opts = self._merge_options(overrides)
        model = overrides.get("model", self.model)

        # Build Ollama options dict
        ollama_options = {}
        opt_map = {
            "temperature": "temperature",
            "num_ctx": "num_ctx",
            "num_predict": "num_predict",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
        }
        for key, ollama_key in opt_map.items():
            if key in opts:
                ollama_options[ollama_key] = opts[key]

        kwargs = {
            "model": model,
            "messages": messages,
            "options": ollama_options,
        }
        if tools:
            kwargs["tools"] = tools
        # Pin the model in VRAM. Without this, Ollama unloads idle models after
        # 5 minutes (default) — and reloading a 60+GB model from a networked
        # filesystem (e.g. PACE /storage/...) can take 30+ minutes. Users who
        # want different behaviour can set 'keep_alive' in their provider config.
        kwargs["keep_alive"] = opts.get("keep_alive", -1)

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if stream:
                    return self._stream_chat(client, kwargs)
                else:
                    return self._sync_chat(client, kwargs)
            except Exception as e:
                err_str = str(e)
                # Malformed tool call JSON from model — not retryable, return as text
                if "error parsing tool call" in err_str:
                    logger.warning(f"Model produced malformed tool call: {err_str}")
                    import re
                    raw_match = re.search(r"raw='(.+?)'", err_str, re.DOTALL)
                    raw_text = raw_match.group(1) if raw_match else ""
                    # Try to fix common JSON issues and parse the tool call
                    fixed_json = _fix_malformed_tool_json(raw_text)
                    if fixed_json:
                        return LLMResponse(
                            text="",
                            tool_calls=[fixed_json],
                            model=model,
                            done_reason="stop",
                        )
                    # Fall back to returning raw text for text-based parser
                    return LLMResponse(
                        text=raw_text,
                        model=model,
                        done_reason="error",
                    )
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Ollama request failed (attempt {attempt + 1}): {e}. "
                                   f"Retrying in {wait}s...")
                    time.sleep(wait)

        raise ConnectionError(f"Ollama request failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _sync_chat(self, client, kwargs: dict) -> LLMResponse:
        kwargs["stream"] = False
        data = client.chat(**kwargs)

        text = data.message.content or ""
        tool_calls = []
        if data.message.tool_calls:
            for tc in data.message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        prompt_tokens = getattr(data, "prompt_eval_count", 0) or 0
        completion_tokens = getattr(data, "eval_count", 0) or 0
        done_reason = getattr(data, "done_reason", "") or ""

        # Context window warning
        num_ctx = kwargs.get("options", {}).get("num_ctx")
        if num_ctx and prompt_tokens >= num_ctx * 0.95:
            status = "EXCEEDS" if prompt_tokens >= num_ctx else "near"
            logger.warning(
                f"Prompt is {prompt_tokens} tokens — {status} num_ctx limit ({num_ctx}). "
                f"Output quality may degrade. Increase num_ctx in config."
            )

        # Output truncation warning
        if done_reason == "length":
            num_predict = kwargs.get("options", {}).get("num_predict", "?")
            logger.warning(
                f"LLM output truncated — hit num_predict limit ({num_predict}). "
                f"Increase num_predict in config for longer responses."
            )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            done_reason=done_reason,
            model=kwargs["model"],
        )

    def _stream_chat(self, client, kwargs: dict) -> LLMResponse:
        kwargs["stream"] = True
        response = LLMResponse(model=kwargs["model"])

        def _iter():
            full_text = ""
            all_tool_calls = []

            try:
                stream = client.chat(**kwargs)
                for chunk in stream:
                    chunk_text = ""
                    chunk_tools = []

                    if chunk.message.content:
                        chunk_text = chunk.message.content
                        full_text += chunk_text

                    if chunk.message.tool_calls:
                        for tc in chunk.message.tool_calls:
                            tc_dict = {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            all_tool_calls.append(tc_dict)
                            chunk_tools.append(tc_dict)

                    if chunk_text or chunk_tools:
                        yield LLMChunk(text=chunk_text, tool_calls=chunk_tools)

            except Exception as e:
                err_str = str(e)
                if "error parsing tool call" in err_str:
                    raw_match = re.search(r"raw='(.+?)'", err_str, re.DOTALL)
                    raw_text = raw_match.group(1) if raw_match else ""
                    fixed_json = _fix_malformed_tool_json(raw_text)
                    if fixed_json:
                        response.tool_calls = [fixed_json]
                        return
                    response.text = raw_text
                    response.done_reason = "error"
                    return
                raise

            # After stream completes, update the response object
            response.text = full_text
            response.tool_calls = all_tool_calls
            response.prompt_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
            response.completion_tokens = getattr(chunk, "eval_count", 0) or 0
            response.done_reason = getattr(chunk, "done_reason", "") or ""

            # Context window warning
            num_ctx = kwargs.get("options", {}).get("num_ctx")
            if num_ctx and response.prompt_tokens >= num_ctx * 0.95:
                status = "EXCEEDS" if response.prompt_tokens >= num_ctx else "near"
                logger.warning(
                    f"Prompt is {response.prompt_tokens} tokens — {status} "
                    f"num_ctx limit ({num_ctx}). Increase num_ctx in config."
                )

            if response.done_reason == "length":
                logger.warning("LLM output truncated — hit num_predict limit.")

        response._stream_iter = _iter()
        return response
