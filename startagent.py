#!/usr/bin/env python3
"""startagent — run coding agents without vendor lock.

Point Claude Code, Codex, OpenCode or Qwen Code at any Anthropic- or
OpenAI-compatible provider. One file, standard library only.

MIT License. See README.md.
"""
from __future__ import annotations

import getpass
import http.client
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

VERSION = "0.2.2"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value).expanduser()


XDG_CONFIG = _env_path("XDG_CONFIG_HOME", Path.home() / ".config")
DEFAULT_CONFIG_DIR = XDG_CONFIG / "startagent"
LEGACY_CONFIG_DIR = XDG_CONFIG / "deepseek"


class StartagentError(Exception):
    """A user-facing error."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    codex_id: str | None = None


@dataclass(frozen=True)
class Provider:
    key: str
    title: str
    anthropic_url: str
    openai_url: str | None
    main_models: tuple[Model, ...]
    fast_models: tuple[Model, ...]
    efforts: tuple[str, ...]
    default_effort: str
    compact_window: str | None
    codex_url: str | None
    key_env: str
    key_alt_envs: tuple[str, ...]
    key_url: str
    context_window: int
    custom: bool = False


PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        key="deepseek",
        title="DeepSeek",
        anthropic_url="https://api.deepseek.com/anthropic",
        openai_url="https://api.deepseek.com/v1",
        main_models=(
            Model("deepseek-flash[1m]", "DeepSeek V4.1 Flash — 1M, vision", "deepseek-v4-flash"),
            Model("deepseek-v4-pro[1m]", "DeepSeek V4 Pro — 1M, text", "deepseek-v4-pro"),
        ),
        fast_models=(
            Model("deepseek-flash", "DeepSeek Flash — fast"),
            Model("deepseek-flash[1m]", "DeepSeek Flash — 1M"),
        ),
        efforts=("low", "high", "max"),
        default_effort="max",
        compact_window="786432",
        codex_url="https://api.deepseek.com/",
        key_env="DEEPSEEK_API_KEY",
        key_alt_envs=(),
        key_url="https://platform.deepseek.com/api_keys",
        context_window=1048576,
    ),
    "kimi": Provider(
        key="kimi",
        title="Kimi / Moonshot",
        anthropic_url="https://api.moonshot.ai/anthropic",
        openai_url="https://api.moonshot.ai/v1",
        main_models=(
            Model("kimi-k3[1m]", "Kimi K3 — 1M"),
            Model("kimi-k2.7-code", "Kimi K2.7 Code"),
        ),
        fast_models=(Model("kimi-k2.7-code", "Kimi K2.7 Code — fast"),),
        efforts=("low", "high", "max"),
        default_effort="max",
        compact_window="1000000",
        codex_url=None,
        key_env="MOONSHOT_API_KEY",
        key_alt_envs=("KIMI_API_KEY",),
        key_url="https://platform.moonshot.ai/console/api-keys",
        context_window=1048576,
    ),
    "glm": Provider(
        key="glm",
        title="Z.ai GLM",
        anthropic_url="https://api.z.ai/api/anthropic",
        openai_url="https://api.z.ai/api/paas/v4",
        main_models=(
            Model("glm-5.3[1m]", "GLM-5.3 — 1M", "glm-5.3"),
            Model("glm-5.3-flash[1m]", "GLM-5.3 Flash — 1M", "glm-5.3-flash"),
        ),
        fast_models=(Model("glm-5.3-flash", "GLM-5.3 Flash — fast"),),
        efforts=("low", "high", "max"),
        default_effort="max",
        compact_window="1000000",
        codex_url="https://api.z.ai/api/v1",
        key_env="ZAI_API_KEY",
        key_alt_envs=("GLM_API_KEY",),
        key_url="https://z.ai/manage-apikey/apikey-list",
        context_window=1048576,
    ),
    "minimax": Provider(
        key="minimax",
        title="MiniMax",
        anthropic_url="https://api.minimax.io/anthropic",
        openai_url="https://api.minimax.io/v1",
        main_models=(
            Model("MiniMax-M3[1m]", "MiniMax M3 — 1M"),
            Model("MiniMax-M3.1-Flash[1m]", "MiniMax M3.1 Flash — 1M"),
        ),
        fast_models=(Model("MiniMax-M3.1-Flash", "MiniMax M3.1 Flash — fast"),),
        efforts=("low", "high", "max"),
        default_effort="max",
        compact_window="524288",
        codex_url=None,
        key_env="MINIMAX_API_KEY",
        key_alt_envs=(),
        key_url="https://platform.minimax.io",
        context_window=1048576,
    ),
}

PROVIDER_ORDER = ("deepseek", "kimi", "glm", "minimax")
PROVIDER_KEYS = (*PROVIDER_ORDER, "custom")


def make_custom_provider(
    title: str,
    anthropic_url: str,
    openai_url: str,
    main_model: str,
    fast_model: str,
    effort: str,
    context_window: int,
) -> Provider:
    efforts = (effort,) if effort else ()
    return Provider(
        key="custom",
        title=title or "Custom",
        anthropic_url=anthropic_url.rstrip("/"),
        openai_url=openai_url.rstrip("/") if openai_url else None,
        main_models=(Model(main_model, main_model),),
        fast_models=(Model(fast_model or main_model, fast_model or main_model),),
        efforts=efforts,
        default_effort=effort,
        compact_window=None,
        codex_url=None,
        key_env="STARTAGENT_API_KEY",
        key_alt_envs=(),
        key_url="",
        context_window=context_window,
        custom=True,
    )


def strip_context_suffix(model: str) -> str:
    """Claude Code uses [1m] to request a 1M context; other clients do not."""
    return model.replace("[1m]", "")


def agent_supports(provider: Provider, agent: str) -> bool:
    entry = AGENTS.get(agent)
    return bool(entry and entry.supports(provider))


@dataclass(frozen=True)
class Agent:
    key: str
    title: str
    command: str
    install: tuple[str, ...]
    supports: Callable[[Provider], bool]


AGENTS: dict[str, Agent] = {
    "claude": Agent(
        key="claude",
        title="Claude Code",
        command="claude",
        install=("npm", "install", "-g", "@anthropic-ai/claude-code"),
        supports=lambda p: True,
    ),
    "codex": Agent(
        key="codex",
        title="Codex",
        command="codex",
        install=("npm", "install", "-g", "@openai/codex"),
        supports=lambda p: p.codex_url is not None,
    ),
    # The shell is used only for the official curl|bash installer below.
    # It is a trusted constant from this registry, not user input, and the
    # wizard shows the command and asks before running it.
    "opencode": Agent(
        key="opencode",
        title="OpenCode",
        command="opencode",
        install=("sh", "-c", "curl -fsSL https://opencode.ai/install | bash"),
        supports=lambda p: True,
    ),
    "qwen": Agent(
        key="qwen",
        title="Qwen Code",
        command="qwen",
        install=("npm", "install", "-g", "@qwen-code/qwen-code"),
        supports=lambda p: p.openai_url is not None,
    ),
}

AGENT_ORDER = ("claude", "codex", "opencode", "qwen")


def install_text(agent: Agent) -> str:
    """Human-readable install command (unwrap the sh -c pipeline)."""
    if agent.install[:2] == ("sh", "-c"):
        return agent.install[2]
    return shlex.join(agent.install)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def config_dir() -> Path:
    return _env_path("STARTAGENT_CONFIG_DIR", DEFAULT_CONFIG_DIR)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _write_private(path: Path, text: str) -> None:
    try:
        ensure_dir(path.parent)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp"
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise StartagentError(f"cannot write {path}: {exc}") from exc


def save_provider_config(
    directory: Path,
    provider: Provider,
    model: str,
    fast_model: str,
    effort: str,
    key: str,
) -> Path:
    ensure_dir(directory)
    data = {
        "version": 1,
        "key": provider.key,
        "title": provider.title,
        "anthropic_url": provider.anthropic_url,
        "openai_url": provider.openai_url,
        "model": model,
        "fast_model": fast_model,
        "effort": effort,
        "compact_window": provider.compact_window,
        "codex_url": provider.codex_url,
        "context_window": provider.context_window,
        "key_env": provider.key_env,
        "key_url": provider.key_url,
        "custom": provider.custom,
    }
    path = directory / f"{provider.key}.json"
    _write_private(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _write_private(directory / f"{provider.key}.key", key)
    return path


def save_current(directory: Path, provider_key: str, agent_key: str) -> None:
    ensure_dir(directory)
    _write_private(
        directory / "current.json",
        json.dumps({"provider": provider_key, "agent": agent_key}, indent=2) + "\n",
    )


def load_current(directory: Path) -> dict | None:
    path = directory / "current.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StartagentError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or "provider" not in data:
        return None
    return data


def load_provider_config(directory: Path, provider_key: str) -> dict | None:
    path = directory / f"{provider_key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StartagentError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise StartagentError(f"invalid config: {path}")
    return data


def read_key(directory: Path, provider_key: str) -> str | None:
    path = directory / f"{provider_key}.key"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def find_legacy_key(
    directory: Path, provider_key: str, legacy_dir: Path | None = None
) -> str | None:
    if provider_key != "deepseek":
        return None
    legacy = legacy_dir or LEGACY_CONFIG_DIR
    for candidate in (legacy / "api_key", legacy / "deepseek" / "api_key"):
        if candidate.is_file():
            value = candidate.read_text(encoding="utf-8").strip()
            return value or None
    return None


def provider_from_config(cfg: dict) -> Provider:
    key = str(cfg.get("key", ""))
    if key in PROVIDERS:
        return PROVIDERS[key]
    if cfg.get("custom"):
        return make_custom_provider(
            title=str(cfg.get("title") or "Custom"),
            anthropic_url=str(cfg.get("anthropic_url") or ""),
            openai_url=str(cfg.get("openai_url") or ""),
            main_model=str(cfg.get("model") or ""),
            fast_model=str(cfg.get("fast_model") or cfg.get("model") or ""),
            effort=str(cfg.get("effort") or ""),
            context_window=int(cfg.get("context_window") or 200000),
        )
    raise StartagentError(f"unknown provider in config: {key!r}")


@dataclass
class RunContext:
    directory: Path
    provider: Provider
    cfg: dict
    key: str


def load_run_context(directory: Path, provider_key: str | None = None) -> RunContext:
    current = load_current(directory) or {}
    key = provider_key or current.get("provider")
    if not key:
        raise StartagentError("no configuration found — run: startagent --setup")
    if key not in PROVIDER_KEYS:
        raise StartagentError(f"unknown provider: {key}")
    cfg = load_provider_config(directory, key)
    if not cfg:
        raise StartagentError(
            f"no configuration for {key} — run: startagent --setup --provider {key}"
        )
    secret = read_key(directory, key)
    if not secret:
        raise StartagentError(
            f"no API key for {key} — run: startagent --setup --provider {key}"
        )
    return RunContext(directory, provider_from_config(cfg), cfg, secret)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _join_v1(url: str) -> str:
    base = url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def opencode_config(provider: Provider, model: str, fast: str) -> dict:
    models: dict[str, dict] = {}
    for model_id in (model, fast):
        bare = strip_context_suffix(model_id)
        models[bare] = {"name": model_id}
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"anthropic/{strip_context_suffix(model)}",
        "provider": {
            "anthropic": {
                "options": {
                    "baseURL": _join_v1(provider.anthropic_url),
                    "apiKey": "{env:STARTAGENT_API_KEY}",
                },
                "models": models,
            }
        },
    }


def codex_model_id(provider: Provider, model: str) -> str:
    for entry in provider.main_models:
        if entry.id == model:
            return entry.codex_id or strip_context_suffix(entry.id)
    return strip_context_suffix(model)


def _codex_model_entry(provider: Provider, model_id: str) -> dict:
    efforts = provider.efforts or ("high",)
    default = provider.default_effort if provider.default_effort in efforts else efforts[0]
    return {
        "slug": model_id,
        "display_name": model_id,
        "description": provider.title,
        "default_reasoning_level": default,
        "supported_reasoning_levels": [
            {"effort": effort, "description": effort.title()} for effort in efforts
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "base_instructions": "",
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "context_window": provider.context_window,
        "max_context_window": provider.context_window,
        "effective_context_window_percent": 95,
        "supports_parallel_tool_calls": True,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
    }


def codex_config(
    provider: Provider, model: str, effort: str, catalog_path: Path | None = None
) -> tuple[str, dict]:
    if not provider.codex_url:
        raise StartagentError(f"{provider.title} does not support Codex")
    model_id = codex_model_id(provider, model)
    provider_id = "zai" if provider.key == "glm" else provider.key
    lines = [
        f"model = {json.dumps(model_id)}",
        f"model_provider = {json.dumps(provider_id)}",
        'preferred_auth_method = "apikey"',
    ]
    if catalog_path is not None:
        lines.append(f"model_catalog_json = {json.dumps(str(catalog_path))}")
    if effort:
        lines.append(f"model_reasoning_effort = {json.dumps(effort)}")
    lines.extend(
        [
            "",
            f"[model_providers.{provider_id}]",
            f"name = {json.dumps(provider.title)}",
            f"base_url = {json.dumps(provider.codex_url)}",
            'env_key = "STARTAGENT_API_KEY"',
            'wire_api = "responses"',
            "",
        ]
    )
    return "\n".join(lines), {"models": [_codex_model_entry(provider, model_id)]}


def check_request(provider: Provider, model: str, key: str) -> tuple[str, dict, dict]:
    url = _join_v1(provider.anthropic_url) + "/messages"
    headers = {
        "Authorization": f"Bearer {key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "User-Agent": f"startagent/{VERSION}",
    }
    body = {
        "model": strip_context_suffix(model),
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    return url, headers, body


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def claude_env(
    provider: Provider,
    model: str,
    fast: str,
    key: str,
    effort: str,
    compact_window: str | int | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ANTHROPIC_BASE_URL": provider.anthropic_url,
            "ANTHROPIC_AUTH_TOKEN": key,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": fast,
            "ANTHROPIC_DEFAULT_FABLE_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": fast,
        }
    )
    env.pop("CLAUDE_CODE_EFFORT_LEVEL", None)
    if effort:
        env["CLAUDE_CODE_EFFORT_LEVEL"] = effort
    env.pop("CLAUDE_CODE_AUTO_COMPACT_WINDOW", None)
    if compact_window:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(compact_window)
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def qwen_env(provider: Provider, model: str, key: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = key
    env["OPENAI_BASE_URL"] = provider.openai_url or ""
    env["OPENAI_MODEL"] = strip_context_suffix(model)
    return env


def require_agent(agent_key: str) -> Agent:
    agent = AGENTS[agent_key]
    if not shutil.which(agent.command):
        raise StartagentError(
            f"{agent.title} is not installed. Install: {install_text(agent)}"
        )
    return agent


def ensure_agent_installed(agent_key: str) -> bool:
    """Offer to install the selected agent; return True when it is usable."""
    agent = AGENTS[agent_key]
    if shutil.which(agent.command):
        return True
    command = install_text(agent)
    print(f"\n{agent.title} is not installed.")
    print(f"Install command: {command}")
    if not ask_yes(f"Install {agent.title} now?", True):
        print(f"Skipped. Install it later with: {command}")
        return False
    print(f"Installing {agent.title}...")
    try:
        result = subprocess.run(agent.install)
    except FileNotFoundError as exc:
        print(f"Could not run the installer: {exc}")
        print(f"Run manually: {command}")
        return False
    if result.returncode != 0 or not shutil.which(agent.command):
        print(f"Installation failed. Run manually: {command}")
        return False
    print(f"{agent.title} installed.")
    return True


def _on_path(directory: Path) -> bool:
    target = directory.expanduser().resolve()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and Path(entry).expanduser().resolve() == target:
            return True
    return False


def install_self(target_dir: Path | None = None) -> Path | None:
    """Last setup step: install this script as `startagent` on PATH."""
    source = Path(__file__).resolve()
    if target_dir is None:
        target_dir = _env_path(
            "STARTAGENT_INSTALL_DIR", Path.home() / ".local" / "bin"
        )
    target = target_dir / "startagent"

    if target.exists() and target.resolve() == source:
        print(f"startagent is already installed at {target}")
        return target

    current = shutil.which("startagent")
    if current and Path(current).resolve() == source:
        print(f"startagent is already available as {current}")
        return Path(current)

    print("\nInstall startagent itself so you can run it as `startagent`.")
    print(f"Source: {source}")
    print(f"Target: {target}")
    if not ask_yes("Install startagent now?", True):
        print("Skipped. You can run the script from its current location.")
        return None

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        tmp = target_dir / (target.name + ".tmp")
        shutil.copyfile(source, tmp)
        tmp.chmod(0o755)
        os.replace(tmp, target)
    except OSError as exc:
        print(f"Could not install startagent: {exc}")
        print(
            f"Run manually: cp {shlex.quote(str(source))} "
            f"{shlex.quote(str(target))}"
        )
        return None

    print(f"Installed: {target}")
    if not _on_path(target_dir):
        print(f"Add {target_dir} to PATH to run `startagent` from anywhere.")
    return target


def write_codex_files(home: Path, provider: Provider, model: str, effort: str) -> None:
    ensure_dir(home)
    catalog = home / "models.json"
    toml_text, models = codex_config(provider, model, effort, catalog)
    _write_private(home / "config.toml", toml_text)
    _write_private(catalog, json.dumps(models, indent=2, ensure_ascii=False) + "\n")


def launch(agent_key: str, ctx: RunContext, args: list[str]) -> None:
    provider, cfg, key = ctx.provider, ctx.cfg, ctx.key
    model = str(cfg.get("model") or "")
    fast = str(cfg.get("fast_model") or model)
    effort = str(cfg.get("effort") or "")

    if not agent_supports(provider, agent_key):
        raise StartagentError(
            f"{provider.title} does not support {AGENTS[agent_key].title}"
        )

    if agent_key == "claude":
        require_agent("claude")
        env = claude_env(
            provider, model, fast, key, effort, cfg.get("compact_window")
        )
        _exec("claude", args, env)
    elif agent_key == "codex":
        require_agent("codex")
        home = ctx.directory / "agents" / "codex" / provider.key
        write_codex_files(home, provider, model, effort)
        env = os.environ.copy()
        env["CODEX_HOME"] = str(home)
        env["STARTAGENT_API_KEY"] = key
        _exec("codex", args, env)
    elif agent_key == "opencode":
        require_agent("opencode")
        home = ctx.directory / "agents" / "opencode" / provider.key
        ensure_dir(home)
        path = home / "opencode.json"
        _write_private(
            path,
            json.dumps(opencode_config(provider, model, fast), indent=2, ensure_ascii=False)
            + "\n",
        )
        env = os.environ.copy()
        env["OPENCODE_CONFIG"] = str(path)
        env["STARTAGENT_API_KEY"] = key
        _exec("opencode", args, env)
    elif agent_key == "qwen":
        require_agent("qwen")
        _exec("qwen", ["--auth-type", "openai", *args], qwen_env(provider, model, key))
    else:
        raise StartagentError(f"unknown agent: {agent_key}")


def _exec(command: str, args: list[str], env: dict[str, str]) -> None:
    try:
        # exec replaces the process without flushing Python's buffers, which
        # would lose setup messages when stdout is redirected.
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe(command, [command, *args], env)
    except FileNotFoundError as exc:
        raise StartagentError(f"command not found: {command}") from exc


# ---------------------------------------------------------------------------
# Interactive setup
# ---------------------------------------------------------------------------


def _read(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError as exc:
        raise StartagentError("input ended unexpectedly") from exc


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = _read(f"{prompt}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return str(default)
        print("Please enter a value.")


def ask_yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _read(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "д", "да")


def choose(prompt: str, options: list[tuple[str, str]], default: int = 1) -> str:
    print(prompt)
    for index, (_, label) in enumerate(options, 1):
        marker = "  (default)" if index == default else ""
        print(f"  {index}) {label}{marker}")
    while True:
        raw = _read(f"Select [{default}]: ").strip()
        if not raw:
            raw = str(default)
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print("Enter a number from the list.")


def choose_provider(default_key: str | None = None) -> str:
    keys = list(PROVIDER_KEYS)
    options = []
    for key in PROVIDER_ORDER:
        provider = PROVIDERS[key]
        options.append((key, f"{provider.title} — {provider.key_url}"))
    options.append(("custom", "Custom Anthropic-compatible endpoint"))
    default = keys.index(default_key) + 1 if default_key in keys else 1
    return choose("Provider:", options, default)


def validate_agent_override(provider: Provider, agent_key: str | None) -> None:
    if agent_key and (agent_key not in AGENTS or not agent_supports(provider, agent_key)):
        raise StartagentError(f"{provider.title} does not support {agent_key}")


def choose_agent(provider: Provider, default_key: str | None = None) -> str:
    options: list[tuple[str, str]] = []
    keys: list[str] = []
    for key in AGENT_ORDER:
        agent = AGENTS[key]
        if not agent.supports(provider):
            continue
        installed = shutil.which(agent.command) is not None
        label = (
            agent.title
            if installed
            else f"{agent.title} — not installed ({install_text(agent)})"
        )
        options.append((key, label))
        keys.append(key)
    if not options:
        raise StartagentError(f"no agents support {provider.title}")
    default = keys.index(default_key) + 1 if default_key in keys else 1
    return choose("Agent:", options, default)


def choose_model(
    provider: Provider, kind: str, default_id: str | None = None, main_model: str | None = None
) -> str:
    models = provider.main_models if kind == "main" else provider.fast_models
    options = [(entry.id, f"{entry.id} — {entry.label}") for entry in models]
    if kind == "fast" and main_model:
        options.append((main_model, f"same as main ({main_model})"))
    options.append(("__custom__", "custom model ID"))
    default = 1
    for index, (value, _) in enumerate(options, 1):
        if value == default_id:
            default = index
    prompt = "Main model:" if kind == "main" else "Fast model (background/subagents):"
    value = choose(prompt, options, default)
    if value == "__custom__":
        value = ask("Model ID")
    return value


def choose_effort(provider: Provider, default: str | None = None) -> str:
    if not provider.efforts:
        return ""
    options = [(effort, effort) for effort in provider.efforts]
    if default in provider.efforts:
        index = provider.efforts.index(default) + 1
    elif provider.default_effort in provider.efforts:
        index = provider.efforts.index(provider.default_effort) + 1
    else:
        index = 1
    return choose("Reasoning effort:", options, index)


def resolve_key(provider: Provider, directory: Path) -> str:
    existing = read_key(directory, provider.key)
    if existing and ask_yes(f"Keep the existing {provider.title} key?", True):
        return existing

    for env_name in (provider.key_env, *provider.key_alt_envs):
        value = (os.environ.get(env_name) or "").strip()
        if value and ask_yes(f"Use ${env_name}?", True):
            return value

    legacy = find_legacy_key(directory, provider.key)
    if legacy and ask_yes("Found an old DeepSeek key. Use it?", True):
        return legacy

    if provider.key_url:
        print(f"Get a key here: {provider.key_url}")
    while True:
        try:
            secret = getpass.getpass(f"{provider.title} API key (hidden): ").strip()
        except EOFError as exc:
            raise StartagentError("API key was not provided") from exc
        if secret:
            return secret
        print("The key cannot be empty.")


def run_setup(directory: Path, provider_key: str | None = None, agent_key: str | None = None) -> None:
    current = load_current(directory) or {}
    existing_keys = list(PROVIDER_KEYS)
    if provider_key not in existing_keys:
        provider_key = None
    if not provider_key:
        provider_key = choose_provider(current.get("provider"))

    if provider_key == "custom":
        existing = load_provider_config(directory, "custom") or {}
        print("\nCustom Anthropic-compatible endpoint")
        title = ask("Name", existing.get("title") or "Custom")
        anthropic_url = ask(
            "Anthropic base URL (used for Claude Code/OpenCode)",
            existing.get("anthropic_url"),
        )
        openai_url = ask(
            "OpenAI-compatible base URL (blank = no Qwen Code support)",
            existing.get("openai_url") or "",
        )
        raw_context = ask(
            "Context window in tokens", str(existing.get("context_window") or 200000)
        )
        context = int(raw_context) if raw_context.isdigit() else 200000
        provider = make_custom_provider(
            title, anthropic_url, openai_url, "model", "model", "", context
        )
        validate_agent_override(provider, agent_key)
        key = resolve_key(provider, directory)
        agent_key = choose_agent(provider, agent_key or current.get("agent"))
        agent_ready = ensure_agent_installed(agent_key)
        model = ask("Main model ID", existing.get("model"))
        fast = ask("Fast model ID", existing.get("fast_model") or model)
        effort = ask(
            "Reasoning effort (low/high/max, blank = none)",
            existing.get("effort") or "",
        )
        provider = make_custom_provider(
            title, anthropic_url, openai_url, model, fast, effort, context
        )
    else:
        provider = PROVIDERS[provider_key]
        existing = load_provider_config(directory, provider.key) or {}
        validate_agent_override(provider, agent_key)
        key = resolve_key(provider, directory)
        agent_key = choose_agent(provider, agent_key or current.get("agent"))
        agent_ready = ensure_agent_installed(agent_key)
        model = choose_model(provider, "main", existing.get("model"))
        fast = choose_model(
            provider, "fast", existing.get("fast_model"), main_model=model
        )
        effort = choose_effort(provider, existing.get("effort"))

    save_provider_config(directory, provider, model, fast, effort, key)
    save_current(directory, provider.key, agent_key)
    print(f"Saved: {provider.title} · {AGENTS[agent_key].title} · {model}")
    install_self()
    if not agent_ready:
        raise StartagentError(
            f"{AGENTS[agent_key].title} is not installed. Install it with: "
            f"{install_text(AGENTS[agent_key])}"
        )
    ctx = RunContext(directory, provider, load_provider_config(directory, provider.key) or {}, key)
    launch(agent_key, ctx, [])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(directory: Path) -> int:
    current = load_current(directory)
    if not current:
        print("No configuration yet. Run: startagent --setup")
        return 0
    agent_key = current.get("agent", "")
    agent_entry = AGENTS.get(agent_key)
    print(f"Agent: {agent_entry.title if agent_entry else agent_key}")
    print(f"Provider: {current.get('provider')}")
    for key in PROVIDER_KEYS:
        cfg = load_provider_config(directory, key)
        if not cfg:
            continue
        marker = " *" if key == current.get("provider") else ""
        key_state = "key present" if read_key(directory, key) else "no key"
        print(
            f"  {key}{marker}: {cfg.get('model')} · {cfg.get('fast_model')} · {key_state}"
        )
    return 0


def cmd_print(directory: Path, provider_key: str | None = None) -> int:
    ctx = load_run_context(directory, provider_key)
    cfg = ctx.cfg
    env = claude_env(
        ctx.provider,
        str(cfg.get("model") or ""),
        str(cfg.get("fast_model") or cfg.get("model") or ""),
        "",
        str(cfg.get("effort") or ""),
        cfg.get("compact_window"),
    )
    names = (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "CLAUDE_CODE_EFFORT_LEVEL",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    )
    for name in names:
        value = env.get(name)
        if value:
            print(f"export {name}={shlex.quote(value)}")
    key_path = directory / f"{ctx.provider.key}.key"
    print(f'export ANTHROPIC_AUTH_TOKEN="$(cat {shlex.quote(str(key_path))})"')
    print("unset ANTHROPIC_API_KEY")
    return 0


def cmd_check(directory: Path, provider_key: str | None = None) -> int:
    ctx = load_run_context(directory, provider_key)
    url, headers, body = check_request(ctx.provider, str(ctx.cfg.get("model") or ""), ctx.key)
    detail = ""
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            code = response.status
    except urllib.error.HTTPError as exc:
        code = exc.code
        try:
            detail = exc.read().decode(errors="replace")[:300]
        except (OSError, http.client.HTTPException):
            detail = ""
    except ValueError as exc:
        raise StartagentError(f"invalid endpoint: {exc}") from exc
    except (OSError, http.client.HTTPException) as exc:
        reason = getattr(exc, "reason", exc)
        raise StartagentError(f"network error: {reason}") from exc

    model = strip_context_suffix(str(ctx.cfg.get("model") or ""))
    if code == 200:
        print(f"OK: {ctx.provider.title} · {model} (HTTP 200)")
        return 0
    suffix = f"\n{detail}" if detail else ""
    if code in (401, 403):
        raise StartagentError(f"{ctx.provider.title}: key rejected (HTTP {code}){suffix}")
    if code == 404:
        raise StartagentError(f"{ctx.provider.title}: endpoint not found (HTTP 404) {url}{suffix}")
    raise StartagentError(f"{ctx.provider.title}: unexpected response (HTTP {code}){suffix}")


def usage() -> None:
    print(
        f"""startagent {VERSION} — run coding agents without vendor lock

Usage:
  startagent                      run the saved agent
  startagent --setup              reconfigure
  startagent --provider NAME      switch provider and run
  startagent --agent NAME         switch agent and run
  startagent claude|codex|opencode|qwen [args...]
  startagent --check              verify key and endpoint
  startagent --print              print export lines for eval
  startagent --list               show saved configuration
  startagent <command> [args...]  run any command with the provider environment

Providers: {", ".join(PROVIDER_KEYS)}
Agents:    {", ".join(AGENT_ORDER)}
Config:    {DEFAULT_CONFIG_DIR} (override: STARTAGENT_CONFIG_DIR)
"""
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    provider: str | None = None
    agent: str | None = None
    do_setup = False
    do_check = False
    do_print = False
    do_list = False
    rest: list[str] = []

    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-h", "--help"):
            usage()
            return 0
        if arg in ("-V", "--version"):
            print(f"startagent {VERSION}")
            return 0
        if arg == "--setup":
            do_setup = True
            index += 1
        elif arg in ("--provider", "-p"):
            if index + 1 >= len(args):
                raise StartagentError("--provider requires a name")
            provider = args[index + 1]
            index += 2
        elif arg in ("--agent", "--client", "-a"):
            if index + 1 >= len(args):
                raise StartagentError("--agent requires a name")
            agent = args[index + 1]
            index += 2
        elif arg == "--check":
            do_check = True
            index += 1
        elif arg in ("--print", "--print-env"):
            do_print = True
            index += 1
        elif arg in ("--list", "-l"):
            do_list = True
            index += 1
        elif arg.startswith("-"):
            raise StartagentError(f"unknown flag: {arg}")
        else:
            rest = args[index:]
            break

    directory = config_dir()
    if provider is not None and provider not in PROVIDER_KEYS:
        raise StartagentError(f"unknown provider: {provider}")
    if agent is not None and agent not in AGENTS:
        raise StartagentError(f"unknown agent: {agent}")

    if do_list:
        return cmd_list(directory)
    if do_print:
        return cmd_print(directory, provider)
    if do_check:
        return cmd_check(directory, provider)

    current = load_current(directory)
    if do_setup or current is None:
        run_setup(directory, provider, agent)
        return 0

    target_provider = provider or str(current.get("provider") or "")
    target_agent = agent or str(current.get("agent") or "claude")
    if not target_provider:
        raise StartagentError("no provider in config — run startagent --setup")
    if target_agent not in AGENTS:
        raise StartagentError(
            f"unknown agent in config: {target_agent} — run startagent --setup"
        )

    if provider and (
        not load_provider_config(directory, provider) or not read_key(directory, provider)
    ):
        run_setup(directory, provider, agent)
        return 0

    ctx = load_run_context(directory, target_provider)
    if not agent_supports(ctx.provider, target_agent):
        raise StartagentError(
            f"{ctx.provider.title} does not support {AGENTS[target_agent].title}; "
            "use --agent or run startagent --setup"
        )
    if target_provider != current.get("provider") or target_agent != current.get("agent"):
        require_agent(target_agent)
        save_current(directory, target_provider, target_agent)

    if rest and rest[0] in AGENTS:
        target = rest[0]
        if not agent_supports(ctx.provider, target):
            raise StartagentError(
                f"{ctx.provider.title} does not support {AGENTS[target].title}"
            )
        launch(target, ctx, rest[1:])
        return 0

    if rest:
        env = claude_env(
            ctx.provider,
            str(ctx.cfg.get("model") or ""),
            str(ctx.cfg.get("fast_model") or ctx.cfg.get("model") or ""),
            ctx.key,
            str(ctx.cfg.get("effort") or ""),
            ctx.cfg.get("compact_window"),
        )
        _exec(rest[0], rest[1:], env)
        return 0

    launch(target_agent, ctx, [])
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    except StartagentError as exc:
        print(f"startagent: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
