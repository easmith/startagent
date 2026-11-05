# startagent

A universal launcher for coding agents without vendor lock. Point
**Claude Code**, **Codex**, **OpenCode** or **Qwen Code** at any Anthropic-
or OpenAI-compatible provider: DeepSeek, Kimi/Moonshot, Z.ai GLM, MiniMax,
or your own endpoint.

One Python file, standard library only, no build step. Your agent's own
global config is never overwritten.

## Why

A coding agent should not be tied to the vendor that happens to own your API
key. `startagent` keeps one small config per provider and launches the agent
with the right environment for that provider. Switching from DeepSeek to
Kimi is one command:

```sh
startagent --provider kimi
```

## Requirements

- Python 3.11 or newer
- At least one agent CLI on `PATH`: `claude`, `codex`, `opencode`, `qwen`
- An API key from a supported provider

## Install

The package is published as `startagent-cli`; the command it installs is
`startagent`.

Run without installing (needs [uv](https://docs.astral.sh/uv/)):

```sh
uvx startagent-cli
```

Install as a command:

```sh
uv tool install startagent-cli
# or
pipx install startagent-cli
# or
pip install startagent-cli
```

From a clone (installs the local checkout into `~/.local/bin`):

```sh
git clone https://github.com/easmith/startagent.git startagent
cd startagent
./install.sh                 # or: ./install.sh /custom/bin
```

Or copy `startagent.py` to any directory on your `PATH` as `startagent` and
make it executable:

```sh
cp startagent.py ~/.local/bin/startagent
chmod +x ~/.local/bin/startagent
```

Make sure `~/.local/bin` is on your `PATH`.

## Quick start

```sh
startagent
```

The first run asks for:

1. the provider;
2. the API key (hidden input; an existing `DEEPSEEK_API_KEY` or
   `~/.config/deepseek/api_key` is offered as a default);
3. the agent to run; if it is missing, the wizard offers to run its
   official installer right there;
4. the main model, the fast model, and the reasoning effort;
5. the last step: install `startagent` itself to `~/.local/bin/startagent`.
   The command is shown before it runs; `STARTAGENT_INSTALL_DIR` changes the
   target directory.

Later runs launch the saved configuration without asking anything.
Reconfigure with `startagent --setup`.

## Commands

```
startagent                      run the saved agent
startagent --setup              reconfigure
startagent --provider NAME      switch provider and run
startagent --agent NAME         switch agent and run
startagent claude|codex|opencode|qwen [args...]
startagent --check              verify the key and endpoint (one-token request)
startagent --print              print export lines for eval
startagent --list               show the saved configuration
startagent <command> [args...]  run any command with the provider environment
```

`--print` is useful if you want the environment in your current shell:

```sh
eval "$(startagent --print)"
```

The key is printed as `$(cat ...)`, not in plain text.

## Providers

| Provider | Anthropic endpoint | Default models |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/anthropic` | `deepseek-flash[1m]`, `deepseek-v4-pro[1m]` |
| Kimi / Moonshot | `https://api.moonshot.ai/anthropic` | `kimi-k3[1m]`, `kimi-k2.7-code` |
| Z.ai GLM | `https://api.z.ai/api/anthropic` | `glm-5.3[1m]`, `glm-5.3-flash` |
| MiniMax | `https://api.minimax.io/anthropic` | `MiniMax-M3[1m]`, `MiniMax-M3.1-Flash[1m]` |
| Custom | your URL | your model IDs |

Every provider also has an OpenAI-compatible endpoint used by Qwen Code.
The `[1m]` suffix is a Claude Code convention for the 1M context window;
other agents receive the model ID without it.

Provider endpoints and model IDs change often. The wizard always offers a
"custom model ID" option, and the "Custom" provider accepts any
Anthropic-compatible base URL.

## Agents

| Agent | How it is launched |
|---|---|
| Claude Code | `ANTHROPIC_*` environment variables |
| Codex | isolated `CODEX_HOME` with generated `config.toml` + `models.json` |
| OpenCode | isolated `OPENCODE_CONFIG` JSON, Anthropic provider override |
| Qwen Code | `OPENAI_*` environment variables |

If the selected agent is not installed, the wizard shows its official
install command and asks before running it. Nothing is installed silently.
Declining is fine: the config is still saved, and the wizard tells you the
command to run later.

Codex is offered only for providers that expose the OpenAI Responses API
(DeepSeek and Z.ai GLM). The generated `CODEX_HOME` lives under
`~/.config/startagent/agents/codex/<provider>/`; your `~/.codex` is not
modified. DeepSeek requires Codex 0.144 or newer.

Qwen Code is offered only when the provider has an OpenAI-compatible base
URL.

## Configuration

```
~/.config/startagent/
  current.json                 provider and agent currently selected
  deepseek.json                provider settings (no secret)
  deepseek.key                 API key, mode 600
  agents/
    codex/deepseek/            generated Codex config
    opencode/deepseek/         generated OpenCode config
```

Override the config directory with `STARTAGENT_CONFIG_DIR` (useful for tests).
`STARTAGENT_INSTALL_DIR` changes where the wizard installs the script itself
(default: `~/.local/bin`). The non-interactive equivalent is
`./install.sh [directory]`.


## Security

- API keys are stored in `<provider>.key` with mode `600`; the config
  directory is mode `700`.
- `startagent --print` never prints the key itself.
- No telemetry. The only network call is the provider you configured
  (`--check` sends a one-token request).
- Agent configs are generated per provider, so switching providers does
  not touch the agent's global configuration.

## Limitations

- Gemini CLI, GitHub Copilot CLI and Amazon Q Developer use their own
  protocols and are not supported.
- Kimi/Moonshot and MiniMax do not expose the OpenAI Responses API, so
  Codex is not offered for them.
- This is an unofficial launcher; provider APIs and model IDs may change.

## Development

```sh
python3 startagent.py --version   # run from the checkout
python3 -m unittest -v test_startagent.py
```

The tests use a temporary config directory and fake agent executables, so
they do not need API keys or installed agents. Real end-to-end checks are
manual: `startagent --check`, then launch each installed agent.

## License

MIT. See [LICENSE](LICENSE).
