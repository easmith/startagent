#!/usr/bin/env python3
"""Tests for startagent. Run with: python3 -m unittest -v test_startagent.py"""
from __future__ import annotations

import json
import os
import runpy
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "startagent"

FAKE_AGENT = r"""#!/bin/sh
echo "CMD=$(basename "$0")"
echo "BASE=${ANTHROPIC_BASE_URL:-}"
echo "MODEL=${ANTHROPIC_MODEL:-}"
echo "FAST=${ANTHROPIC_DEFAULT_HAIKU_MODEL:-}"
echo "TOKEN=${ANTHROPIC_AUTH_TOKEN:+set}"
echo "OPENAI_BASE=${OPENAI_BASE_URL:-}"
echo "OPENAI_MODEL=${OPENAI_MODEL:-}"
echo "CODEX_HOME=${CODEX_HOME:-}"
echo "ARGS=$*"
"""


def load_module():
    return runpy.run_path(str(SCRIPT), run_name="startagent_test")


class FakeBin:
    def __init__(self, root: Path, names: list[str]):
        self.bin = root / "bin"
        self.bin.mkdir(exist_ok=True)
        for name in names:
            path = self.bin / name
            path.write_text(FAKE_AGENT)
            path.chmod(0o755)

    def env(self, **extra: str) -> dict[str, str]:
        # Keep the test hermetic: only the fake agents plus basic system
        # utilities are visible, so a real agent installed on the machine
        # cannot change the outcome.
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join([str(self.bin), "/bin", "/usr/bin"])
        env.update(extra)
        return env


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def test_strip_context_suffix(self):
        strip = self.m["strip_context_suffix"]
        self.assertEqual(strip("deepseek-flash[1m]"), "deepseek-flash")
        self.assertEqual(strip("glm-5.3"), "glm-5.3")
        self.assertEqual(strip("MiniMax-M3.1-Flash[1m]"), "MiniMax-M3.1-Flash")

    def test_provider_registry(self):
        providers = self.m["PROVIDERS"]
        self.assertEqual(list(providers), ["deepseek", "kimi", "glm", "minimax"])
        self.assertEqual(
            providers["deepseek"].anthropic_url, "https://api.deepseek.com/anthropic"
        )
        self.assertEqual(providers["deepseek"].openai_url, "https://api.deepseek.com/v1")
        self.assertEqual(providers["minimax"].openai_url, "https://api.minimax.io/v1")

    def test_codex_support(self):
        providers = self.m["PROVIDERS"]
        supports = self.m["agent_supports"]
        self.assertTrue(supports(providers["deepseek"], "codex"))
        self.assertTrue(supports(providers["glm"], "codex"))
        self.assertFalse(supports(providers["kimi"], "codex"))
        self.assertFalse(supports(providers["minimax"], "codex"))
        self.assertTrue(supports(providers["kimi"], "qwen"))

    def test_custom_provider(self):
        make = self.m["make_custom_provider"]
        p = make(
            title="My GW",
            anthropic_url="https://gw.example/anthropic",
            openai_url="https://gw.example/v1",
            main_model="my-model[1m]",
            fast_model="my-fast",
            effort="high",
            context_window=200000,
        )
        self.assertTrue(p.custom)
        self.assertEqual(p.main_models[0].id, "my-model[1m]")
        self.assertIsNone(p.codex_url)
        self.assertEqual(p.openai_url, "https://gw.example/v1")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.m = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_and_permissions(self):
        p = self.m["PROVIDERS"]["deepseek"]
        self.m["save_provider_config"](
            self.dir, p, "deepseek-flash[1m]", "deepseek-flash", "max", "sk-test"
        )
        self.m["save_current"](self.dir, "deepseek", "claude")

        loaded = self.m["load_provider_config"](self.dir, "deepseek")
        self.assertEqual(loaded["model"], "deepseek-flash[1m]")
        self.assertEqual(loaded["fast_model"], "deepseek-flash")
        self.assertEqual(loaded["effort"], "max")
        self.assertEqual(self.m["read_key"](self.dir, "deepseek"), "sk-test")
        self.assertEqual(
            stat.S_IMODE((self.dir / "deepseek.key").stat().st_mode), 0o600
        )
        self.assertEqual(
            stat.S_IMODE((self.dir / "deepseek.json").stat().st_mode), 0o600
        )
        self.assertEqual(
            self.m["load_current"](self.dir),
            {"provider": "deepseek", "agent": "claude"},
        )

    def test_legacy_key(self):
        legacy = self.dir / "legacy"
        (legacy / "deepseek").mkdir(parents=True)
        (legacy / "deepseek" / "api_key").write_text("sk-legacy\n")
        self.assertEqual(
            self.m["find_legacy_key"](self.dir, "deepseek", legacy), "sk-legacy"
        )
        self.assertIsNone(self.m["find_legacy_key"](self.dir, "kimi", legacy))

    def test_empty_config_dir_env_falls_back(self):
        with mock.patch.dict(os.environ, {"STARTAGENT_CONFIG_DIR": ""}):
            self.assertEqual(self.m["config_dir"](), self.m["DEFAULT_CONFIG_DIR"])

    def test_config_dir_expands_user(self):
        with mock.patch.dict(
            os.environ, {"STARTAGENT_CONFIG_DIR": "~/startagent-test"}
        ):
            self.assertEqual(
                self.m["config_dir"](), Path.home() / "startagent-test"
            )

    def test_empty_xdg_config_home_falls_back(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": ""}):
            m = load_module()
        self.assertEqual(
            m["DEFAULT_CONFIG_DIR"], Path.home() / ".config" / "startagent"
        )

    def test_write_private_keeps_mode_when_chmod_fails(self):
        target = self.dir / "secret"
        with mock.patch.object(Path, "chmod", side_effect=OSError("denied")):
            self.m["_write_private"](target, "value")
        self.assertEqual(target.read_text(encoding="utf-8"), "value")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)

    def test_write_private_reports_unwritable_path(self):
        blocked = self.dir / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        with self.assertRaises(self.m["StartagentError"]):
            self.m["_write_private"](blocked / "file", "value")

    def test_config_roundtrip_is_utf8(self):
        custom = self.m["make_custom_provider"](
            "Мой шлюз",
            "https://gw.example/anthropic",
            "",
            "модель[1m]",
            "",
            "",
            200000,
        )
        def ascii_when_unset(encoding, stacklevel=2):
            return encoding if encoding is not None else "ascii"

        with mock.patch("io.text_encoding", side_effect=ascii_when_unset):
            self.m["save_provider_config"](
                self.dir, custom, "модель[1m]", "", "", "sk-test"
            )
            loaded = self.m["load_provider_config"](self.dir, "custom")
        self.assertEqual(loaded["title"], "Мой шлюз")


class KeyResolutionTests(unittest.TestCase):
    def setUp(self):
        self.m = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.m["resolve_key"].__globals__["find_legacy_key"] = (
            lambda directory, provider_key, legacy_dir=None: None
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_whitespace_only_env_key_is_ignored(self):
        self.m["resolve_key"].__globals__["ask_yes"] = (
            lambda prompt, default=True: True
        )
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "   "}):
            with mock.patch("getpass.getpass", return_value="sk-real"):
                key = self.m["resolve_key"](
                    self.m["PROVIDERS"]["deepseek"], self.dir
                )
        self.assertEqual(key, "sk-real")


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def test_opencode_config(self):
        p = self.m["PROVIDERS"]["deepseek"]
        cfg = self.m["opencode_config"](p, "deepseek-flash[1m]", "deepseek-flash")
        self.assertEqual(cfg["model"], "anthropic/deepseek-flash")
        prov = cfg["provider"]["anthropic"]
        self.assertEqual(
            prov["options"]["baseURL"], "https://api.deepseek.com/anthropic/v1"
        )
        self.assertEqual(prov["options"]["apiKey"], "{env:STARTAGENT_API_KEY}")
        self.assertIn("deepseek-flash", prov["models"])

    def test_opencode_does_not_double_v1(self):
        make = self.m["make_custom_provider"]
        p = make(
            title="X",
            anthropic_url="https://gw.example/anthropic/v1",
            openai_url="",
            main_model="m",
            fast_model="m",
            effort="",
            context_window=200000,
        )
        cfg = self.m["opencode_config"](p, "m", "m")
        self.assertEqual(
            cfg["provider"]["anthropic"]["options"]["baseURL"],
            "https://gw.example/anthropic/v1",
        )

    def test_codex_config(self):
        p = self.m["PROVIDERS"]["deepseek"]
        toml_text, models = self.m["codex_config"](
            p, "deepseek-flash[1m]", "max"
        )
        data = tomllib.loads(toml_text)
        self.assertEqual(data["model"], "deepseek-v4-flash")
        self.assertEqual(data["model_provider"], "deepseek")
        self.assertEqual(data["model_reasoning_effort"], "max")
        provider = data["model_providers"]["deepseek"]
        self.assertEqual(provider["base_url"], "https://api.deepseek.com/")
        self.assertEqual(provider["wire_api"], "responses")
        self.assertEqual(provider["env_key"], "STARTAGENT_API_KEY")
        entry = models["models"][0]
        self.assertEqual(entry["slug"], "deepseek-v4-flash")
        self.assertEqual(entry["context_window"], 1048576)

    def test_check_request(self):
        p = self.m["PROVIDERS"]["deepseek"]
        url, headers, body = self.m["check_request"](
            p, "deepseek-flash[1m]", "sk-test"
        )
        self.assertEqual(url, "https://api.deepseek.com/anthropic/v1/messages")
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertTrue(headers["User-Agent"].startswith("startagent/"))
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["max_tokens"], 1)


class EnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def test_claude_env_drops_stale_settings(self):
        p = self.m["PROVIDERS"]["deepseek"]
        saved = {
            name: os.environ.get(name)
            for name in ("CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CODE_AUTO_COMPACT_WINDOW")
        }
        os.environ["CLAUDE_CODE_EFFORT_LEVEL"] = "stale"
        os.environ["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "1"
        try:
            env = self.m["claude_env"](p, "m", "f", "k", "", None)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.assertNotIn("CLAUDE_CODE_EFFORT_LEVEL", env)
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", env)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.m = load_module()
        self.cfg = self.root / "cfg"
        p = self.m["PROVIDERS"]["deepseek"]
        self.m["save_provider_config"](
            self.cfg, p, "deepseek-flash[1m]", "deepseek-flash", "max", "sk-test"
        )
        self.m["save_current"](self.cfg, "deepseek", "claude")
        self.fake = FakeBin(self.root, ["claude", "qwen", "opencode"])

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args: str, input: str | None = None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            env=self.fake.env(STARTAGENT_CONFIG_DIR=str(self.cfg)),
            input=input,
            text=True,
            capture_output=True,
        )

    def test_claude_env(self):
        r = self.run_cli("claude")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("BASE=https://api.deepseek.com/anthropic", r.stdout)
        self.assertIn("MODEL=deepseek-flash[1m]", r.stdout)
        self.assertIn("FAST=deepseek-flash", r.stdout)
        self.assertIn("TOKEN=set", r.stdout)

    def test_qwen_env(self):
        self.m["save_current"](self.cfg, "deepseek", "qwen")
        r = self.run_cli("qwen")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OPENAI_BASE=https://api.deepseek.com/v1", r.stdout)
        self.assertIn("OPENAI_MODEL=deepseek-flash", r.stdout)
        self.assertIn("--auth-type openai", r.stdout)

    def test_opencode_config_written(self):
        self.m["save_current"](self.cfg, "deepseek", "opencode")
        r = self.run_cli("opencode")
        self.assertEqual(r.returncode, 0, r.stderr)
        path = self.cfg / "agents" / "opencode" / "deepseek" / "opencode.json"
        self.assertTrue(path.exists())
        data = json.loads(path.read_text())
        self.assertEqual(data["model"], "anthropic/deepseek-flash")

    def test_codex_missing_hint(self):
        self.m["save_current"](self.cfg, "deepseek", "codex")
        r = self.run_cli("codex")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("npm install -g @openai/codex", r.stderr)

    def test_print_hides_key(self):
        r = self.run_cli("--print")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("sk-test", r.stdout)
        self.assertIn("$(cat", r.stdout)

    def test_unsupported_agent_switch_does_not_save(self):
        p = self.m["PROVIDERS"]["minimax"]
        self.m["save_provider_config"](
            self.cfg, p, "MiniMax-M3[1m]", "MiniMax-M3.1-Flash", "max", "sk-mini"
        )
        self.m["save_current"](self.cfg, "minimax", "claude")
        r = self.run_cli("--agent", "codex")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.m["load_current"](self.cfg)["agent"], "claude")

    def test_failed_agent_switch_does_not_save(self):
        r = self.run_cli("--agent", "codex")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.m["load_current"](self.cfg)["agent"], "claude")

    def test_print_rejects_provider_outside_config_dir(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "probe.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "key": "deepseek",
                    "title": "Probe",
                    "anthropic_url": "https://api.deepseek.com/anthropic",
                    "openai_url": None,
                    "model": "probe-model",
                    "fast_model": "probe-fast",
                    "effort": "max",
                    "compact_window": None,
                    "codex_url": None,
                    "context_window": 1048576,
                    "key_env": "DEEPSEEK_API_KEY",
                    "key_url": "",
                    "custom": False,
                }
            ),
            encoding="utf-8",
        )
        (outside / "probe.key").write_text("sk-probe", encoding="utf-8")
        r = self.run_cli("--print", "--provider", "../outside/probe")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unknown provider", r.stderr)

    def test_check_reports_invalid_url_without_traceback(self):
        custom = self.m["make_custom_provider"](
            "X", "gw.example/anthropic", "", "m", "m", "", 200000
        )
        self.m["save_provider_config"](self.cfg, custom, "m", "", "", "sk-test")
        self.m["save_current"](self.cfg, "custom", "claude")
        r = self.run_cli("--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("invalid endpoint", r.stderr)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.m = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join([str(self.bin), "/bin", "/usr/bin"])

    def tearDown(self):
        os.environ["PATH"] = self.old_path
        self.tmp.cleanup()

    def fake_agent(self, install: str):
        return self.m["Agent"](
            key="fake",
            title="Fake Agent",
            command="fake-agent",
            install=("sh", "-c", install),
            supports=lambda p: True,
        )

    def test_installs_missing_agent(self):
        target = self.bin / "fake-agent"
        command = (
            f"printf '#!/bin/sh\\n' > {shlex.quote(str(target))} "
            f"&& chmod +x {shlex.quote(str(target))}"
        )
        self.m["AGENTS"]["fake"] = self.fake_agent(command)
        self.m["ensure_agent_installed"].__globals__["ask_yes"] = (
            lambda prompt, default=True: True
        )
        self.assertTrue(self.m["ensure_agent_installed"]("fake"))
        self.assertTrue(target.exists())
        self.assertIsNotNone(shutil.which("fake-agent"))

    def test_declined_install_returns_false(self):
        target = self.bin / "fake-agent"
        self.m["AGENTS"]["fake"] = self.fake_agent(
            f"touch {shlex.quote(str(target))}"
        )
        self.m["ensure_agent_installed"].__globals__["ask_yes"] = (
            lambda prompt, default=True: False
        )
        self.assertFalse(self.m["ensure_agent_installed"]("fake"))
        self.assertFalse(target.exists())


class SelfInstallTests(unittest.TestCase):
    def setUp(self):
        self.m = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = "/bin:/usr/bin"

    def tearDown(self):
        os.environ["PATH"] = self.old_path
        self.tmp.cleanup()

    def test_installs_script(self):
        target_dir = self.root / "bin"
        self.m["install_self"].__globals__["ask_yes"] = (
            lambda prompt, default=True: True
        )
        target = self.m["install_self"](target_dir)
        self.assertEqual(target, target_dir / "startagent")
        self.assertTrue(target.exists())
        self.assertEqual(target.read_bytes(), SCRIPT.read_bytes())
        self.assertTrue(os.access(target, os.X_OK))

    def test_declined_install(self):
        target_dir = self.root / "bin"
        self.m["install_self"].__globals__["ask_yes"] = (
            lambda prompt, default=True: False
        )
        self.assertIsNone(self.m["install_self"](target_dir))
        self.assertFalse((target_dir / "startagent").exists())

    def test_skips_when_target_is_source(self):
        self.assertEqual(self.m["install_self"](SCRIPT.parent), SCRIPT)


class WizardTests(unittest.TestCase):
    def test_first_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "cfg"
            fake = FakeBin(root, ["claude"])
            env = fake.env(
                STARTAGENT_CONFIG_DIR=str(cfg),
                STARTAGENT_INSTALL_DIR=str(root / "install"),
                DEEPSEEK_API_KEY="sk-env",
            )
            answers = "1\ny\n1\n1\n1\n3\nn\n"
            r = subprocess.run(
                [sys.executable, str(SCRIPT)],
                input=answers,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((cfg / "deepseek.json").exists())
            self.assertEqual((cfg / "deepseek.key").read_text(), "sk-env")
            self.assertIn("MODEL=deepseek-flash[1m]", r.stdout)
            self.assertFalse((root / "install" / "startagent").exists())

    def test_first_run_installs_script(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "cfg"
            install = root / "install"
            fake = FakeBin(root, ["claude"])
            env = fake.env(
                STARTAGENT_CONFIG_DIR=str(cfg),
                STARTAGENT_INSTALL_DIR=str(install),
                DEEPSEEK_API_KEY="sk-env",
            )
            answers = "1\ny\n1\n1\n1\n3\ny\n"
            r = subprocess.run(
                [sys.executable, str(SCRIPT)],
                input=answers,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((install / "startagent").exists())
            self.assertIn("Installed:", r.stdout)
            self.assertIn("MODEL=deepseek-flash[1m]", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
