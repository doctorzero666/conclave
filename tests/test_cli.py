"""Test CLI: --prompt-file / --output / config show escaping."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from conclave import cli as cli_module
from conclave.cli import main
from conclave.config import Config, ProviderConfig
from conclave.protocol import (
    DeliberationResult,
    DeliberationRound,
    ProviderResponse,
    RoundType,
)


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """强制 load_config 返回可控的 Config，避免读用户的 ~/.conclave/config.yaml。"""
    fake_provider = ProviderConfig(
        name="mock", provider_type="cli", model="mock-model",
        executable="echo", args_template=["{prompt}"],
    )
    cfg = Config(
        default_providers=["mock"],
        judge_provider="mock",
        providers={"mock": fake_provider},
    )
    monkeypatch.setattr(cli_module, "load_config", lambda: cfg)
    return cfg


@pytest.fixture
def stub_engine(monkeypatch):
    """Stub DeliberationEngine 以返回一个可预测的 DeliberationResult。"""
    class _FakeEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            self.providers = providers

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="RECEIVED:" + prompt,
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _FakeEngine)


@pytest.fixture
def stub_provider(monkeypatch):
    """Stub create_provider 以避免依赖真实 CLI 可执行文件。"""
    from conclave.protocol import ProviderResponse as _PR
    from conclave.providers.base import BaseProvider, ProviderCapabilities

    class _FakeProvider(BaseProvider):
        def __init__(self, config):
            super().__init__(name=config.name, model=config.model)

        def capabilities(self):
            return ProviderCapabilities(supports_streaming=False, models=[self.model])

        def invoke(self, prompt, system_prompt=None, timeout=180):
            return _PR(provider_name=self.name, model=self.model,
                       text="ok", elapsed_ms=1)

    monkeypatch.setattr(cli_module, "create_provider", _FakeProvider)


# ─── --prompt-file ─────────────────────────────────────────────────

def test_run_requires_prompt_source(isolated_config, stub_provider, stub_engine):
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--rounds", "1"])
    assert result.exit_code != 0
    assert "PROMPT" in result.output or "prompt-file" in result.output


def test_run_prompt_and_prompt_file_mutually_exclusive(
    isolated_config, stub_provider, stub_engine, tmp_path
):
    pf = tmp_path / "p.txt"
    pf.write_text("hi", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["run", "hello", "--prompt-file", str(pf),
                                  "--rounds", "1"])
    assert result.exit_code != 0
    assert "互斥" in result.output


def test_run_prompt_file_missing(isolated_config, stub_provider, stub_engine,
                                 tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--prompt-file",
                                  str(tmp_path / "nope.txt"), "--rounds", "1"])
    assert result.exit_code == 1
    assert "不存在" in result.output


def test_run_prompt_file_is_directory(isolated_config, stub_provider,
                                      stub_engine, tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--prompt-file", str(d),
                                  "--rounds", "1"])
    assert result.exit_code == 1
    assert "目录" in result.output


def test_run_prompt_file_empty(isolated_config, stub_provider, stub_engine,
                               tmp_path):
    pf = tmp_path / "empty.txt"
    pf.write_text("   \n\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--prompt-file", str(pf),
                                  "--rounds", "1"])
    assert result.exit_code == 1
    assert "为空" in result.output


def test_run_prompt_file_non_utf8(isolated_config, stub_provider, stub_engine,
                                  tmp_path):
    pf = tmp_path / "bin.txt"
    # 无效 UTF-8 起始字节
    pf.write_bytes(b"\xff\xfe\xfd bad bytes")
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--prompt-file", str(pf),
                                  "--rounds", "1"])
    assert result.exit_code == 1
    assert "UTF-8" in result.output


def test_run_prompt_file_success_json_output(
    isolated_config, stub_provider, stub_engine, tmp_path
):
    pf = tmp_path / "spec.txt"
    # 超过 2000 字符以验证 --output 不截断
    long_prompt = "多行 prompt 内容 " * 500
    pf.write_text(long_prompt, encoding="utf-8")
    out = tmp_path / "result.json"

    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "--prompt-file", str(pf), "--output", str(out),
        "--format", "json", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()

    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["prompt"] == long_prompt
    # 落盘不截断: 完整 "RECEIVED:" + long_prompt 均保留
    resp_text = data["rounds"][0]["responses"][0]["text"]
    assert resp_text == "RECEIVED:" + long_prompt
    assert len(resp_text) > 2000


def test_run_markdown_output_ansi_free(
    isolated_config, stub_provider, stub_engine, tmp_path
):
    out = tmp_path / "result.md"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "hello world", "--output", str(out),
        "--format", "markdown", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "\x1b[" not in text  # ANSI-free
    assert "hello world" in text


def test_run_providers_short_alias_dash_p(
    isolated_config, stub_provider, stub_engine
):
    """`-p` 必须是 `--providers` 的短别名，兼容 v2 用法。"""
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "hello", "-p", "mock", "--rounds", "1"]
    )
    assert result.exit_code == 0, result.output


def test_strip_ansi_helper_removes_csi_osc_and_controls():
    from conclave.cli import _strip_ansi

    # CSI (color/cursor), OSC (title), and stray control chars
    dirty = (
        "hello\x1b[31m red\x1b[0m world"
        "\x1b]0;window title\x07 tail"
        "\x07\x1b\x08 keep\n and\t tab"
    )
    clean = _strip_ansi(dirty)
    assert "\x1b[" not in clean
    assert "\x1b]" not in clean
    assert "\x1b" not in clean
    assert "\x07" not in clean
    assert "\x08" not in clean
    # Readable text preserved
    assert "hello" in clean and "red" in clean and "world" in clean
    assert "tail" in clean and "keep" in clean and "tab" in clean
    # \n and \t preserved
    assert "\n" in clean and "\t" in clean


def test_run_markdown_output_strips_ansi_in_prompt_and_response(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """Engine 返回带 ANSI/OSC/control 的 prompt/text → markdown 文件必须干净。"""
    class _AnsiEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            dirty_text = (
                "answer \x1b[31mred\x1b[0m body"
                "\x1b]0;evil title\x07 tail\x07"
            )
            resp = ProviderResponse(
                provider_name="mock\x1b[1m", model="mock-\x1b[32mmodel\x1b[0m",
                text=dirty_text,
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                # dirty prompt too
                prompt="user \x1b[7mprompt\x1b[0m here",
                rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _AnsiEngine)

    out = tmp_path / "dirty.md"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "markdown", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    # No ANSI CSI or OSC introducers or stray control chars
    assert "\x1b[" not in text
    assert "\x1b]" not in text
    assert "\x1b" not in text
    assert "\x07" not in text
    # Readable text survives strip
    assert "prompt" in text
    assert "answer" in text
    assert "red" in text
    assert "body" in text
    assert "tail" in text
    assert "mock" in text


def test_run_json_output_preserves_ansi_lossless(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """JSON 落盘保持 lossless — 不 strip ANSI (仅 markdown strip)。"""
    import json

    class _AnsiEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="ANSI:\x1b[31mred\x1b[0m",
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt="p\x1b[1m!", rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _AnsiEngine)

    out = tmp_path / "raw.json"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "json", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["prompt"] == "p\x1b[1m!"
    assert data["rounds"][0]["responses"][0]["text"] == "ANSI:\x1b[31mred\x1b[0m"


def test_run_markdown_output_survives_malformed_rich_markup(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """provider 返回带 malformed Rich markup (如 [/bold]) 不能让 --output 写文件失败。"""
    class _MarkupEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            evil = "answer with [/bold] and [/red] closers with no opener"
            resp = ProviderResponse(
                provider_name="mock[/x]", model="mock-[/model]",
                text=evil,
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _MarkupEngine)

    out = tmp_path / "markup.md"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "markdown", "--rounds", "1",
    ])
    # 文件必须写成，不能因 MarkupError 中断
    assert out.exists(), result.output
    text = out.read_text(encoding="utf-8")
    assert "[/bold]" in text
    assert "[/red]" in text
    # 且 CLI 未以 MarkupError 崩溃
    assert result.exit_code == 0, result.output


def test_markdown_output_preserves_partial_text_when_error(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """provider 同时给出 text + error (如 truncated) 时 markdown 仍保留已收集文本。"""
    class _PartialEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="partial answer body",
                error="truncated",
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            # 该 response 的 resp.ok 为 False (有 error)
            assert not resp.ok
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _PartialEngine)

    out = tmp_path / "partial.md"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "markdown", "--rounds", "1",
    ])
    assert out.exists(), result.output
    text = out.read_text(encoding="utf-8")
    # FAIL header 保留
    assert "FAIL" in text
    assert "truncated" in text
    # 已收集的文本必须保留 (核心断言)
    assert "partial answer body" in text


def test_run_prompt_file_same_as_output_rejected(
    isolated_config, stub_provider, stub_engine, tmp_path
):
    pf = tmp_path / "shared.txt"
    pf.write_text("hello", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "--prompt-file", str(pf), "--output", str(pf),
        "--rounds", "1",
    ])
    assert result.exit_code == 1
    assert "同一路径" in result.output or "同一" in result.output


def test_run_output_write_failure_is_warning_not_error(
    isolated_config, stub_provider, stub_engine, tmp_path
):
    # 写到一个不存在的父目录 → OSError → 只 warning，不改变审议退出码
    bad_out = tmp_path / "no_such_dir" / "out.json"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "hello", "--output", str(bad_out),
        "--format", "json", "--rounds", "1",
    ])
    # 审议本身成功 → exit code 0
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output or "无法写入" in result.output


# ─── C1 control stripping / Unicode write failure ────────────────

def test_strip_ansi_helper_removes_c1_controls():
    """8-bit C1 controls (U+0080–U+009F, incl. CSI U+009B / OSC U+009D) must be stripped."""
    from conclave.cli import _strip_ansi

    dirty = "before31m mid0;title tail end"
    clean = _strip_ansi(dirty)
    for cp in ("", "", "", "", ""):
        assert cp not in clean
    assert "before" in clean
    assert "mid" in clean
    assert "tail" in clean
    assert "end" in clean


def test_run_markdown_output_strips_c1_controls(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """Engine 返回含 C1 控制字符时 markdown 落盘必须清除。"""
    class _C1Engine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="body 31m red 0;evil tail",
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _C1Engine)

    out = tmp_path / "c1.md"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "markdown", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    for cp in ("", "", ""):
        assert cp not in text
    assert "body" in text and "red" in text and "tail" in text


def test_run_json_output_preserves_c1_lossless(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """JSON 落盘保持 lossless — C1 控制字符不 strip。"""
    import json

    class _C1Engine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="rawCSIOSC",
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _C1Engine)

    out = tmp_path / "c1.json"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "json", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["rounds"][0]["responses"][0]["text"] == "rawCSIOSC"


def test_run_output_unicode_error_is_warning_not_error_markdown(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """text 含 unpaired surrogate → markdown 落盘 UnicodeEncodeError 只 warning，不改退出码。"""
    class _SurrogateEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            # unpaired surrogate — utf-8 encode 会抛 UnicodeEncodeError
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="broken \ud800 surrogate",
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _SurrogateEngine)

    out = tmp_path / "surrogate.md"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "markdown", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output or "无法写入" in result.output


def test_run_output_unicode_error_is_warning_not_error_json(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """text 含 unpaired surrogate → JSON 落盘 UnicodeEncodeError 只 warning，不改退出码。"""
    class _SurrogateEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="broken \ud800 surrogate",
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _SurrogateEngine)

    out = tmp_path / "surrogate.json"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "json", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output or "无法写入" in result.output


# ─── Atomic --output writes: old content preserved on failure ────

def test_run_output_atomic_preserves_old_on_unicode_failure_markdown(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """写入失败 (unpaired surrogate) 时目标文件原内容保留，无遗留临时文件。"""
    class _SurrogateEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="broken \ud800 surrogate",
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _SurrogateEngine)

    out = tmp_path / "existing.md"
    out.write_text("OLD RESULT", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "markdown", "--rounds", "1",
    ])
    # 审议 exit code 未变
    assert result.exit_code == 0, result.output
    # 原文件内容保留
    assert out.read_text(encoding="utf-8") == "OLD RESULT"
    # 无遗留临时文件 (只剩 existing.md)
    leftovers = [
        p.name for p in tmp_path.iterdir()
        if p.name != "existing.md"
    ]
    assert leftovers == [], f"unexpected leftover temp files: {leftovers}"


def test_run_output_atomic_preserves_old_on_unicode_failure_json(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """JSON 写入失败也保留旧文件、清理临时文件。"""
    class _SurrogateEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="broken \ud800 surrogate",
                elapsed_ms=10, round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_],
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _SurrogateEngine)

    out = tmp_path / "existing.json"
    out.write_text("OLD RESULT", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "json", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == "OLD RESULT"
    leftovers = [
        p.name for p in tmp_path.iterdir()
        if p.name != "existing.json"
    ]
    assert leftovers == [], f"unexpected leftover temp files: {leftovers}"


# ─── config show Rich markup escaping ──────────────────────────────

def test_config_show_escapes_provider_names(monkeypatch):
    """provider 名 [claude] 不能被 Rich 当作 markup 标签解析。"""
    tricky = ProviderConfig(
        name="openai-gpt5", provider_type="cli", model="gpt-5.6-sol",
    )
    cfg = Config(
        default_providers=["openai-gpt5"],
        judge_provider="openai-gpt5",
        providers={"openai-gpt5": tricky},
    )
    monkeypatch.setattr(cli_module, "load_config", lambda: cfg)

    runner = CliRunner()
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0, result.output
    # 名字应字面显示（Rich 会把 \[foo] 渲染为 [foo]）
    assert "[openai-gpt5]" in result.output
    assert "gpt-5.6-sol" in result.output


# ─── Claude CLI preset ────────────────────────────────────────────

def test_claude_preset_includes_dash_p():
    from conclave.providers.cli import CLI_PRESETS
    assert CLI_PRESETS["claude"][:1] == ["-p"]
    assert "{prompt}" in CLI_PRESETS["claude"]


def test_default_claude_config_uses_dash_p():
    from conclave.config import _default_providers
    providers = _default_providers()
    assert "claude" in providers
    assert providers["claude"].args_template == ["-p", "{prompt}"]


# ─── Config migration ─────────────────────────────────────────────

def _write_config_yaml(path: Path, providers: dict):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"providers": providers}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_load_config_migrates_old_claude_args_template(monkeypatch, tmp_path):
    from conclave import config as config_module

    cfg_path = tmp_path / "config.yaml"
    _write_config_yaml(cfg_path, {
        "claude": {
            "provider_type": "cli",
            "model": "claude-sonnet-4",
            "executable": "claude",
            "args_template": ["{prompt}"],
        }
    })
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)

    cfg = config_module.load_config()
    assert cfg.providers["claude"].args_template == ["-p", "{prompt}"]


def test_load_config_preserves_custom_claude_args_template(monkeypatch, tmp_path):
    from conclave import config as config_module

    cfg_path = tmp_path / "config.yaml"
    custom = ["--model", "sonnet", "-p", "{prompt}"]
    _write_config_yaml(cfg_path, {
        "claude": {
            "provider_type": "cli",
            "model": "claude-sonnet-4",
            "executable": "claude",
            "args_template": custom,
        }
    })
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)

    cfg = config_module.load_config()
    assert cfg.providers["claude"].args_template == custom


def test_load_config_preserves_custom_wrapper_executable(monkeypatch, tmp_path):
    """provider 名叫 claude 但 executable 是自定义 wrapper — 保留 args_template。

    wrapper 可能不支持 -p，强行改写会破坏用户显式配置。
    """
    from conclave import config as config_module

    cfg_path = tmp_path / "config.yaml"
    _write_config_yaml(cfg_path, {
        "claude": {
            "provider_type": "cli",
            "model": "claude-sonnet-4",
            "executable": "/usr/local/bin/my-claude-wrapper",
            "args_template": ["{prompt}"],
        }
    })
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)

    cfg = config_module.load_config()
    # executable 非 "claude" → 不迁移，保留用户显式配置
    assert cfg.providers["claude"].args_template == ["{prompt}"]


# ─── --output JSON full serialization ─────────────────────────────

def test_output_json_includes_full_metadata_and_synthesis(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """--output JSON 落盘必须包含 total_cost_usd/failed_providers/synthesis 全字段。"""
    from conclave.protocol import (
        DeliberationResult,
        DeliberationRound,
        ProviderResponse,
        RoundType,
        SynthesisConsensus,
        SynthesisDivergence,
        SynthesisInsight,
        SynthesisReport,
        SynthesisStatus,
    )

    class _RichEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp_ok = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="answer", finish_reason="stop",
                elapsed_ms=42, round_num=1, round_type=RoundType.INITIAL,
                timestamp="2026-07-22T00:00:00",
            )
            resp_fail = ProviderResponse(
                provider_name="broken", model="broken-model",
                text="", finish_reason="error", error="boom",
                elapsed_ms=5, round_num=1, round_type=RoundType.INITIAL,
                timestamp="2026-07-22T00:00:01",
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL,
                responses=[resp_ok, resp_fail],
                started_at="2026-07-22T00:00:00",
                finished_at="2026-07-22T00:00:02",
            )
            syn = SynthesisReport(
                consensus=[SynthesisConsensus(
                    point="p1", agreed_by=["mock"], confidence="high",
                )],
                divergences=[SynthesisDivergence(
                    point="d1", positions={"mock": "A"}, critical=True,
                )],
                insights=[SynthesisInsight(
                    point="i1", provider_name="mock", category="risk",
                )],
                executive_summary="sum",
                judge_model="judge-model",
                status=SynthesisStatus.SUCCESS,
                error=None,
                total_cost_usd=0.42,
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_], synthesis=syn,
                total_duration_ms=1234, total_cost_usd=1.23,
                provider_count=2, round_count=1,
                failed_providers=["broken"],
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _RichEngine)

    out = tmp_path / "full.json"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "hello", "--output", str(out),
        "--format", "json", "--rounds", "1",
    ])
    # 有 failed_providers 但至少一个 ok → exit code 2
    assert result.exit_code in (0, 2), result.output
    assert out.exists()

    import json
    data = json.loads(out.read_text(encoding="utf-8"))

    # top-level metadata
    assert data["prompt"] == "hello"
    assert data["total_duration_ms"] == 1234
    assert data["total_cost_usd"] == 1.23
    assert data["provider_count"] == 2
    assert data["round_count"] == 1
    assert data["failed_providers"] == ["broken"]

    # round fields
    rd = data["rounds"][0]
    assert rd["round_num"] == 1
    assert rd["round_type"] == "initial"
    assert rd["started_at"] == "2026-07-22T00:00:00"
    assert rd["finished_at"] == "2026-07-22T00:00:02"

    # response fields
    r0 = rd["responses"][0]
    for key in ("provider_name", "model", "text", "finish_reason", "error",
                "elapsed_ms", "round_num", "round_type", "timestamp"):
        assert key in r0, f"missing {key}"
    assert r0["provider_name"] == "mock"
    assert r0["finish_reason"] == "stop"
    assert r0["round_type"] == "initial"
    assert r0["timestamp"] == "2026-07-22T00:00:00"

    # synthesis full fields
    syn = data["synthesis"]
    for key in ("consensus", "divergences", "insights", "executive_summary",
                "judge_model", "status", "error", "total_cost_usd"):
        assert key in syn, f"missing synthesis.{key}"
    assert syn["executive_summary"] == "sum"
    assert syn["judge_model"] == "judge-model"
    assert syn["status"] == "success"
    assert syn["total_cost_usd"] == 0.42
    assert syn["consensus"][0]["point"] == "p1"
    assert syn["consensus"][0]["agreed_by"] == ["mock"]
    assert syn["divergences"][0]["critical"] is True
    assert syn["divergences"][0]["positions"] == {"mock": "A"}
    assert syn["insights"][0]["category"] == "risk"


# ─── Non-successful synthesis markdown rendering ──────────────────

def test_markdown_output_includes_fallback_synthesis(
    isolated_config, stub_provider, monkeypatch, tmp_path
):
    """FALLBACK 合成报告 (含 executive_summary + error) 必须落盘，不能静默丢。"""
    from conclave.protocol import (
        DeliberationResult,
        DeliberationRound,
        ProviderResponse,
        RoundType,
        SynthesisConsensus,
        SynthesisReport,
        SynthesisStatus,
    )

    class _FallbackEngine:
        def __init__(self, providers, judge_provider=None, config=None):
            pass

        def run(self, prompt, rounds, timeout, max_cost, on_round_callback,
                synthesis_enabled):
            resp = ProviderResponse(
                provider_name="mock", model="mock-model",
                text="body", elapsed_ms=10,
                round_num=1, round_type=RoundType.INITIAL,
            )
            round_ = DeliberationRound(
                round_num=1, round_type=RoundType.INITIAL, responses=[resp],
            )
            syn = SynthesisReport(
                consensus=[SynthesisConsensus(
                    point="partial-agreement",
                    agreed_by=["mock"], confidence="medium",
                )],
                executive_summary="best-effort fallback summary",
                judge_model="judge-fallback",
                status=SynthesisStatus.FALLBACK,
                error="judge parse failed, used heuristic",
            )
            return DeliberationResult(
                prompt=prompt, rounds=[round_], synthesis=syn,
                total_duration_ms=10, provider_count=1, round_count=1,
            )

    monkeypatch.setattr(cli_module, "DeliberationEngine", _FallbackEngine)

    out = tmp_path / "fallback.md"
    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "seed", "--output", str(out),
        "--format", "markdown", "--rounds", "1",
    ])
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "fallback" in text  # status token
    assert "best-effort fallback summary" in text
    assert "judge parse failed, used heuristic" in text
    assert "partial-agreement" in text


# ─── Path identity via realpath / samefile ────────────────────────

def test_run_prompt_file_symlink_as_output_rejected(
    isolated_config, stub_provider, stub_engine, tmp_path
):
    """symlink 指向 prompt-file 作为 --output 时必须拒绝 (P2)。"""
    import os as _os

    pf = tmp_path / "prompt.txt"
    pf.write_text("hi", encoding="utf-8")
    link = tmp_path / "link.txt"
    _os.symlink(pf, link)

    runner = CliRunner()
    result = runner.invoke(main, [
        "run", "--prompt-file", str(pf), "--output", str(link),
        "--rounds", "1",
    ])
    assert result.exit_code == 1
    assert "同一路径" in result.output or "同一" in result.output
