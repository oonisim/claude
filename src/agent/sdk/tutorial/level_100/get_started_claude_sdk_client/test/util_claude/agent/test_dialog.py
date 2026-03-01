"""Tests for util_claude.agent.dialog."""

import pytest
from unittest.mock import MagicMock

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from util_claude.agent.dialog import (
    _TOOL_RESULT_PREVIEW_LINES,
    _truncate_lines,
    print_agent_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock(spec_class: type, **attrs: object) -> MagicMock:
    """Return a MagicMock that passes isinstance checks for spec_class."""
    m = MagicMock(spec=spec_class)
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


def _long_text(n_lines: int) -> str:
    """Return a string with n_lines lines numbered from 1."""
    return "\n".join(f"line {i}" for i in range(1, n_lines + 1))


# ---------------------------------------------------------------------------
# Tests — _truncate_lines
# ---------------------------------------------------------------------------

class TestTruncateLines:
    """_truncate_lines returns text unchanged or truncated with a marker."""

    def test_fewer_lines_than_max_unchanged(self) -> None:
        text = "a\nb\nc"
        assert _truncate_lines(text, 10) == text

    def test_exactly_max_lines_unchanged(self) -> None:
        text = _long_text(10)
        assert _truncate_lines(text, 10) == text

    def test_more_lines_truncated(self) -> None:
        text = _long_text(15)
        result = _truncate_lines(text, 10)
        result_lines = result.splitlines()
        # First 10 source lines must be present verbatim.
        assert result_lines[:10] == text.splitlines()[:10]

    def test_marker_shows_omitted_count(self) -> None:
        text = _long_text(15)
        result = _truncate_lines(text, 10)
        assert "(+5 more lines)" in result

    def test_single_line_below_max_unchanged(self) -> None:
        text = "only one line"
        assert _truncate_lines(text, 5) == text

    def test_empty_string_unchanged(self) -> None:
        assert _truncate_lines("", 5) == ""

    def test_max_lines_one(self) -> None:
        text = "first\nsecond\nthird"
        result = _truncate_lines(text, 1)
        assert result.startswith("first")
        assert "(+2 more lines)" in result


# ---------------------------------------------------------------------------
# Tests — print_agent_message: AssistantMessage
# ---------------------------------------------------------------------------

class TestPrintAssistantMessage:
    """print_agent_message handles AssistantMessage content blocks."""

    def test_text_block_prints_claude_header(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        block = _mock(TextBlock, text="Hello, world!")
        msg = _mock(AssistantMessage, content=[block])
        print_agent_message(msg)
        assert "[Claude]" in capsys.readouterr().out

    def test_text_block_prints_text(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        block = _mock(TextBlock, text="Hello, world!")
        msg = _mock(AssistantMessage, content=[block])
        print_agent_message(msg)
        assert "Hello, world!" in capsys.readouterr().out

    def test_thinking_block_prints_placeholder(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        block = _mock(ThinkingBlock, thinking="internal", signature="sig")
        msg = _mock(AssistantMessage, content=[block])
        print_agent_message(msg)
        assert "[Thinking]" in capsys.readouterr().out

    def test_tool_use_block_prints_tool_name(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        block = _mock(ToolUseBlock, name="bash", input={"command": "ls"})
        msg = _mock(AssistantMessage, content=[block])
        print_agent_message(msg)
        assert "[Tool] bash" in capsys.readouterr().out

    def test_tool_use_block_prints_args(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        block = _mock(ToolUseBlock, name="bash", input={"command": "ls -la"})
        msg = _mock(AssistantMessage, content=[block])
        print_agent_message(msg)
        assert "ls -la" in capsys.readouterr().out

    def test_multiple_content_blocks_all_printed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        text_block = _mock(TextBlock, text="Plan:")
        tool_block = _mock(ToolUseBlock, name="read", input={"file_path": "f.py"})
        msg = _mock(AssistantMessage, content=[text_block, tool_block])
        print_agent_message(msg)
        out = capsys.readouterr().out
        assert "Plan:" in out
        assert "[Tool] read" in out


# ---------------------------------------------------------------------------
# Tests — print_agent_message: UserMessage
# ---------------------------------------------------------------------------

class TestPrintUserMessage:
    """print_agent_message handles UserMessage content blocks."""

    def test_tool_result_ok_status(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        block = _mock(ToolResultBlock, tool_use_id="t1", content="output", is_error=False)
        msg = _mock(UserMessage, content=[block])
        print_agent_message(msg)
        assert "[Tool Result/OK]" in capsys.readouterr().out

    def test_tool_result_error_status(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        block = _mock(ToolResultBlock, tool_use_id="t1", content="boom", is_error=True)
        msg = _mock(UserMessage, content=[block])
        print_agent_message(msg)
        assert "[Tool Result/ERROR]" in capsys.readouterr().out

    def test_tool_result_content_printed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        block = _mock(ToolResultBlock, tool_use_id="t1", content="42\n", is_error=False)
        msg = _mock(UserMessage, content=[block])
        print_agent_message(msg)
        assert "42" in capsys.readouterr().out

    def test_tool_result_long_content_truncated(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # Content longer than _TOOL_RESULT_PREVIEW_LINES must be truncated.
        long_content = _long_text(_TOOL_RESULT_PREVIEW_LINES + 5)
        block = _mock(
            ToolResultBlock, tool_use_id="t1", content=long_content, is_error=False
        )
        msg = _mock(UserMessage, content=[block])
        print_agent_message(msg)
        out = capsys.readouterr().out
        assert "(+5 more lines)" in out

    def test_tool_result_short_content_not_truncated(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        short_content = _long_text(_TOOL_RESULT_PREVIEW_LINES - 1)
        block = _mock(
            ToolResultBlock, tool_use_id="t1", content=short_content, is_error=False
        )
        msg = _mock(UserMessage, content=[block])
        print_agent_message(msg)
        assert "more lines" not in capsys.readouterr().out

    def test_string_content_produces_no_output(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # Plain string content is an echo of the user's own query — skip it.
        msg = _mock(UserMessage, content="Review utils.py")
        print_agent_message(msg)
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Tests — print_agent_message: ResultMessage
# ---------------------------------------------------------------------------

class TestPrintResultMessage:
    """print_agent_message handles ResultMessage summary stats."""

    def test_done_header_printed(self, capsys: pytest.CaptureFixture) -> None:
        msg = _mock(
            ResultMessage,
            subtype="success",
            is_error=False,
            num_turns=3,
            duration_ms=1200,
            duration_api_ms=900,
            total_cost_usd=0.0042,
            session_id="s1",
        )
        print_agent_message(msg)
        assert "[Done]" in capsys.readouterr().out

    def test_status_ok_when_not_error(self, capsys: pytest.CaptureFixture) -> None:
        msg = _mock(
            ResultMessage,
            subtype="success",
            is_error=False,
            num_turns=1,
            duration_ms=500,
            duration_api_ms=400,
            total_cost_usd=None,
            session_id="s1",
        )
        print_agent_message(msg)
        assert "status=OK" in capsys.readouterr().out

    def test_status_failed_when_error(self, capsys: pytest.CaptureFixture) -> None:
        msg = _mock(
            ResultMessage,
            subtype="error",
            is_error=True,
            num_turns=1,
            duration_ms=500,
            duration_api_ms=400,
            total_cost_usd=None,
            session_id="s1",
        )
        print_agent_message(msg)
        assert "status=FAILED" in capsys.readouterr().out

    def test_cost_printed_when_available(self, capsys: pytest.CaptureFixture) -> None:
        msg = _mock(
            ResultMessage,
            subtype="success",
            is_error=False,
            num_turns=2,
            duration_ms=800,
            duration_api_ms=600,
            total_cost_usd=0.0012,
            session_id="s1",
        )
        print_agent_message(msg)
        assert "$0.0012" in capsys.readouterr().out

    def test_cost_na_when_none(self, capsys: pytest.CaptureFixture) -> None:
        msg = _mock(
            ResultMessage,
            subtype="success",
            is_error=False,
            num_turns=1,
            duration_ms=300,
            duration_api_ms=200,
            total_cost_usd=None,
            session_id="s1",
        )
        print_agent_message(msg)
        assert "N/A" in capsys.readouterr().out

    def test_num_turns_printed(self, capsys: pytest.CaptureFixture) -> None:
        msg = _mock(
            ResultMessage,
            subtype="success",
            is_error=False,
            num_turns=7,
            duration_ms=300,
            duration_api_ms=200,
            total_cost_usd=None,
            session_id="s1",
        )
        print_agent_message(msg)
        assert "turns=7" in capsys.readouterr().out

    def test_duration_printed(self, capsys: pytest.CaptureFixture) -> None:
        msg = _mock(
            ResultMessage,
            subtype="success",
            is_error=False,
            num_turns=1,
            duration_ms=4321,
            duration_api_ms=3000,
            total_cost_usd=None,
            session_id="s1",
        )
        print_agent_message(msg)
        assert "4321ms" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tests — print_agent_message: SystemMessage
# ---------------------------------------------------------------------------

class TestPrintSystemMessage:
    """print_agent_message handles SystemMessage."""

    def test_system_header_with_subtype(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        msg = _mock(SystemMessage, subtype="init", data={})
        print_agent_message(msg)
        assert "[System/init]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tests — print_agent_message: unknown type
# ---------------------------------------------------------------------------

class TestPrintUnknownMessage:
    """print_agent_message produces no output for unrecognised message types."""

    def test_unknown_type_silent(self, capsys: pytest.CaptureFixture) -> None:
        print_agent_message(object())
        assert capsys.readouterr().out == ""
