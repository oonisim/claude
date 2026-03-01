"""Tests for util_claude.common.

Covers every public function in three groups:

1. Message content extractors  — text_blocks, tool_use_blocks, tool_result_blocks
2. Tool input summariser        — summarise_tool_input
3. One-line log summaries       — assistant_message_oneliner, user_message_oneliner
"""

import pytest
from unittest.mock import MagicMock

from claude_agent_sdk import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from util_claude.common import (
    _first_sentence,
    _tool_key_arg,
    assistant_message_oneliner,
    summarise_tool_input,
    text_blocks,
    tool_name_map,
    tool_result_blocks,
    tool_use_blocks,
    user_message_oneliner,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock(spec_class: type, **attrs: object) -> MagicMock:
    """Return a MagicMock that passes isinstance checks for spec_class."""
    m = MagicMock(spec=spec_class)
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


def _assistant(content: list) -> MagicMock:
    return _mock(AssistantMessage, content=content)


def _user(content) -> MagicMock:
    return _mock(UserMessage, content=content)


def _text(text: str) -> MagicMock:
    return _mock(TextBlock, text=text)


def _thinking(thought: str = "internal") -> MagicMock:
    return _mock(ThinkingBlock, thinking=thought, signature="sig")


def _tool_use(name: str, input: dict, id: str = "tu_1") -> MagicMock:
    return _mock(ToolUseBlock, id=id, name=name, input=input)


def _tool_result(content, *, is_error: bool = False) -> MagicMock:
    return _mock(ToolResultBlock, tool_use_id="tu_1", content=content, is_error=is_error)


# ===========================================================================
# 1. Message content extractors
# ===========================================================================

class TestTextBlocks:
    """text_blocks extracts TextBlock.text strings from AssistantMessage.content."""

    def test_single_text_block_returns_its_text(self) -> None:
        msg = _assistant([_text("Hello")])
        assert text_blocks(msg) == ["Hello"]

    def test_multiple_text_blocks_returned_in_order(self) -> None:
        msg = _assistant([_text("first"), _text("second")])
        assert text_blocks(msg) == ["first", "second"]

    def test_no_text_blocks_returns_empty_list(self) -> None:
        # Content contains only a ToolUseBlock — no TextBlock present.
        msg = _assistant([_tool_use("bash", {"command": "ls"})])
        assert text_blocks(msg) == []

    def test_mixed_content_returns_only_text(self) -> None:
        # AssistantMessage.content may be [TextBlock, ThinkingBlock, ToolUseBlock].
        # Only TextBlock.text values should be returned.
        msg = _assistant([
            _text("I'll read the file."),
            _thinking("internal reasoning"),
            _tool_use("read", {"file_path": "utils.py"}),
        ])
        assert text_blocks(msg) == ["I'll read the file."]

    def test_empty_content_returns_empty_list(self) -> None:
        msg = _assistant([])
        assert text_blocks(msg) == []

    def test_thinking_block_not_included(self) -> None:
        # ThinkingBlock.thinking is internal — must not appear in results.
        msg = _assistant([_thinking("secret")])
        assert text_blocks(msg) == []


class TestToolUseBlocks:
    """tool_use_blocks extracts ToolUseBlock objects from AssistantMessage.content."""

    def test_single_tool_use_block_returned(self) -> None:
        block = _tool_use("bash", {"command": "pytest"})
        msg = _assistant([block])
        result = tool_use_blocks(msg)
        assert len(result) == 1
        assert result[0].name == "bash"

    def test_multiple_tool_use_blocks_returned_in_order(self) -> None:
        b1 = _tool_use("read", {"file_path": "a.py"})
        b2 = _tool_use("bash", {"command": "ls"})
        msg = _assistant([b1, b2])
        result = tool_use_blocks(msg)
        assert [b.name for b in result] == ["read", "bash"]

    def test_no_tool_use_blocks_returns_empty_list(self) -> None:
        msg = _assistant([_text("Just prose.")])
        assert tool_use_blocks(msg) == []

    def test_mixed_content_returns_only_tool_use(self) -> None:
        msg = _assistant([
            _text("Running now."),
            _tool_use("bash", {"command": "ls"}),
            _thinking("chain of thought"),
        ])
        result = tool_use_blocks(msg)
        assert len(result) == 1
        assert result[0].name == "bash"

    def test_tool_use_block_input_preserved(self) -> None:
        # ToolUseBlock.input (the argument dict) must be the same object.
        args = {"command": "pytest -v", "description": "run tests"}
        block = _tool_use("bash", args)
        msg = _assistant([block])
        assert tool_use_blocks(msg)[0].input == args

    def test_empty_content_returns_empty_list(self) -> None:
        assert tool_use_blocks(_assistant([])) == []


class TestToolResultBlocks:
    """tool_result_blocks extracts ToolResultBlock objects from UserMessage.content."""

    def test_list_content_with_tool_result_returned(self) -> None:
        # UserMessage.content is a list when the SDK feeds tool results back.
        block = _tool_result("10 passed")
        msg = _user([block])
        result = tool_result_blocks(msg)
        assert len(result) == 1
        assert result[0].content == "10 passed"

    def test_multiple_tool_results_returned_in_order(self) -> None:
        b1 = _tool_result("output A")
        b2 = _tool_result("output B", is_error=True)
        msg = _user([b1, b2])
        result = tool_result_blocks(msg)
        assert len(result) == 2
        assert result[0].content == "output A"
        assert result[1].is_error is True

    def test_plain_string_content_returns_empty_list(self) -> None:
        # UserMessage.content is a str when it echoes the user's own query.
        # No tool results are present; return [].
        msg = _user("Review utils.py")
        assert tool_result_blocks(msg) == []

    def test_empty_list_content_returns_empty_list(self) -> None:
        msg = _user([])
        assert tool_result_blocks(msg) == []

    def test_is_error_flag_preserved(self) -> None:
        # ToolResultBlock.is_error distinguishes a failed tool from a
        # successful one.  The value must be passed through unchanged.
        block = _tool_result("Permission denied", is_error=True)
        msg = _user([block])
        assert tool_result_blocks(msg)[0].is_error is True


# ===========================================================================
# 2. Tool input summariser
# ===========================================================================

class TestSummariseToolInputBash:
    """bash: shows tool_input["command"] and optional tool_input["description"]."""

    def test_command_only_shown_with_dollar_prefix(self) -> None:
        # tool_input["command"] is the shell command string.
        result = summarise_tool_input("bash", {"command": "pytest -v"})
        assert "pytest -v" in result
        assert result.startswith("$")

    def test_description_shown_before_command(self) -> None:
        # tool_input["description"] is the human-readable intent.
        result = summarise_tool_input("bash", {
            "command": "pytest -v",
            "description": "run test suite",
        })
        lines = result.splitlines()
        assert lines[0] == "run test suite"
        assert "pytest -v" in lines[1]

    def test_no_description_produces_single_line(self) -> None:
        result = summarise_tool_input("bash", {"command": "ls"})
        assert "\n" not in result

    def test_case_insensitive_tool_name(self) -> None:
        result = summarise_tool_input("Bash", {"command": "ls"})
        assert "ls" in result

    def test_missing_command_key_does_not_raise(self) -> None:
        result = summarise_tool_input("bash", {})
        assert result == "$ "


class TestSummariseToolInputFileTools:
    """edit / write / multiedit / read: shows tool_input["file_path"]."""

    @pytest.mark.parametrize("tool", ["edit", "write", "multiedit"])
    def test_file_path_shown(self, tool: str) -> None:
        # tool_input["file_path"] is the path of the file being modified.
        result = summarise_tool_input(tool, {"file_path": "src/utils.py"})
        assert "src/utils.py" in result

    @pytest.mark.parametrize("tool", ["edit", "write", "multiedit"])
    def test_file_path_missing_shows_unknown(self, tool: str) -> None:
        result = summarise_tool_input(tool, {})
        assert "(unknown)" in result

    def test_read_shows_file_path(self) -> None:
        # tool_input["file_path"] is the path of the file to read.
        result = summarise_tool_input("read", {"file_path": "utils.py"})
        assert "utils.py" in result

    def test_read_missing_file_path_shows_unknown(self) -> None:
        result = summarise_tool_input("read", {})
        assert "(unknown)" in result

    @pytest.mark.parametrize("tool", ["edit", "write", "multiedit", "read"])
    def test_other_fields_not_shown(self, tool: str) -> None:
        # The summary must not leak unrelated fields like old_string / new_string.
        result = summarise_tool_input(tool, {
            "file_path": "f.py",
            "old_string": "very long old content",
            "new_string": "very long new content",
        })
        assert "old_string" not in result
        assert "new_string" not in result


class TestSummariseToolInputExitPlanMode:
    """exitplanmode: shows plan title + allowedPrompts bullet list."""

    def test_plan_title_extracted_from_heading(self) -> None:
        # tool_input["plan"] first line: "# Plan: <title>"
        # The "#" and "Plan: " prefix must be stripped.
        result = summarise_tool_input("ExitPlanMode", {
            "plan": "# Plan: Harden utils.py\n\ndetails...",
            "allowedPrompts": [],
        })
        assert "Harden utils.py" in result
        assert "#" not in result
        assert "Plan:" not in result

    def test_allowed_prompts_shown_as_bullets(self) -> None:
        # tool_input["allowedPrompts"]: list[{"tool": str, "prompt": str}]
        # prompt["tool"]   — e.g. "Bash"
        # prompt["prompt"] — natural-language description of the action
        result = summarise_tool_input("ExitPlanMode", {
            "plan": "# Plan: Fix tests",
            "allowedPrompts": [
                {"tool": "Bash", "prompt": "run pytest"},
                {"tool": "Bash", "prompt": "update utils.py"},
            ],
        })
        assert "run pytest" in result
        assert "update utils.py" in result
        assert "[Bash]" in result

    def test_no_allowed_prompts_no_actions_section(self) -> None:
        result = summarise_tool_input("ExitPlanMode", {
            "plan": "# Plan: Fix tests",
            "allowedPrompts": [],
        })
        assert "actions" not in result.lower()

    def test_empty_plan_shows_no_title_placeholder(self) -> None:
        result = summarise_tool_input("ExitPlanMode", {
            "plan": "",
            "allowedPrompts": [],
        })
        assert "(no title)" in result

    def test_plan_body_not_included_in_summary(self) -> None:
        # The full plan markdown may be thousands of characters; only the
        # first-line title must appear in the summary.
        result = summarise_tool_input("ExitPlanMode", {
            "plan": "# Plan: Short title\n## Section\nLots of body detail here.",
            "allowedPrompts": [],
        })
        assert "Lots of body detail" not in result

    def test_case_insensitive_tool_name(self) -> None:
        result = summarise_tool_input("exitplanmode", {
            "plan": "# Plan: Test",
            "allowedPrompts": [],
        })
        assert "Test" in result


class TestSummariseToolInputGeneric:
    """Unknown tools: show all key/value pairs with long values truncated."""

    def test_all_fields_shown(self) -> None:
        result = summarise_tool_input("grep", {"pattern": "foo", "path": "src/"})
        assert "pattern" in result
        assert "foo" in result
        assert "path" in result
        assert "src/" in result

    def test_long_value_truncated_with_ellipsis(self) -> None:
        # Values longer than _MAX_VALUE_CHARS (120) must be truncated.
        long_val = "x" * 200
        result = summarise_tool_input("unknown_tool", {"key": long_val})
        assert "…" in result
        assert len(result) < 200

    def test_empty_input_returns_no_arguments(self) -> None:
        result = summarise_tool_input("unknown_tool", {})
        assert result == "(no arguments)"

    def test_short_value_not_truncated(self) -> None:
        result = summarise_tool_input("unknown_tool", {"key": "short"})
        assert "…" not in result


# ===========================================================================
# 3. One-line log summaries
# ===========================================================================

class TestFirstSentence:
    """_first_sentence truncates at a natural sentence boundary or word boundary."""

    def test_short_text_returned_unchanged(self) -> None:
        snippet, more = _first_sentence("Hello.", 80)
        assert snippet == "Hello."
        assert more is False

    def test_text_at_exact_limit_returned_unchanged(self) -> None:
        text = "a" * 60
        snippet, more = _first_sentence(text, 60)
        assert snippet == text
        assert more is False

    def test_long_text_sets_more_true(self) -> None:
        _, more = _first_sentence("x" * 100, 60)
        assert more is True

    def test_sentence_boundary_preferred_over_char_slice(self) -> None:
        # Two short sentences: first fits within max_chars, should be returned whole.
        text = "First sentence. Second sentence continues here."
        snippet, more = _first_sentence(text, 40)
        assert snippet == "First sentence."
        assert more is True

    def test_fallback_ends_at_word_boundary_not_mid_word(self) -> None:
        # No sentence boundary within max_chars → word boundary fallback.
        # "aaaa bbbb cccc dddd" with max_chars=10 should not cut mid-word.
        text = "alpha beta gamma delta epsilon zeta eta theta"
        snippet, more = _first_sentence(text, 15)
        # Snippet must not end mid-word.
        assert not snippet[-1].isalpha() or snippet == snippet.rstrip()
        assert " " not in snippet[len(snippet.rstrip()):]

    def test_newlines_collapsed_to_spaces(self) -> None:
        snippet, _ = _first_sentence("line one\nline two", 40)
        assert "\n" not in snippet

    def test_empty_text_returns_empty(self) -> None:
        snippet, more = _first_sentence("", 60)
        assert snippet == ""
        assert more is False


class TestToolKeyArg:
    """_tool_key_arg returns the single most meaningful argument for a tool label."""

    def test_bash_returns_command(self) -> None:
        # ToolUseBlock.input["command"] is the shell command string.
        assert _tool_key_arg("bash", {"command": "pytest -v"}) == "pytest -v"

    def test_bash_case_insensitive(self) -> None:
        assert _tool_key_arg("Bash", {"command": "ls"}) == "ls"

    def test_read_returns_file_path(self) -> None:
        assert _tool_key_arg("read", {"file_path": "utils.py"}) == "utils.py"

    def test_edit_returns_file_path(self) -> None:
        assert _tool_key_arg("edit", {"file_path": "utils.py"}) == "utils.py"

    def test_write_returns_file_path(self) -> None:
        assert _tool_key_arg("write", {"file_path": "out.py"}) == "out.py"

    def test_multiedit_returns_file_path(self) -> None:
        assert _tool_key_arg("multiedit", {"file_path": "f.py"}) == "f.py"

    def test_exitplanmode_returns_empty(self) -> None:
        # Plan approval: no single key arg makes sense.
        assert _tool_key_arg("exitplanmode", {"plan": "...", "allowedPrompts": []}) == ""

    def test_long_bash_command_truncated_with_ellipsis(self) -> None:
        # Long bash commands are truncated at _MAX_KEY_ARG_CHARS with "…".
        long_cmd = "x" * 100
        result = _tool_key_arg("bash", {"command": long_cmd})
        assert result.endswith("…")
        assert len(result) <= 51  # _MAX_KEY_ARG_CHARS + 1 for ellipsis char

    def test_long_file_path_shows_last_two_components(self) -> None:
        # Long file paths (> _MAX_KEY_ARG_CHARS=50) show "…/<dir>/<basename>"
        # rather than a mid-path character slice.
        long_path = "/Users/oonisim/home/repository/git/oonisim/claude/src/agent/utils.py"
        assert len(long_path) > 50, "precondition: path must exceed the limit"
        result = _tool_key_arg("read", {"file_path": long_path})
        assert result.startswith("…/")
        assert result.endswith("utils.py")
        assert "agent" in result  # penultimate directory component

    def test_short_file_path_returned_as_is(self) -> None:
        # Short file paths are returned unchanged (no truncation).
        assert _tool_key_arg("edit", {"file_path": "src/utils.py"}) == "src/utils.py"

    def test_generic_returns_first_value(self) -> None:
        result = _tool_key_arg("grep", {"pattern": "foo", "path": "src/"})
        assert result == "foo"

    def test_empty_input_returns_empty(self) -> None:
        assert _tool_key_arg("unknown", {}) == ""


class TestAssistantMessageOneliner:
    """assistant_message_oneliner produces a single-line log summary."""

    def test_single_tool_shows_run_tool(self) -> None:
        # Single ToolUseBlock → "Run Tool: <name>(<key_arg>)"
        msg = _assistant([_tool_use("bash", {"command": "ls"})])
        assert "Run Tool: bash(ls)" in assistant_message_oneliner(msg)

    def test_multiple_tools_shows_run_tools(self) -> None:
        # Multiple ToolUseBlocks → "Run Tools: ..."
        msg = _assistant([
            _tool_use("read", {"file_path": "f.py"}),
            _tool_use("bash", {"command": "ls"}),
        ])
        line = assistant_message_oneliner(msg)
        assert "Run Tools:" in line
        assert "read(f.py)" in line
        assert "bash(ls)" in line

    def test_tool_without_key_arg_shows_name_only(self) -> None:
        # exitplanmode returns "" from _tool_key_arg → shown without parens.
        msg = _assistant([_tool_use("ExitPlanMode", {"plan": "...", "allowedPrompts": []})])
        line = assistant_message_oneliner(msg)
        assert "ExitPlanMode" in line
        assert "ExitPlanMode()" not in line  # empty arg → no parens

    def test_text_block_shows_says_snippet(self) -> None:
        # TextBlock.text — first 80 chars shown inside 'says: "..."'.
        msg = _assistant([_text("I will now analyse the repository.")])
        line = assistant_message_oneliner(msg)
        assert 'says: "I will now analyse the repository."' in line

    def test_long_text_truncated_with_ellipsis(self) -> None:
        msg = _assistant([_text("a" * 100)])
        assert "…" in assistant_message_oneliner(msg)

    def test_short_text_no_ellipsis(self) -> None:
        msg = _assistant([_text("short")])
        assert "…" not in assistant_message_oneliner(msg)

    def test_both_tool_and_text_joined_with_pipe(self) -> None:
        msg = _assistant([
            _text("Running tests."),
            _tool_use("bash", {"command": "pytest"}),
        ])
        line = assistant_message_oneliner(msg)
        assert "|" in line
        assert "Run Tool: bash(pytest)" in line
        assert "says:" in line

    def test_thinking_block_only_returns_no_content(self) -> None:
        msg = _assistant([_thinking("internal")])
        assert assistant_message_oneliner(msg) == "(no content)"

    def test_empty_content_returns_no_content(self) -> None:
        msg = _assistant([])
        assert assistant_message_oneliner(msg) == "(no content)"

    def test_newlines_in_text_collapsed_to_spaces(self) -> None:
        msg = _assistant([_text("line one\nline two")])
        assert "\n" not in assistant_message_oneliner(msg)


class TestToolNameMap:
    """tool_name_map builds a ToolUseBlock.id → ToolUseBlock.name dict."""

    def test_single_tool_use_block_mapped(self) -> None:
        # ToolUseBlock.id is matched by ToolResultBlock.tool_use_id.
        block = _tool_use("Bash", {"command": "ls"}, id="tu_abc")
        msg = _assistant([block])
        assert tool_name_map(msg) == {"tu_abc": "Bash"}

    def test_multiple_tool_use_blocks_all_mapped(self) -> None:
        b1 = _tool_use("Read", {"file_path": "f.py"}, id="tu_1")
        b2 = _tool_use("Bash", {"command": "ls"}, id="tu_2")
        msg = _assistant([b1, b2])
        assert tool_name_map(msg) == {"tu_1": "Read", "tu_2": "Bash"}

    def test_no_tool_use_blocks_returns_empty_dict(self) -> None:
        msg = _assistant([_text("Just prose.")])
        assert tool_name_map(msg) == {}

    def test_non_tool_blocks_excluded(self) -> None:
        msg = _assistant([_text("hi"), _thinking(), _tool_use("Bash", {}, id="tu_x")])
        assert tool_name_map(msg) == {"tu_x": "Bash"}


class TestUserMessageOneliner:
    """user_message_oneliner produces a single-line log summary."""

    def test_ok_result_shows_ok_status(self) -> None:
        # ToolResultBlock.is_error == False → "OK"
        msg = _user([_tool_result("10 passed in 0.4s", is_error=False)])
        assert "OK" in user_message_oneliner(msg)

    def test_error_result_shows_error_status(self) -> None:
        # ToolResultBlock.is_error == True → "ERROR"
        msg = _user([_tool_result("Permission denied", is_error=True)])
        assert "ERROR" in user_message_oneliner(msg)

    def test_content_snippet_included(self) -> None:
        # ToolResultBlock.content — first 60 chars shown.
        msg = _user([_tool_result("10 passed in 0.4s")])
        assert "10 passed in 0.4s" in user_message_oneliner(msg)

    def test_long_content_truncated_with_ellipsis(self) -> None:
        # ToolResultBlock.content longer than 60 chars is truncated.
        msg = _user([_tool_result("x" * 100)])
        line = user_message_oneliner(msg)
        assert "…" in line

    def test_short_content_no_ellipsis(self) -> None:
        msg = _user([_tool_result("ok")])
        assert "…" not in user_message_oneliner(msg)

    def test_plain_string_content_returns_query_echo(self) -> None:
        # UserMessage.content is a str when it echoes the user's query.
        msg = _user("Review utils.py")
        assert user_message_oneliner(msg) == "(query echo)"

    def test_empty_list_returns_query_echo(self) -> None:
        msg = _user([])
        assert user_message_oneliner(msg) == "(query echo)"

    def test_starts_with_status_when_no_tool_name(self) -> None:
        # No tool_names mapping supplied → line starts with the bare status.
        # Format: "OK: <snippet>"  (no "tool result —" prefix)
        msg = _user([_tool_result("output")])
        line = user_message_oneliner(msg)
        assert line.startswith("OK")

    def test_multiple_results_all_included(self) -> None:
        msg = _user([
            _tool_result("first output"),
            _tool_result("second output", is_error=True),
        ])
        line = user_message_oneliner(msg)
        assert "first output" in line
        assert "second output" in line
        assert "ERROR" in line

    def test_newlines_in_content_collapsed(self) -> None:
        msg = _user([_tool_result("line1\nline2")])
        assert "\n" not in user_message_oneliner(msg)

    def test_none_content_does_not_raise(self) -> None:
        # ToolResultBlock.content may be None for empty tool output.
        msg = _user([_tool_result(None)])
        line = user_message_oneliner(msg)
        assert "OK" in line   # is_error defaults to False

    def test_tool_name_shown_when_mapping_provided(self) -> None:
        # When tool_names maps tool_use_id → name, the name prefixes the status.
        block = _tool_result("10 passed", is_error=False)
        block.tool_use_id = "tu_abc"
        msg = _user([block])
        line = user_message_oneliner(msg, {"tu_abc": "Bash"})
        assert "Bash" in line
        assert "OK" in line
        assert "10 passed" in line

    def test_tool_name_and_status_shown_together(self) -> None:
        # Format: "<tool> <status>: <snippet>"  (no arrow between name and status)
        block = _tool_result("output", is_error=False)
        block.tool_use_id = "tu_1"
        msg = _user([block])
        line = user_message_oneliner(msg, {"tu_1": "Read"})
        assert "Read OK" in line

    def test_error_with_tool_name(self) -> None:
        block = _tool_result("Permission denied", is_error=True)
        block.tool_use_id = "tu_1"
        msg = _user([block])
        line = user_message_oneliner(msg, {"tu_1": "Bash"})
        assert "Bash ERROR" in line

    def test_unknown_tool_use_id_falls_back_to_status_only(self) -> None:
        # If tool_use_id is not in the mapping, no name prefix is added.
        block = _tool_result("output")
        block.tool_use_id = "tu_missing"
        msg = _user([block])
        line = user_message_oneliner(msg, {"tu_other": "Bash"})
        assert "→" not in line
        assert "OK" in line

    def test_none_tool_names_behaves_same_as_no_arg(self) -> None:
        msg = _user([_tool_result("output")])
        assert user_message_oneliner(msg, None) == user_message_oneliner(msg)
