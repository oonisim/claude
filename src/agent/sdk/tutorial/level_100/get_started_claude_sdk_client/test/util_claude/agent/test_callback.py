"""Tests for util_claude.agent.human_in_the_loop."""

import pytest
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from util_claude.agent.human_in_the_loop import (
    AgentExecutionPlanTracker,
    _show_review,
    make_approval_callback,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def context() -> MagicMock:
    """Return a minimal mock ToolPermissionContext."""
    return MagicMock(spec=ToolPermissionContext)


@pytest.fixture()
def clarification_options() -> MagicMock:
    """Return a minimal mock ClaudeAgentOptions for clarification sessions."""
    return MagicMock(spec=ClaudeAgentOptions)


@pytest.fixture()
def callback(clarification_options: MagicMock):
    """Return the CanUseTool callback produced by make_approval_callback."""
    return make_approval_callback(clarification_options)


def _make_loop(*choices: str) -> MagicMock:
    """Return a mock event loop whose run_in_executor yields choices in order.

    Each call to ``run_in_executor`` consumes one value from ``choices`` in
    order.  Use this to simulate menu selections and text input:

    * Option 1 (Approve)  — one value: the menu choice ``"1"``
    * Option 2 (Ask)      — two values: ``"2"`` then the user's question text
    * Option 3 (Review)   — one value: ``"3"`` (then the next menu choice)
    * Option 4 (Reject)   — two values: ``"4"`` then the reason text
    * Option 5 (Abort)    — one value: ``"5"``
    """
    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(side_effect=list(choices))
    return mock_loop


def _make_sdk_client() -> MagicMock:
    """Return a mock ClaudeSDKClient that yields no messages."""

    async def _empty_receive():
        return
        yield  # pragma: no cover — makes it an async generator

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.query = AsyncMock()
    mock_client.receive_response = _empty_receive
    return mock_client


_EXITPLANMODE_INPUT = {
    "plan": "# Plan: Harden utils.py\n\ndetails",
    "allowedPrompts": [
        {"tool": "Bash", "prompt": "run pytest"},
        {"tool": "Bash", "prompt": "update utils.py"},
    ],
}


# ===========================================================================
# AgentExecutionPlanTracker unit tests
# ===========================================================================

class TestPlanTrackerLoad:
    """AgentExecutionPlanTracker.load extracts title and steps from ExitPlanMode input."""

    def test_title_extracted_from_plan_heading(self) -> None:
        # tool_input["plan"] first line: "# Plan: <title>"
        tracker = AgentExecutionPlanTracker()
        tracker.load({"plan": "# Plan: My Task\ndetails", "allowedPrompts": []})
        assert tracker.title == "My Task"

    def test_steps_extracted_from_allowed_prompts(self) -> None:
        # tool_input["allowedPrompts"]: list[{"tool": str, "prompt": str}]
        tracker = AgentExecutionPlanTracker()
        tracker.load({
            "plan": "# Plan: T",
            "allowedPrompts": [
                {"tool": "Bash", "prompt": "run tests"},
                {"tool": "Bash", "prompt": "update file"},
            ],
        })
        assert tracker.total == 2
        assert tracker.has_plan

    def test_first_load_returns_false(self) -> None:
        # changed=False when there was no previous plan.
        tracker = AgentExecutionPlanTracker()
        changed = tracker.load({"plan": "# Plan: T", "allowedPrompts": []})
        assert changed is False

    def test_second_load_with_same_plan_returns_false(self) -> None:
        tracker = AgentExecutionPlanTracker()
        inp = {"plan": "# Plan: T", "allowedPrompts": [{"tool": "Bash", "prompt": "x"}]}
        tracker.load(inp)
        assert tracker.load(inp) is False

    def test_second_load_with_different_plan_returns_true(self) -> None:
        tracker = AgentExecutionPlanTracker()
        tracker.load({"plan": "# Plan: Old", "allowedPrompts": []})
        changed = tracker.load({"plan": "# Plan: New", "allowedPrompts": []})
        assert changed is True

    def test_load_resets_step_counter(self) -> None:
        tracker = AgentExecutionPlanTracker()
        tracker.load({
            "plan": "# Plan: T",
            "allowedPrompts": [{"tool": "Bash", "prompt": "step 1"}],
        })
        tracker.try_advance("Bash")
        # Second load should reset _current to 0.
        tracker.load({"plan": "# Plan: T2", "allowedPrompts": []})
        assert not tracker.has_plan

    def test_empty_plan_text_uses_fallback_title(self) -> None:
        tracker = AgentExecutionPlanTracker()
        tracker.load({"plan": "", "allowedPrompts": []})
        assert tracker.title == "Execution Plan"


class TestPlanTrackerAdvance:
    """AgentExecutionPlanTracker.try_advance advances the step counter on tool-type match."""

    def _loaded(self, steps: list) -> AgentExecutionPlanTracker:
        tracker = AgentExecutionPlanTracker()
        tracker.load({
            "plan": "# Plan: T",
            "allowedPrompts": [{"tool": t, "prompt": p} for t, p in steps],
        })
        return tracker

    def test_matching_tool_advances_counter(self) -> None:
        tracker = self._loaded([("Bash", "run tests")])
        tracker.try_advance("Bash")
        # After advancing, no more steps match Bash at position 0.
        assert tracker._current == 1

    def test_non_matching_tool_does_not_advance(self) -> None:
        tracker = self._loaded([("Bash", "run tests")])
        tracker.try_advance("Read")   # not Bash
        assert tracker._current == 0

    def test_advance_is_case_insensitive(self) -> None:
        tracker = self._loaded([("Bash", "step")])
        tracker.try_advance("bash")   # lowercase
        assert tracker._current == 1

    def test_advance_past_end_does_not_raise(self) -> None:
        tracker = self._loaded([("Bash", "only step")])
        tracker.try_advance("Bash")   # moves to 1
        tracker.try_advance("Bash")   # already at end — should not raise
        assert tracker._current == 1  # still at end, not beyond

    def test_sequential_steps_advance_correctly(self) -> None:
        tracker = self._loaded([("Bash", "step 1"), ("Bash", "step 2")])
        tracker.try_advance("Bash")
        assert tracker._current == 1
        tracker.try_advance("Bash")
        assert tracker._current == 2


class TestPlanTrackerDisplay:
    """AgentExecutionPlanTracker print methods produce expected output."""

    def _loaded_tracker(self) -> AgentExecutionPlanTracker:
        tracker = AgentExecutionPlanTracker()
        tracker.load({
            "plan": "# Plan: Harden utils.py",
            "allowedPrompts": [
                {"tool": "Bash", "prompt": "run pytest"},
                {"tool": "Bash", "prompt": "update utils.py"},
            ],
        })
        return tracker

    def test_print_full_plan_shows_title(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = self._loaded_tracker()
        tracker.print_full_plan()
        assert "Harden utils.py" in capsys.readouterr().out

    def test_print_full_plan_shows_all_steps(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = self._loaded_tracker()
        tracker.print_full_plan()
        out = capsys.readouterr().out
        assert "run pytest" in out
        assert "update utils.py" in out

    def test_print_full_plan_shows_updated_banner(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = self._loaded_tracker()
        tracker.print_full_plan(changed=True)
        assert "PLAN UPDATED" in capsys.readouterr().out

    def test_print_full_plan_no_banner_when_not_changed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = self._loaded_tracker()
        tracker.print_full_plan(changed=False)
        assert "PLAN UPDATED" not in capsys.readouterr().out

    def test_context_bar_shows_plan_title(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = self._loaded_tracker()
        tracker.print_context_bar("Bash")
        assert "Harden utils.py" in capsys.readouterr().out

    def test_context_bar_shows_current_step(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = self._loaded_tracker()
        tracker.print_context_bar("Bash")
        out = capsys.readouterr().out
        assert "Step 1" in out
        assert "run pytest" in out

    def test_context_bar_shows_exploring_for_unmatched_tool(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = self._loaded_tracker()
        tracker.print_context_bar("Read")   # not in allowedPrompts
        assert "Exploring" in capsys.readouterr().out

    def test_context_bar_shows_completed_when_all_done(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = self._loaded_tracker()
        tracker.try_advance("Bash")
        tracker.try_advance("Bash")
        tracker.print_context_bar("Bash")
        assert "completed" in capsys.readouterr().out.lower()

    def test_context_bar_silent_when_no_plan(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tracker = AgentExecutionPlanTracker()   # never loaded
        tracker.print_context_bar("Bash")
        assert capsys.readouterr().out == ""


class TestPlanTrackerStepLabel:
    """step_label returns a formatted 'Step N/T: description' string or None."""

    def _loaded(self, steps: list) -> AgentExecutionPlanTracker:
        tracker = AgentExecutionPlanTracker()
        tracker.load({
            "plan": "# Plan: T",
            "allowedPrompts": [{"tool": t, "prompt": p} for t, p in steps],
        })
        return tracker

    def test_matching_tool_returns_label(self) -> None:
        tracker = self._loaded([("Bash", "run pytest"), ("Bash", "update file")])
        label = tracker.step_label("Bash")
        assert label == "Step 1/2: run pytest"

    def test_label_advances_after_step_approved(self) -> None:
        tracker = self._loaded([("Bash", "run pytest"), ("Bash", "update file")])
        tracker.try_advance("Bash")
        label = tracker.step_label("Bash")
        assert label == "Step 2/2: update file"

    def test_non_matching_tool_returns_none(self) -> None:
        tracker = self._loaded([("Bash", "run tests")])
        assert tracker.step_label("Read") is None

    def test_no_plan_returns_none(self) -> None:
        tracker = AgentExecutionPlanTracker()
        assert tracker.step_label("Bash") is None

    def test_all_steps_done_returns_none(self) -> None:
        tracker = self._loaded([("Bash", "step")])
        tracker.try_advance("Bash")
        assert tracker.step_label("Bash") is None

    def test_label_case_insensitive(self) -> None:
        tracker = self._loaded([("Bash", "run tests")])
        assert tracker.step_label("bash") is not None


# ===========================================================================
# Callback integration tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Tests — option 1 (Approve)
# ---------------------------------------------------------------------------

class TestApprove:
    """Choice '1' returns PermissionResultAllow."""

    async def test_approve(self, callback, context: MagicMock) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("1"),
        ):
            result = await callback("bash", {}, context)
        assert isinstance(result, PermissionResultAllow)

    async def test_approve_ignores_whitespace(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("  1  "),
        ):
            result = await callback("bash", {}, context)
        assert isinstance(result, PermissionResultAllow)

    async def test_approve_exitplanmode_loads_plan(
        self, callback, context: MagicMock
    ) -> None:
        # ExitPlanMode approval should not raise and should return Allow.
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("1"),
        ):
            result = await callback("ExitPlanMode", _EXITPLANMODE_INPUT, context)
        assert isinstance(result, PermissionResultAllow)


# ---------------------------------------------------------------------------
# Tests — option 2 (Ask)
# ---------------------------------------------------------------------------

class TestAsk:
    """Choice '2' prompts user for a question, opens a clarification session, re-prompts.

    run_in_executor is called twice for option 2:
      1st call → menu choice "2"
      2nd call → user's question text (empty string → auto-explain prompt)
    """

    async def test_ask_with_empty_question_then_approve(
        self, callback, context: MagicMock
    ) -> None:
        # Empty question → auto-explain prompt; then approve.
        mock_client = _make_sdk_client()
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("2", "", "1"),
        ):
            with patch(
                "util_claude.agent.human_in_the_loop.ClaudeSDKClient",
                return_value=mock_client,
            ):
                result = await callback("bash", {"command": "ls"}, context)
        assert isinstance(result, PermissionResultAllow)

    async def test_ask_with_user_question_then_abort(
        self, callback, context: MagicMock
    ) -> None:
        # User types a question → clarification sent; then abort.
        mock_client = _make_sdk_client()
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("2", "why this tool?", "5"),
        ):
            with patch(
                "util_claude.agent.human_in_the_loop.ClaudeSDKClient",
                return_value=mock_client,
            ):
                result = await callback("bash", {}, context)
        assert isinstance(result, PermissionResultDeny)
        assert result.interrupt is True

    async def test_auto_explain_prompt_includes_tool_context(
        self, callback, context: MagicMock
    ) -> None:
        # Empty question → auto-explain prompt must mention tool name and args.
        mock_client = _make_sdk_client()
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("2", "", "5"),
        ):
            with patch(
                "util_claude.agent.human_in_the_loop.ClaudeSDKClient",
                return_value=mock_client,
            ):
                await callback("bash", {"command": "rm -rf /"}, context)

        sent_prompt: str = mock_client.query.call_args[0][0]
        assert "bash" in sent_prompt
        assert "rm -rf /" in sent_prompt

    async def test_user_question_sent_to_clarification(
        self, callback, context: MagicMock
    ) -> None:
        # Non-empty question → _CLARIFICATION_PROMPT with user question is sent.
        mock_client = _make_sdk_client()
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("2", "is there a safer way?", "5"),
        ):
            with patch(
                "util_claude.agent.human_in_the_loop.ClaudeSDKClient",
                return_value=mock_client,
            ):
                await callback("bash", {"command": "ls"}, context)

        sent_prompt: str = mock_client.query.call_args[0][0]
        assert "is there a safer way?" in sent_prompt


# ---------------------------------------------------------------------------
# Tests — option 3 (Review)
# ---------------------------------------------------------------------------

class TestReview:
    """Choice '3' calls _show_review then re-prompts."""

    async def test_review_then_approve(
        self, callback, context: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("3", "1"),
        ):
            with patch("util_claude.agent.human_in_the_loop._show_review") as mock_review:
                result = await callback("bash", {"command": "ls"}, context)
        mock_review.assert_called_once_with("bash", {"command": "ls"})
        assert isinstance(result, PermissionResultAllow)

    async def test_review_then_abort(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("3", "5"),
        ):
            with patch("util_claude.agent.human_in_the_loop._show_review"):
                result = await callback("read", {"file_path": "utils.py"}, context)
        assert isinstance(result, PermissionResultDeny)
        assert result.interrupt is True


class TestShowReview:
    """_show_review prints the right details for each tool type."""

    def test_bash_shows_command(self, capsys: pytest.CaptureFixture) -> None:
        _show_review("bash", {"command": "pytest -v", "description": "run tests"})
        out = capsys.readouterr().out
        assert "pytest -v" in out
        assert "run tests" in out

    def test_bash_no_description(self, capsys: pytest.CaptureFixture) -> None:
        _show_review("bash", {"command": "ls"})
        out = capsys.readouterr().out
        assert "ls" in out
        assert "Description" not in out

    def test_read_shows_file_path(self, capsys: pytest.CaptureFixture) -> None:
        _show_review("read", {"file_path": "/tmp/nonexistent_xyz.py"})
        out = capsys.readouterr().out
        assert "/tmp/nonexistent_xyz.py" in out

    def test_generic_shows_all_fields(self, capsys: pytest.CaptureFixture) -> None:
        _show_review("grep", {"pattern": "foo", "path": "src/"})
        out = capsys.readouterr().out
        assert "pattern" in out
        assert "foo" in out
        assert "path" in out

    def test_edit_shows_file_path(self, capsys: pytest.CaptureFixture) -> None:
        with patch("util_claude.agent.human_in_the_loop.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            _show_review("edit", {"file_path": "utils.py"})
        out = capsys.readouterr().out
        assert "utils.py" in out

    def test_edit_shows_diff_when_available(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        diff_output = "\n".join(f"+line {i}" for i in range(5))
        with patch("util_claude.agent.human_in_the_loop.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=diff_output, returncode=0)
            _show_review("edit", {"file_path": "utils.py"})
        out = capsys.readouterr().out
        assert "+line 0" in out

    def test_edit_handles_git_failure_gracefully(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.subprocess.run",
            side_effect=FileNotFoundError("git not found"),
        ):
            _show_review("edit", {"file_path": "utils.py"})
        assert "utils.py" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tests — option 4 (Reject)
# ---------------------------------------------------------------------------

class TestReject:
    """Choice '4' prompts for a reason and returns PermissionResultDeny(interrupt=False)."""

    async def test_reject_returns_deny(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("4", ""),
        ):
            result = await callback("bash", {}, context)
        assert isinstance(result, PermissionResultDeny)

    async def test_reject_interrupt_is_false(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("4", ""),
        ):
            result = await callback("bash", {}, context)
        assert result.interrupt is False

    async def test_reject_default_message_when_no_reason(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("4", ""),
        ):
            result = await callback("bash", {}, context)
        assert result.message == "Rejected by user."

    async def test_reject_includes_reason_in_message(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("4", "Use H1 headings instead of H2"),
        ):
            result = await callback("edit", {"file_path": "doc.md"}, context)
        assert "Use H1 headings instead of H2" in result.message

    async def test_reject_whitespace_only_reason_treated_as_empty(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("4", "   "),
        ):
            result = await callback("bash", {}, context)
        assert result.message == "Rejected by user."


# ---------------------------------------------------------------------------
# Tests — option 5 (Abort)
# ---------------------------------------------------------------------------

class TestAbort:
    """Choice '5' returns PermissionResultDeny with interrupt=True."""

    async def test_abort_returns_deny(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("5"),
        ):
            result = await callback("bash", {}, context)
        assert isinstance(result, PermissionResultDeny)

    async def test_abort_interrupt_is_true(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("5"),
        ):
            result = await callback("bash", {}, context)
        assert result.interrupt is True

    async def test_abort_message(self, callback, context: MagicMock) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("5"),
        ):
            result = await callback("bash", {}, context)
        assert result.message == "Aborted by user."


# ---------------------------------------------------------------------------
# Tests — invalid choice loops back
# ---------------------------------------------------------------------------

class TestInvalidChoice:
    """Unrecognised input re-shows the menu and prompts again."""

    async def test_invalid_then_approve(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("x", "1"),
        ):
            result = await callback("bash", {}, context)
        assert isinstance(result, PermissionResultAllow)

    async def test_empty_input_loops(
        self, callback, context: MagicMock
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("", "5"),
        ):
            result = await callback("bash", {}, context)
        assert isinstance(result, PermissionResultDeny)


# ---------------------------------------------------------------------------
# Tests — terminal output
# ---------------------------------------------------------------------------

class TestTerminalOutput:
    """Callback always prints tool name, input, and menu before the prompt."""

    async def test_prints_tool_name(
        self, callback, context: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("5"),
        ):
            await callback("edit", {"file_path": "utils.py"}, context)
        assert "edit" in capsys.readouterr().out

    async def test_prints_menu(
        self, callback, context: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("5"),
        ):
            await callback("bash", {}, context)
        out = capsys.readouterr().out
        assert "Approve" in out
        assert "Ask" in out
        assert "Review" in out
        assert "Reject" in out
        assert "Abort" in out

    async def test_exitplanmode_prints_execution_plan_table(
        self, callback, context: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        # When ExitPlanMode is the tool, the full plan table must be shown.
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("5"),
        ):
            await callback("ExitPlanMode", _EXITPLANMODE_INPUT, context)
        out = capsys.readouterr().out
        assert "EXECUTION PLAN" in out
        assert "run pytest" in out
        assert "update utils.py" in out

    async def test_permission_header_includes_step_label_when_plan_active(
        self, callback, context: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        # After ExitPlanMode is approved, the next Bash call should embed
        # "Step 1/2: run pytest" in the [Permission Request] header.
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("1", "5"),
        ):
            await callback("ExitPlanMode", _EXITPLANMODE_INPUT, context)
            await callback("bash", {"command": "pytest"}, context)
        out = capsys.readouterr().out
        assert "Permission Request — Step 1/2: run pytest" in out

    async def test_permission_header_plain_when_no_plan(
        self, callback, context: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        # Without a plan loaded, the header is just [Permission Request].
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("5"),
        ):
            await callback("bash", {}, context)
        out = capsys.readouterr().out
        assert "[Permission Request]" in out
        assert "Step" not in out.split("[Permission Request]")[1].split("\n")[0]

    async def test_non_plan_tool_prints_context_bar_when_plan_exists(
        self, callback, context: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        # First approve ExitPlanMode to load the plan, then a Bash call should
        # show the context bar with the plan title.
        with patch(
            "util_claude.agent.human_in_the_loop.asyncio.get_event_loop",
            return_value=_make_loop("1", "5"),   # approve plan, then abort bash
        ):
            await callback("ExitPlanMode", _EXITPLANMODE_INPUT, context)
            await callback("bash", {"command": "pytest"}, context)
        out = capsys.readouterr().out
        assert "Harden utils.py" in out   # plan title in context bar
        assert "run pytest" in out         # current step description
