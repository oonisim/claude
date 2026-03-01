"""Human-in-the-loop tool approval for Claude Agent SDK sessions."""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import (
    Callable,
    Union,
)

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from util_claude.agent.dialog import print_agent_message
from util_claude.common import summarise_tool_input

logger = logging.getLogger(__name__)

# Display width for plan boxes and context bars.
_W: int = 60


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

_MENU: str = (
    "  1) Approve — proceed with execution\n"
    "  2) Ask     — type a question for the agent before deciding\n"
    "  3) Review  — show command / file details\n"
    "  4) Reject  — skip this tool, let agent try another approach\n"
    "  5) Abort   — terminate the session (agent may re-propose a new plan)"
)

# ---------------------------------------------------------------------------
# Prompts sent to the clarification session
# ---------------------------------------------------------------------------

# Used when the user selects Ask (option 2) and types their own question.
_CLARIFICATION_PROMPT: str = (
    "The user is reviewing a pending tool call and has a question before "
    "deciding whether to allow or deny it. Answer concisely.\n\n"
    "Pending tool call:\n"
    "  Tool  : {tool_name}\n"
    "  Input : {tool_input}\n\n"
    "User question: {question}"
)

# Used when the user selects Ask but presses Enter without typing anything,
# asking Claude to self-justify the action.
_EXPLAIN_PROMPT: str = (
    "The user is reviewing a pending tool call and wants you to justify it "
    "before deciding whether to allow or deny it. Explain concisely:\n"
    "  1. Why this tool call is necessary for the task.\n"
    "  2. What it will change or produce.\n"
    "  3. Whether a less invasive alternative exists.\n\n"
    "Pending tool call:\n"
    "  Tool  : {tool_name}\n"
    "  Input : {tool_input}"
)

# ---------------------------------------------------------------------------
# Execution plan tracker
# ---------------------------------------------------------------------------

class _PlanTracker:
    """Tracks the agent's execution plan extracted from ExitPlanMode calls.

    ExitPlanMode.input carries two fields used here:

    * ``plan: str``            — full plan as Markdown; first line is the
                                 heading ``# Plan: <title>``
    * ``allowedPrompts: list`` — ordered list of ``{"tool": str, "prompt": str}``
                                 entries that describe each planned action

    The tracker shows the plan as a numbered table when first received and
    displays a compact status bar above every subsequent permission request
    so the human always sees where they are in the overall plan.

    Step matching uses the ``tool`` field of each ``allowedPrompts`` entry
    (e.g. ``"Bash"``, ``"Read"``) compared case-insensitively against the
    current tool name.  When approved, the tracker advances past that step.
    Unplanned tool calls (no matching step) are shown as "Exploring" so the
    human can see that the agent is doing preparatory work.
    """

    def __init__(self) -> None:
        self.title: str = ""
        # Each entry: (tool_type: str, description: str)
        self._steps: list[tuple[str, str]] = []
        # Index of the next pending planned step (0-based).
        self._current: int = 0
        # True after the first successful load — used to detect plan revisions.
        self._loaded: bool = False

    @property
    def has_plan(self) -> bool:
        return bool(self._steps)

    @property
    def total(self) -> int:
        return len(self._steps)

    def load(self, tool_input: dict) -> bool:
        """Extract plan from ExitPlanMode.input.

        Parses:
        * ``tool_input["plan"]``           → title from first heading line
        * ``tool_input["allowedPrompts"]`` → list of (tool, prompt) steps

        Args:
            tool_input: The ``input`` dict from a ``ExitPlanMode`` ToolUseBlock.

        Returns:
            ``True`` when a previous plan exists and the new plan differs
            (i.e. the agent has revised its plan mid-session).
        """
        plan_text = tool_input.get("plan", "")
        first_line = plan_text.splitlines()[0] if plan_text else ""
        # Strip "# Plan: " prefix → plain title
        title = first_line.lstrip("#").strip()
        if title.lower().startswith("plan:"):
            title = title[len("plan:"):].strip()

        prompts = tool_input.get("allowedPrompts", [])
        # allowedPrompts[i]["tool"]   — SDK tool class, e.g. "Bash"
        # allowedPrompts[i]["prompt"] — natural-language step description
        new_steps: list[tuple[str, str]] = [
            (p.get("tool", ""), p.get("prompt", "")) for p in prompts
        ]

        changed = self._loaded and (new_steps != self._steps or title != self.title)
        self.title = title or "Execution Plan"
        self._steps = new_steps
        self._current = 0
        self._loaded = True
        return changed

    def _find_step(self, tool_name: str) -> tuple[int, str] | None:
        """Return ``(step_number, description)`` for the first pending step matching tool_name.

        Scans ``_steps`` from ``_current`` onwards.  Comparison is
        case-insensitive against ``allowedPrompts[i].tool``.

        Args:
            tool_name: Name of the tool to match.

        Returns:
            ``(1-based step number, description)`` on a match, or ``None``
            when no pending step has a matching tool type.
        """
        tl = tool_name.lower()
        for i in range(self._current, len(self._steps)):
            if self._steps[i][0].lower() == tl:
                return (i + 1, self._steps[i][1])
        return None

    def step_label(self, tool_name: str) -> str | None:
        """Return a short label identifying which execution-plan step this tool call executes.

        The label is suitable for embedding directly in the ``[Permission
        Request]`` header so the human can see the plan step at a glance
        without reading the context bar.

        Example return values::

            "Step 1/2: run pytest on test_utils.py"
            "Step 2/2: update utils.py with robust implementation"
            None     ← tool does not match any pending planned step

        Args:
            tool_name: Name of the tool awaiting approval.

        Returns:
            A formatted string when the tool matches a pending planned step,
            otherwise ``None``.
        """
        match = self._find_step(tool_name)
        if match:
            step_num, desc = match
            return f"Step {step_num}/{self.total}: {desc}"
        return None

    def try_advance(self, tool_name: str) -> None:
        """Advance to the next step if tool_name matches the current planned step.

        Compares ``tool_name`` (case-insensitive) against
        ``allowedPrompts[_current].tool``.  Advances ``_current`` by one
        when they match.  Has no effect when there is no current step or
        when the tool type does not match (unplanned exploratory call).

        Args:
            tool_name: Name of the tool that was just approved.
        """
        if (self._current < len(self._steps) and
                self._steps[self._current][0].lower() == tool_name.lower()):
            self._current += 1

    def print_full_plan(self, *, changed: bool = False) -> None:
        """Print the full numbered execution plan table.

        Called when ``ExitPlanMode`` is intercepted.  Prints a bordered table
        with one row per ``allowedPrompts`` entry.  When ``changed`` is True
        a warning header is shown to signal that the agent has revised its
        plan.

        Args:
            changed: Set to ``True`` when a prior plan exists and the new
                plan differs — indicates the agent revised its plan.
        """
        bar_top = f"╔{'═' * _W}╗"
        bar_mid = f"╠{'─' * _W}╣"
        bar_bot = f"╚{'═' * _W}╝"

        if changed:
            print(f"\n{'━' * (_W + 2)}")
            print(f"  ⚠  PLAN UPDATED — agent has revised the execution plan")
            print(f"{'━' * (_W + 2)}")

        print(f"\n{bar_top}")
        header = f"  EXECUTION PLAN  ·  {self.title}"
        print(f"║{header:<{_W}}║")
        print(bar_mid)

        col = f"  {'Step':<8}{'Tool':<14}Action"
        print(f"║{col:<{_W}}║")
        div = f"  {'─'*6}  {'─'*12}  {'─' * (_W - 24)}"
        print(f"║{div:<{_W}}║")

        for i, (tool_type, desc) in enumerate(self._steps):
            if i < self._current:
                marker = "✓"
            elif i == self._current:
                marker = "▶"
            else:
                marker = " "
            num = f"{i + 1}."
            line = f"  {marker} {num:<5}  {tool_type:<12}  {desc}"
            print(f"║{line:<{_W}}║")

        print(f"{bar_bot}\n")

    def print_context_bar(self, tool_name: str) -> None:
        """Print a compact plan status header above a permission request.

        Shows:
        * The plan title
        * The current step (if the tool matches a planned step) with ``▶``
        * An "Exploring" line when the tool does not match any planned step,
          with the next planned step shown for context
        * A "Completed" line when all planned steps are done

        Args:
            tool_name: Name of the tool awaiting approval.
        """
        if not self.has_plan:
            return

        matched = self._find_step(tool_name)

        bar = "─" * _W
        title_disp = (
            self.title[:_W - 9] + "…" if len(self.title) > _W - 9 else self.title
        )
        print(f"  {bar}")
        print(f"  Plan: {title_disp}")

        if matched:
            step_num, desc = matched
            print(f"  ▶ Step {step_num}/{self.total}: {desc}")
        elif self._current < self.total:
            next_tool, next_desc = self._steps[self._current]
            print(
                f"  ⟳ Exploring  "
                f"(next planned: Step {self._current + 1} — {next_tool}: {next_desc})"
            )
        else:
            print(f"  ✓ All {self.total} planned step(s) completed")

        print(f"  {bar}")


# ---------------------------------------------------------------------------
# Review helper
# ---------------------------------------------------------------------------

_DIFF_PREVIEW_LINES: int = 30


def _show_review(tool_name: str, tool_input: dict) -> None:
    """Print human-readable detail about the pending tool call.

    Args:
        tool_name: Name of the tool awaiting approval.
        tool_input: Arguments the tool will be invoked with.
    """
    print("\n[Review]")
    tool = tool_name.lower()

    if tool == "bash":
        print(f"  Command    : {tool_input.get('command', '')}")
        description = tool_input.get("description", "")
        if description:
            print(f"  Description: {description}")

    elif tool in ("edit", "write", "multiedit"):
        file_path: str = tool_input.get("file_path", "")
        print(f"  File: {file_path}")
        try:
            result = subprocess.run(
                ["git", "diff", "--", file_path],
                capture_output=True, text=True, timeout=5,
            )
            diff_lines = result.stdout.splitlines()
            if diff_lines:
                preview = "\n".join(f"    {ln}" for ln in diff_lines[:_DIFF_PREVIEW_LINES])
                omitted = len(diff_lines) - _DIFF_PREVIEW_LINES
                suffix = f"\n    ... (+{omitted} more lines)" if omitted > 0 else ""
                print(f"  Diff:\n{preview}{suffix}")
            else:
                print("  (no committed diff yet — file will be modified)")
        except Exception as exc:
            logger.debug("git diff failed: %s", exc)
            print("  (could not retrieve diff)")

    elif tool == "read":
        file_path = tool_input.get("file_path", "")
        print(f"  File: {file_path}")
        try:
            p = Path(file_path)
            if p.exists():
                size = p.stat().st_size
                line_count = sum(1 for _ in p.open(errors="replace"))
                print(f"  Size: {size} bytes  ({line_count} lines)")
        except Exception as exc:
            logger.debug("file stat failed: %s", exc)

    else:
        for key, value in tool_input.items():
            print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# Clarification session
# ---------------------------------------------------------------------------

async def _ask_for_clarification(
    tool_name: str,
    tool_input: dict,
    question: str,
    clarification_options: ClaudeAgentOptions,
) -> None:
    """Open a short-lived session to answer a question about a pending tool call.

    Args:
        tool_name: Name of the tool awaiting approval.
        tool_input: Arguments the tool will be invoked with.
        question: Question sent to the model (user-typed or auto-generated).
        clarification_options: Options for the short-lived session — typically
            configured without tools so the model responds conversationally.
    """
    logger.info("Sending clarification question to model...")
    async with ClaudeSDKClient(options=clarification_options) as client:
        await client.query(question)
        logger.info("Awaiting clarification response...")
        async for msg in client.receive_response():
            logger.info("Clarification received: %s", type(msg).__name__)
            print_agent_message(msg)
    logger.info("Clarification complete.")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_approval_callback(
    clarification_options: ClaudeAgentOptions,
) -> Callable:
    """Return a ``CanUseTool`` callback with plan tracking and a structured menu.

    Each call to the returned callback follows this flow:

    **ExitPlanMode** (agent proposes or revises a plan)
        The plan table is printed in full before showing the permission menu.
        Approval signals to the SDK that the agent may proceed with execution.
        If the agent has already proposed a plan earlier in the session, the
        updated plan is highlighted with a ``⚠ PLAN UPDATED`` banner.

    **All other tools**
        A compact context bar shows the plan title and which planned step the
        current tool call corresponds to (or an "Exploring" line when the tool
        does not match any pending planned step).

    Menu options:

    1. **Approve** — allow the tool call; advance the plan step counter if the
       tool matches the current planned step.
    2. **Ask** — type a question for the agent; the model answers in a
       short-lived session, then the menu repeats so the human can decide.
       Pressing Enter without typing uses an auto-generated explain prompt.
    3. **Review** — show command / diff / file details, then repeat the menu.
    4. **Reject** — deny this tool call with an optional reason; Claude may
       try another approach (``interrupt=False``).
    5. **Abort** — terminate the session immediately (``interrupt=True``).

    Args:
        clarification_options: ``ClaudeAgentOptions`` for the short-lived
            session opened on Ask. Typically configured without tools.

    Returns:
        An async callable matching the ``CanUseTool`` signature:
        ``(tool_name, tool_input, context) -> PermissionResult``.
    """
    _tracker = _PlanTracker()

    async def _callback(
        tool_name: str,
        tool_input: dict,
        context: ToolPermissionContext,
    ) -> Union[PermissionResultAllow, PermissionResultDeny]:
        loop = asyncio.get_event_loop()
        is_plan_tool = tool_name.lower() == "exitplanmode"

        # When the agent proposes (or revises) a plan, load it and display
        # the full execution plan table before showing the permission menu.
        if is_plan_tool:
            changed = _tracker.load(tool_input)
            _tracker.print_full_plan(changed=changed)

        while True:
            # Show compact plan context bar for non-plan tool calls so the
            # human always sees where this action fits in the overall plan.
            if not is_plan_tool:
                _tracker.print_context_bar(tool_name)

            step = _tracker.step_label(tool_name) if not is_plan_tool else None
            header = f"Permission Request — {step}" if step else "Permission Request"
            print(f"\n[{header}]")
            print(f"  Tool : {tool_name}")
            print(f"  Input:")
            for line in summarise_tool_input(tool_name, tool_input).splitlines():
                print(f"    {line}")
            print(f"\n{_MENU}")

            choice: str = await loop.run_in_executor(
                None, input, "\n  Choice > "
            )
            choice = choice.strip()

            if choice == "1":
                # Advance the plan step counter if this tool matches the
                # current planned step.
                if not is_plan_tool:
                    _tracker.try_advance(tool_name)
                return PermissionResultAllow()

            if choice == "2":
                # Prompt the human for their question.  Pressing Enter without
                # typing falls back to the auto-generated explain prompt.
                raw: str = await loop.run_in_executor(
                    None,
                    input,
                    "  Your question (Enter for auto-explain): ",
                )
                question = raw.strip()
                if question:
                    prompt = _CLARIFICATION_PROMPT.format(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        question=question,
                    )
                else:
                    prompt = _EXPLAIN_PROMPT.format(
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                await _ask_for_clarification(
                    tool_name, tool_input, prompt, clarification_options
                )
                continue  # re-show plan bar + menu after the answer

            if choice == "3":
                _show_review(tool_name, tool_input)
                continue  # re-show plan bar + menu after review

            if choice == "4":
                reason: str = await loop.run_in_executor(
                    None, input, "  Reason (optional, Enter to skip): "
                )
                reason = reason.strip()
                message: str = (
                    f"Rejected by user: {reason}" if reason else "Rejected by user."
                )
                print(f"  Rejected — agent will try another approach.")
                return PermissionResultDeny(message=message, interrupt=False)

            if choice == "5":
                print("  Aborting session.")
                return PermissionResultDeny(
                    message="Aborted by user.", interrupt=True
                )

            print("  Invalid choice — enter 1, 2, 3, 4, or 5.")

    return _callback
