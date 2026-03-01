"""Agent message display helpers for Claude Agent SDK responses."""

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UserMessage,
)

from util_claude.common import (
    summarise_tool_input,
    tool_result_blocks,
)

# Maximum number of content lines shown for a tool result before truncating.
_TOOL_RESULT_PREVIEW_LINES: int = 10


def _truncate_lines(text: str, max_lines: int) -> str:
    """Return text truncated to max_lines with a count of omitted lines.

    Args:
        text: The string to truncate.
        max_lines: Maximum number of lines to keep.

    Returns:
        The original text when it fits within max_lines, otherwise the first
        max_lines lines followed by a ``... (+N more lines)`` marker.
    """
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted: int = len(lines) - max_lines
    preview: str = "\n".join(lines[:max_lines])
    return f"{preview}\n  ... (+{omitted} more lines)"


def print_agent_message(msg: object) -> None:
    """Pretty-print a single message yielded by ``receive_response()``.

    Dispatches on the concrete message type so each variant is rendered in
    a human-readable format instead of dumping the raw dataclass repr.
    Tool arguments are formatted via ``summarise_tool_input`` and long tool
    result content is truncated to ``_TOOL_RESULT_PREVIEW_LINES`` lines.

    Args:
        msg: A ``Message`` object (``AssistantMessage``, ``UserMessage``,
            ``ResultMessage``, ``SystemMessage``, or ``StreamEvent``) as
            returned by ``ClaudeSDKClient.receive_response()``.
    """
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(f"\n[Claude]\n{block.text}")
            elif isinstance(block, ThinkingBlock):
                print("\n[Thinking] ...")
            elif isinstance(block, ToolUseBlock):
                print(f"\n[Tool] {block.name}")
                for line in summarise_tool_input(block.name, block.input).splitlines():
                    print(f"  {line}")

    elif isinstance(msg, UserMessage):
        # UserMessage wraps tool results fed back to the model.
        # Plain string content is just an echo of our own query — skip it.
        for block in tool_result_blocks(msg):
            status: str = "ERROR" if block.is_error else "OK"
            raw = block.content or ""
            preview: str = (
                _truncate_lines(raw, _TOOL_RESULT_PREVIEW_LINES)
                if isinstance(raw, str)
                else str(raw)
            )
            print(f"  [Tool Result/{status}]\n{preview}")

    elif isinstance(msg, ResultMessage):
        cost: str = (
            f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd else "N/A"
        )
        status = "FAILED" if msg.is_error else "OK"
        print(
            f"\n[Done] status={status}  turns={msg.num_turns}"
            f"  cost={cost}  duration={msg.duration_ms}ms"
        )

    elif isinstance(msg, SystemMessage):
        print(f"\n[System/{msg.subtype}]")

    # StreamEvent carries raw stream data from the Anthropic API and is not
    # meaningful to display in a human-readable log.
