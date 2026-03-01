"""Common utilities for extracting content from Claude Agent SDK messages.

---

## SDK Message Structure

``ClaudeSDKClient.receive_response()`` yields a stream of typed message objects.
Each type carries a different piece of the conversation:

```
AssistantMessage          ← Claude's response (text, reasoning, tool calls)
  .content: list[
      TextBlock           ← prose text intended for the human
        .text: str
      ThinkingBlock       ← internal chain-of-thought (extended thinking)
        .thinking: str
      ToolUseBlock        ← Claude wants to call a tool
        .name: str        ← tool name, e.g. "bash", "read", "edit"
        .input: dict      ← arguments the tool will be invoked with
  ]

UserMessage               ← tool results fed back to Claude after execution
  .content:
      str                 ← plain echo of the user's original query (skip it)
    | list[
          ToolResultBlock ← output returned by an executed tool
            .tool_use_id: str
            .content: str | list   ← stdout / file content / structured data
            .is_error: bool        ← True when the tool raised an exception
      ]

ResultMessage             ← end-of-session summary (after all turns complete)
  .is_error: bool
  .num_turns: int
  .total_cost_usd: float | None
  .duration_ms: int

SystemMessage             ← SDK lifecycle events (session init, etc.)
  .subtype: str

StreamEvent               ← raw Anthropic API stream data (rarely useful)
```

---

## What This Module Provides

Three groups of helpers:

1. **Message content extractors** — pull typed blocks out of ``AssistantMessage``
   and ``UserMessage`` so callers do not need to repeat ``isinstance`` filters.

2. **Tool input summariser** — ``summarise_tool_input`` converts a tool's
   argument dict into a concise human-readable string.  Each known tool type
   exposes its most actionable field (the shell command, the file path, the
   plan title) rather than dumping the raw dict.

3. **One-line log summaries** — ``assistant_message_oneliner`` and
   ``user_message_oneliner`` compress a full message into a single log line
   so ``logger.info`` output is informative without being verbose.
"""

from claude_agent_sdk import (
    AssistantMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

# Maximum characters shown for a single generic field value before truncating.
_MAX_VALUE_CHARS: int = 120


# ---------------------------------------------------------------------------
# Message content extractors
#
# AssistantMessage.content is a heterogeneous list.  These helpers isolate
# a single block type so callers receive a typed, homogeneous list instead
# of performing isinstance checks themselves.
# ---------------------------------------------------------------------------

def text_blocks(msg: AssistantMessage) -> list[str]:
    """Extract the text strings from every TextBlock in an AssistantMessage.

    ``AssistantMessage.content`` is a list that may contain ``TextBlock``,
    ``ThinkingBlock``, and ``ToolUseBlock`` objects in any order.  This
    function filters to ``TextBlock`` only and returns the raw text strings
    (``TextBlock.text: str``) — the prose Claude intends for the human.

    Example content list::

        [
            TextBlock(text="I'll run the tests now."),
            ToolUseBlock(name="bash", input={"command": "pytest -v"}),
        ]

    Calling ``text_blocks(msg)`` on the above returns::

        ["I'll run the tests now."]

    Args:
        msg: An ``AssistantMessage`` from ``receive_response()``.

    Returns:
        Ordered list of ``TextBlock.text`` strings, preserving the original
        block order.  Empty list if the message contains no text blocks.
    """
    return [b.text for b in msg.content if isinstance(b, TextBlock)]


def tool_use_blocks(msg: AssistantMessage) -> list[ToolUseBlock]:
    """Extract every ToolUseBlock from an AssistantMessage.

    ``AssistantMessage.content`` is a heterogeneous list.  This function
    filters to ``ToolUseBlock`` only.  Each ``ToolUseBlock`` represents one
    tool call Claude wants to make:

    * ``ToolUseBlock.name: str``  — tool name, e.g. ``"bash"``, ``"read"``
    * ``ToolUseBlock.input: dict`` — arguments, e.g. ``{"command": "ls -la"}``

    Example content list::

        [
            TextBlock(text="Let me check the file."),
            ToolUseBlock(name="read", input={"file_path": "utils.py"}),
            ToolUseBlock(name="bash", input={"command": "pytest -v"}),
        ]

    Calling ``tool_use_blocks(msg)`` returns the two ``ToolUseBlock`` objects.

    Args:
        msg: An ``AssistantMessage`` from ``receive_response()``.

    Returns:
        Ordered list of ``ToolUseBlock`` objects.  Empty list if the message
        contains no tool-use blocks.
    """
    return [b for b in msg.content if isinstance(b, ToolUseBlock)]


def tool_result_blocks(msg: UserMessage) -> list[ToolResultBlock]:
    """Extract every ToolResultBlock from a UserMessage.

    ``UserMessage.content`` is either:

    * A plain ``str`` — an echo of the user's own query sent at the start of
      the session.  There are no tool results to extract; return ``[]``.
    * A ``list`` — tool execution results fed back to Claude after each tool
      call.  Each ``ToolResultBlock`` in the list carries:

      * ``ToolResultBlock.tool_use_id: str`` — matches the ``ToolUseBlock.id``
        that triggered this result.
      * ``ToolResultBlock.content: str | list`` — stdout, file content, or
        structured data returned by the tool.
      * ``ToolResultBlock.is_error: bool`` — ``True`` when the tool raised an
        exception (e.g. permission denied, command not found).

    Example ``UserMessage.content`` list::

        [
            ToolResultBlock(
                tool_use_id="tu_abc",
                content="10 passed in 0.4s",
                is_error=False,
            ),
        ]

    Args:
        msg: A ``UserMessage`` from ``receive_response()``.

    Returns:
        Ordered list of ``ToolResultBlock`` objects.  Empty list when
        ``msg.content`` is a plain string (query echo).
    """
    if isinstance(msg.content, list):
        return [b for b in msg.content if isinstance(b, ToolResultBlock)]
    return []


# ---------------------------------------------------------------------------
# Tool input summariser
#
# Each tool the SDK exposes has a known input dict shape.  This function
# selects the most human-meaningful field(s) for each tool type rather than
# dumping the full dict, which may contain large blobs (e.g. the full plan
# markdown in ExitPlanMode).
#
# Input dict shapes by tool (only relevant keys documented here):
#
#   bash
#       input["command"]:     str  — the shell command to execute
#       input["description"]: str  — optional human-readable intent (may be absent)
#
#   read
#       input["file_path"]:   str  — absolute or relative path of the file to read
#
#   edit / write / multiedit
#       input["file_path"]:   str  — path of the file being created or modified
#
#   exitplanmode
#       input["plan"]:            str  — full plan as markdown; first line is the
#                                        heading "# Plan: <title>"
#       input["allowedPrompts"]:  list[{"tool": str, "prompt": str}]
#                                     — actions the agent is authorised to take
#
#   (all other tools)
#       All key/value pairs are printed; values longer than _MAX_VALUE_CHARS
#       characters are truncated with an ellipsis.
# ---------------------------------------------------------------------------

def summarise_tool_input(tool_name: str, tool_input: dict) -> str:
    """Return a concise, human-readable summary of a tool call's arguments.

    Selects the most actionable field(s) from ``tool_input`` for each known
    tool type.  The result is suitable for display in a permission menu or
    verbose log line.  It may be multi-line; callers should indent each line
    uniformly, e.g.::

        for line in summarise_tool_input(name, args).splitlines():
            print(f"    {line}")

    **Extraction rules by tool:**

    ``bash``
        Extracts ``tool_input["command"]`` (the shell command) and, when
        present, ``tool_input["description"]`` (the stated human intent).
        Output::

            <description>       ← omitted if absent
            $ <command>

    ``read``
        Extracts ``tool_input["file_path"]``.
        Output::

            file: <file_path>

    ``edit`` / ``write`` / ``multiedit``
        Extracts ``tool_input["file_path"]``.
        Output::

            file: <file_path>

    ``exitplanmode``
        Extracts the plan title from the first line of
        ``tool_input["plan"]`` (strips leading ``#`` and ``"Plan: "``),
        then lists each entry in ``tool_input["allowedPrompts"]`` as a
        bullet showing ``prompt["tool"]`` and ``prompt["prompt"]``.
        Output::

            plan   : <title>
            actions:
              • [<tool>] <prompt>
              • [<tool>] <prompt>

    **Generic fallback**
        All key/value pairs joined by newlines; values longer than
        ``_MAX_VALUE_CHARS`` (120) characters are truncated with ``…``.

    Args:
        tool_name: Name of the tool (case-insensitive).
        tool_input: Argument dict the SDK will pass to the tool
                    (``ToolUseBlock.input``).

    Returns:
        A non-empty string.  May contain ``\\n`` but no trailing newline.
    """
    tool = tool_name.lower()

    if tool == "bash":
        # tool_input["command"]     — the shell command string
        # tool_input["description"] — optional one-line statement of intent
        cmd = tool_input.get("command", "")
        desc = tool_input.get("description", "")
        if desc:
            return f"{desc}\n$ {cmd}"
        return f"$ {cmd}"

    if tool in ("edit", "write", "multiedit"):
        # tool_input["file_path"] — path of the file being created/modified
        return f"file: {tool_input.get('file_path', '(unknown)')}"

    if tool == "read":
        # tool_input["file_path"] — path of the file to read
        return f"file: {tool_input.get('file_path', '(unknown)')}"

    if tool == "exitplanmode":
        # tool_input["plan"]           — full markdown plan; first line is the heading
        # tool_input["allowedPrompts"] — list of {"tool": str, "prompt": str}
        plan_text = tool_input.get("plan", "")
        first_line = plan_text.splitlines()[0] if plan_text else ""
        # Strip leading "#" markers and the conventional "Plan: " prefix
        # so "# Plan: Harden utils.py" becomes "Harden utils.py".
        title = first_line.lstrip("#").strip()
        if title.lower().startswith("plan:"):
            title = title[len("plan:"):].strip()
        title = title or "(no title)"

        prompts = tool_input.get("allowedPrompts", [])
        lines = [f"plan   : {title}"]
        if prompts:
            lines.append("actions:")
            for p in prompts:
                # p["tool"]   — SDK tool class, e.g. "Bash"
                # p["prompt"] — natural-language description of the action
                lines.append(f"  • [{p.get('tool', '?')}] {p.get('prompt', '')}")
        return "\n".join(lines)

    # Generic fallback: all fields with long values truncated.
    parts = []
    for key, value in tool_input.items():
        text = str(value)
        if len(text) > _MAX_VALUE_CHARS:
            text = text[:_MAX_VALUE_CHARS] + "…"
        parts.append(f"{key}: {text}")
    return "\n".join(parts) if parts else "(no arguments)"


# ---------------------------------------------------------------------------
# Tool-use ID → name mapping
#
# ToolResultBlock.tool_use_id matches ToolUseBlock.id from the preceding
# AssistantMessage.  Building this map lets user_message_oneliner show the
# tool name alongside the result status without re-inspecting the stream.
# ---------------------------------------------------------------------------

def tool_name_map(msg: AssistantMessage) -> dict[str, str]:
    """Build a mapping from tool_use_id to tool name from an AssistantMessage.

    Each ``ToolUseBlock`` in the message carries:

    * ``ToolUseBlock.id: str``   — unique identifier for this tool call;
      matches ``ToolResultBlock.tool_use_id`` in the subsequent UserMessage.
    * ``ToolUseBlock.name: str`` — tool name, e.g. ``"Bash"``, ``"Read"``.

    Pass the returned dict to ``user_message_oneliner`` so the UserMessage
    log line can show which tool produced each result:

    Example::

        # In the receive_response() loop:
        names = {}
        if isinstance(msg, AssistantMessage):
            names = tool_name_map(msg)
        elif isinstance(msg, UserMessage):
            logger.info(user_message_oneliner(msg, names))

    Args:
        msg: An ``AssistantMessage`` from ``receive_response()``.

    Returns:
        ``{ToolUseBlock.id: ToolUseBlock.name}`` for every tool-use block
        in the message.  Empty dict if the message contains no tool calls.
    """
    return {b.id: b.name for b in tool_use_blocks(msg)}


# ---------------------------------------------------------------------------
# One-line message summaries for log output
#
# Used by main.py to emit a human-readable logger.info line for each message
# received from receive_response(), replacing the bare type-name log.
#
# Format:
#   AssistantMessage — Run Tool: Read(utils.py)
#   UserMessage      — Read OK: <first 60 chars of output>
# ---------------------------------------------------------------------------

# Maximum characters shown for the key argument in a tool call label.
_MAX_KEY_ARG_CHARS: int = 50


def _tool_key_arg(name: str, inp: dict) -> str:
    """Return the single most meaningful argument value for a tool call label.

    Used by ``assistant_message_oneliner`` to produce ``ToolName(key_arg)``
    labels.  Selects the most actionable field for each known tool type:

    * ``bash``                       → ``inp["command"]``   (the shell command)
    * ``read`` / ``edit`` / ``write`` / ``multiedit`` → ``inp["file_path"]``
    * ``exitplanmode``               → ``""``  (plan approval needs no arg)
    * any other                      → first value in the dict

    Values longer than ``_MAX_KEY_ARG_CHARS`` are truncated with ``…``.

    Args:
        name: Tool name (case-insensitive).
        inp:  ``ToolUseBlock.input`` dict.

    Returns:
        A short string, possibly empty.
    """
    tool = name.lower()
    if tool == "bash":
        val = inp.get("command", "")
    elif tool in ("read", "edit", "write", "multiedit"):
        val = inp.get("file_path", "")
    elif tool == "exitplanmode":
        return ""
    else:
        val = str(next(iter(inp.values()))) if inp else ""
    return val[:_MAX_KEY_ARG_CHARS] + "…" if len(val) > _MAX_KEY_ARG_CHARS else val


def assistant_message_oneliner(msg: AssistantMessage) -> str:
    """Return a single-line description of an AssistantMessage for logging.

    Format::

        Run Tool: Read(utils.py)
        Run Tools: Read(f.py), Bash(pytest -v)
        says: "I will now analyse the repository…"
        Run Tool: Bash(pytest -v) | says: "Running the test suite."
        (no content)

    For each tool call, the key argument is extracted via ``_tool_key_arg``
    and shown in parentheses after the tool name:

    * ``ToolUseBlock.name``  — tool name, e.g. ``"Read"``
    * ``ToolUseBlock.input`` — argument dict; ``_tool_key_arg`` selects
      ``input["command"]`` for Bash, ``input["file_path"]`` for file tools.

    Args:
        msg: An ``AssistantMessage`` from ``receive_response()``.

    Returns:
        A non-empty single-line string.
    """
    tools = tool_use_blocks(msg)
    texts = text_blocks(msg)
    parts: list[str] = []
    if tools:
        label = "Run Tools" if len(tools) > 1 else "Run Tool"
        calls = []
        for b in tools:
            key = _tool_key_arg(b.name, b.input)
            calls.append(f"{b.name}({key})" if key else b.name)
        parts.append(f"{label}: {', '.join(calls)}")
    if texts:
        # TextBlock.text — first 80 chars of Claude's prose, newlines collapsed
        snippet = texts[0][:80].replace("\n", " ")
        ellipsis = "…" if len(texts[0]) > 80 else ""
        parts.append(f'says: "{snippet}{ellipsis}"')
    return " | ".join(parts) if parts else "(no content)"


def user_message_oneliner(
    msg: UserMessage,
    tool_names: dict[str, str] | None = None,
) -> str:
    """Return a single-line description of a UserMessage for logging.

    Reads ``ToolResultBlock.tool_use_id`` (matched against ``tool_names`` to
    recover the tool name), ``ToolResultBlock.is_error`` (OK / ERROR status),
    and the first 60 characters of ``ToolResultBlock.content`` (the tool's
    output) to produce a compact log line.

    When ``msg.content`` is a plain string (the query echo at session start)
    the function returns ``"(query echo)"`` to signal that no tool result is
    present.

    ``tool_names`` should be the dict returned by ``tool_name_map`` called on
    the immediately preceding ``AssistantMessage``::

        names = {}
        if isinstance(msg, AssistantMessage):
            names = tool_name_map(msg)
        elif isinstance(msg, UserMessage):
            logger.info(user_message_oneliner(msg, names))

    Examples::

        "Bash OK: 10 passed in 0.4s"
        "Bash ERROR: Permission denied…"
        "OK: 10 passed"        ← when tool_names not supplied
        "(query echo)"

    Args:
        msg: A ``UserMessage`` from ``receive_response()``.
        tool_names: Optional mapping of ``ToolUseBlock.id`` →
            ``ToolUseBlock.name``, built from the preceding
            ``AssistantMessage`` via ``tool_name_map``.  When provided,
            each result entry is prefixed with the tool name.

    Returns:
        A non-empty single-line string.
    """
    results = tool_result_blocks(msg)
    if not results:
        # msg.content is a plain str — just the original query echoed back.
        return "(query echo)"
    names = tool_names or {}
    parts: list[str] = []
    for b in results:
        # ToolResultBlock.tool_use_id — matches ToolUseBlock.id
        # ToolResultBlock.is_error    — True when tool execution failed
        # ToolResultBlock.content     — stdout / file content / error message
        tool = names.get(b.tool_use_id, "")
        status = "ERROR" if b.is_error else "OK"
        prefix = f"{tool} {status}" if tool else status
        content = str(b.content or "")
        snippet = content[:60].replace("\n", " ")
        ellipsis = "…" if len(content) > 60 else ""
        entry = f"{prefix}: {snippet}{ellipsis}" if snippet else prefix
        parts.append(entry)
    return ", ".join(parts)
