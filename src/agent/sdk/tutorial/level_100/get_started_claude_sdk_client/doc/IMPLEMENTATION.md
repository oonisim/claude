# Implementation Notes

---

## Module Overview

```
util_claude/
├── common.py                   # Message extractors, tool summariser, log one-liners
└── agent/
    ├── human_in_the_loop.py    # HITL approval callback + AgentExecutionPlanTracker
    └── dialog.py               # Verbose agent message display
```

---

## `util_claude/common.py` — Shared Utilities

### Problem

Multiple modules need to filter typed blocks out of heterogeneous SDK message
objects (`AssistantMessage.content`, `UserMessage.content`) and format them for
display. Repeating `isinstance` filters and truncation logic in every caller
leads to duplication and inconsistency.

### Solution — three groups of helpers

#### 1. Message content extractors

```python
text_blocks(msg: AssistantMessage)       -> list[str]
tool_use_blocks(msg: AssistantMessage)   -> list[ToolUseBlock]
tool_result_blocks(msg: UserMessage)     -> list[ToolResultBlock]
```

Each filters one block type out of the heterogeneous `.content` list so callers
receive a typed, homogeneous list without writing `isinstance` checks themselves.

#### 2. Tool input summariser — `summarise_tool_input`

```python
summarise_tool_input(tool_name: str, tool_input: dict) -> str
```

Selects the most actionable field(s) from the argument dict for each known tool
type rather than dumping the full dict (which may contain large blobs, such as
the full plan Markdown in `ExitPlanMode`):

| Tool | Fields shown |
|---|---|
| `bash` | `input["command"]` + optional `input["description"]` |
| `read` / `edit` / `write` / `multiedit` | `input["file_path"]` |
| `exitplanmode` | Plan title (first heading line) + `allowedPrompts` bullets |
| Any other | All key/value pairs; values > 120 chars truncated with `…` |

#### 3. One-line log summaries

```python
assistant_message_oneliner(msg: AssistantMessage) -> str
user_message_oneliner(msg: UserMessage, tool_names: dict | None) -> str
tool_name_map(msg: AssistantMessage) -> dict[str, str]
```

Compress a full SDK message into a single `logger.info` line.

**Format:**
```
AssistantMessage — Run Tool: Read(utils.py)
AssistantMessage — Run Tools: Read(f.py), Bash(pytest -v)
AssistantMessage — says: "I will now analyse the repository."
UserMessage      — Read OK: def divide(a, b): return a / b
UserMessage      — Bash ERROR: Permission denied.
```

`"AssistantMessage —"` and `"UserMessage      —"` (5-space pad) are the same
character width so columns align in the terminal.

`tool_name_map` builds a `{ToolUseBlock.id → ToolUseBlock.name}` dict from the
preceding `AssistantMessage` so `user_message_oneliner` can label each result
by tool name without re-examining the stream.

#### Sentence-boundary truncation — `pysbd`

Text snippets in one-line summaries are truncated at a **natural sentence
boundary** rather than a fixed character count using
[pysbd](https://github.com/nipunsadvilkar/pysbd):

```python
try:
    import pysbd as _pysbd
    _SENTENCE_SEGMENTER = _pysbd.Segmenter(language="en", clean=False)
except ImportError:
    _SENTENCE_SEGMENTER = None   # falls back to hard slice

def _first_sentence(text: str, max_chars: int) -> tuple[str, bool]:
    flat = text.replace("\n", " ")
    if len(flat) <= max_chars:
        return flat, False          # fits — no truncation needed
    if _SENTENCE_SEGMENTER is not None:
        sentences = _SENTENCE_SEGMENTER.segment(flat)
        first = sentences[0].rstrip()
        if len(first) <= max_chars:
            return first, True      # ends at sentence boundary
    return flat[:max_chars], True   # fallback: hard slice
```

The segmenter is created once at module load to avoid per-call overhead.

---

## Human-in-the-Loop Tool Approval

### Problem

Without a permission handler the SDK returns `"This command requires approval"`
as an error `ToolResultBlock` for any tool it cannot auto-approve, and the tool
call is silently skipped.

```
UserMessage(
    content=[ToolResultBlock(
        tool_use_id='...',
        content='This command requires approval',
        is_error=True,
    )],
    ...
)
```

With `permission_mode="acceptEdits"` this affects shell commands (e.g. `bash`).
With `permission_mode="default"` it affects **all** tools, including file edits.

A secondary problem: when `PermissionResultDeny(interrupt=False)` is returned,
the SDK surfaces the denial as a tool error and Claude retries the same call in
the next turn, causing an infinite approval loop.

### Solution — `make_approval_callback` factory

`ClaudeAgentOptions` exposes a typed async callback slot:

```python
CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResult]         # Allow | Deny
]
```

`make_approval_callback(clarification_options)` is a factory that returns a
`CanUseTool` closure. The factory accepts `clarification_options` — a separate
`ClaudeAgentOptions` configured without tools — so the closure can open
short-lived sessions for clarification questions.

Register it in `main.py`:

```python
clarification_options = ClaudeAgentOptions(cwd=directory)  # no tools

options = ClaudeAgentOptions(
    ...
    permission_mode="default",
    can_use_tool=make_approval_callback(clarification_options),
)
```

### `AgentExecutionPlanTracker`

`make_approval_callback` creates one `AgentExecutionPlanTracker` instance that
persists across all tool calls in the session. It intercepts `ExitPlanMode`
calls to extract the agent's execution plan and then tracks progress step by
step.

```python
class AgentExecutionPlanTracker:
    title: str                          # plan heading, e.g. "Harden utils.py"
    _steps: list[tuple[str, str]]       # [(tool_type, description), …]
    _current: int                       # index of next pending planned step
    _loaded: bool                       # True after first successful load
```

**`load(tool_input)`** — parses `ExitPlanMode.input`:

```python
# tool_input["plan"]           → first heading line → title
# tool_input["allowedPrompts"] → list[{"tool": str, "prompt": str}] → _steps
```

Returns `True` when the agent has revised a previously loaded plan.

**`print_full_plan(changed)`** — Oracle-style numbered table printed once when
`ExitPlanMode` is intercepted:

```
╔════════════════════════════════════════════════════════════════════════════════╗
║  EXECUTION PLAN  ·  Harden utils.py                                           ║
╠────────────────────────────────────────────────────────────────────────────────╣
║  Step    Tool          Action                                                  ║
║  ──────  ────────────  ────────────────────────────────────────────────────── ║
║  ▶ 1.    Bash          run pytest on test_utils.py                             ║
║    2.    Bash          update utils.py with robust implementation              ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

Box inner width is `EXECUTION_PLAN_BOX_WIDTH = 80` characters so plan
descriptions are not clipped at the standard terminal column count.

**`print_context_bar(tool_name)`** — compact plan status bar printed above every
non-plan tool's permission menu:

```
  ────────────────────────────────────────────────────────────────────────────────
  Plan: Harden utils.py
  ▶ Step 1/2: run pytest on test_utils.py
  ────────────────────────────────────────────────────────────────────────────────
```

Shows `⟳ Exploring` when the tool does not match any planned step, and
`✓ All N planned step(s) completed` when the plan is exhausted.

**`step_label(tool_name)`** — returns `"Step N/T: description"` embedded
directly in the `[Permission Request]` header:

```
[Permission Request — Step 1/2: run pytest on test_utils.py]
```

**`try_advance(tool_name)`** — advances `_current` by one when the approved tool
matches the current planned step type (case-insensitive).

**`_find_step(tool_name)`** — scans `_steps` from `_current` onwards; returns
`(1-based step number, description)` or `None`.

### Menu

```python
_MENU: str = (
    "  1) Approve — proceed with execution\n"
    "  2) Ask     — type a question for the agent before deciding\n"
    "  3) Review  — show command / file details\n"
    "  4) Reject  — skip this tool, let agent try another approach\n"
    "  5) Abort   — terminate the session (agent may re-propose a new plan)"
)
```

| Choice | Behaviour | `interrupt` |
|---|---|---|
| `1` Approve | `PermissionResultAllow()` — SDK executes the tool; `try_advance` called | — |
| `2` Ask | Prompt user for a question; if empty → auto-explain; model answers in short-lived session; menu repeats | — |
| `3` Review | Calls `_show_review`; menu repeats | — |
| `4` Reject | `PermissionResultDeny(interrupt=False)` — Claude sees denial, may try another approach | `False` |
| `5` Abort | `PermissionResultDeny(interrupt=True)` — SDK stops the run immediately | `True` |
| other | Prints error, loops | — |

### Option 2 — Ask (two paths)

```python
if choice == "2":
    raw = await loop.run_in_executor(
        None, input, "  Your question (Enter for auto-explain): "
    )
    question = raw.strip()
    if question:
        prompt = _CLARIFICATION_PROMPT.format(
            tool_name=tool_name, tool_input=tool_input, question=question
        )
    else:
        prompt = _EXPLAIN_PROMPT.format(
            tool_name=tool_name, tool_input=tool_input
        )
    await _ask_for_clarification(tool_name, tool_input, prompt, clarification_options)
    continue
```

- **User types a question** → `_CLARIFICATION_PROMPT`: wraps the question with tool context.
- **User presses Enter (empty)** → `_EXPLAIN_PROMPT`: asks the model to justify the tool call (why it is necessary, what it changes, whether a less invasive alternative exists).

### Review helper — `_show_review`

Option 3 calls `_show_review(tool_name, tool_input)` which dispatches on tool
type to show the most useful pre-execution context:

| Tool | What is shown |
|---|---|
| `bash` | Full command + optional description |
| `edit` / `write` / `multiedit` | File path + `git diff -- <file>` (≤ 30 lines; failure handled gracefully) |
| `read` | File path, byte size, line count |
| Any other | All `tool_input` key/value pairs |

### Clarification session — `_ask_for_clarification`

Both Ask paths open a **separate** `ClaudeSDKClient` instance so the exchange
does not interfere with the main session paused on the permission request.

```python
async def _ask_for_clarification(tool_name, tool_input, question, options):
    async with ClaudeSDKClient(options=options) as client:
        await client.query(question)
        async for msg in client.receive_response():
            print_agent_message(msg)
```

The clarification `ClaudeAgentOptions` is configured with `cwd` only — no
`allowed_tools` — so the model responds conversationally.

### Control Flow

```
Claude decides to call a tool (edit, bash, pytest, …)
        │
        ▼
SDK: permission_mode="default" → all tools require approval
        │
        ▼
SDK awaits make_approval_callback closure (tool_name, tool_input, ctx)
        │
        ├── ExitPlanMode ──► AgentExecutionPlanTracker.load()
        │                    AgentExecutionPlanTracker.print_full_plan()
        │
        └── other tools  ──► AgentExecutionPlanTracker.print_context_bar()
        │
        ▼
Shows [Permission Request — Step N/T: …] + 5-option menu; awaits choice
        │
   [1] ──► PermissionResultAllow()              ──► try_advance(); SDK executes the tool
        │
   [2] ──► prompt user for question
        │       │
        │       ├── non-empty → _CLARIFICATION_PROMPT
        │       └── empty     → _EXPLAIN_PROMPT
        │       │
        │       ▼
        │   short-lived ClaudeSDKClient (no tools)
        │   → model answers / justifies the action
        │   → printed via print_agent_message
        │       │
        └───────┘  loops back to plan bar + menu
        │
   [3] ──► _show_review() (command / diff / file info)
        │       │
        └───────┘  loops back to plan bar + menu
        │
   [4] ──► PermissionResultDeny(interrupt=False)  ──► Claude may retry
        │
   [5] ──► PermissionResultDeny(interrupt=True)   ──► SDK aborts the run
        │
 [other] ──► "Invalid choice" message, loops back to menu
```

### Permission Mode Reference

| Mode | Behaviour |
|---|---|
| `"default"` | All tools (including file edits) require approval — routed to `can_use_tool` |
| `"acceptEdits"` | File edits auto-approved; other tools require a handler |
| `"bypassPermissions"` | All tools run without any approval check |
| `"plan"` | Planning/review mode; execution is restricted |

---

## Progress Logging

### Problem

`ClaudeSDKClient` startup involves launching a subprocess and authenticating,
which can take 10+ seconds. Without feedback the terminal appears frozen.
Raw `"Received: AssistantMessage"` log lines give no information about what
Claude is actually doing.

### Solution — one-line summaries in `main.py`

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
```

Session lifecycle events use descriptive messages. Stream messages use
`assistant_message_oneliner` / `user_message_oneliner` from `util_claude/common.py`:

```python
_last_tool_names: dict[str, str] = {}

async for msg in client.receive_response():
    if isinstance(msg, AssistantMessage):
        _last_tool_names = tool_name_map(msg)
        logger.info("AssistantMessage — %s", assistant_message_oneliner(msg))
    elif isinstance(msg, UserMessage):
        logger.info(
            "UserMessage      — %s",
            user_message_oneliner(msg, _last_tool_names),
        )
    else:
        logger.info("Received: %s", type(msg).__name__)
```

`_last_tool_names` carries the `{tool_use_id → name}` mapping from the
preceding `AssistantMessage` so `user_message_oneliner` can label each tool
result by name.

**Sample output:**
```
09:41:02  Configuring agent (cwd=..., permission_mode=default)
09:41:02  Starting Claude Code session — launching subprocess and connecting...
09:41:14  Session established.
09:41:14  Sending query to cloud-hosted model...
09:41:14  Query sent. Awaiting model response stream...
09:41:19  AssistantMessage — Run Tool: ExitPlanMode
09:41:20  UserMessage      — (query echo)
09:41:22  AssistantMessage — Run Tool: Read(utils.py)
09:41:23  UserMessage      — Read OK: def divide(a, b): return a / b
09:41:25  AssistantMessage — Run Tool: Bash(pytest test_utils.py -v)
09:41:28  UserMessage      — Bash OK: 3 failed, 5 passed in 0.6s
09:42:10  Session closed.
```

---

## Agent Message Display

### Problem

`receive_response()` yields raw SDK message objects. Printing them directly
produces verbose, unreadable output. In particular, tool results such as file
reads dump the entire file content to the terminal.

### Solution — `print_agent_message` in `util_claude/agent/dialog.py`

`print_agent_message` dispatches on the concrete message type and renders only
the human-relevant parts. Long `ToolResultBlock` content is truncated to a
configurable number of preview lines.

### Implementation — `util_claude/agent/dialog.py`

```python
_TOOL_RESULT_PREVIEW_LINES: int = 10

def _truncate_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted: int = len(lines) - max_lines
    preview: str = "\n".join(lines[:max_lines])
    return f"{preview}\n  ... (+{omitted} more lines)"

def print_agent_message(msg: object) -> None:
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(f"\n[Claude]\n{block.text}")
            elif isinstance(block, ThinkingBlock):
                print("\n[Thinking] ...")
            elif isinstance(block, ToolUseBlock):
                print(f"\n[Tool] {block.name}")
                args = summarise_tool_input(block.name, block.input)
                for line in args.splitlines():
                    print(f"  {line}")
    elif isinstance(msg, UserMessage):
        for block in tool_result_blocks(msg):
            status = "ERROR" if block.is_error else "OK"
            raw = str(block.content or "")
            preview = _truncate_lines(raw, _TOOL_RESULT_PREVIEW_LINES)
            print(f"\n[Tool Result/{status}]\n{preview}")
    elif isinstance(msg, ResultMessage):
        cost = f"${msg.total_cost_usd:.4f}" if msg.total_cost_usd else "N/A"
        status = "FAILED" if msg.is_error else "OK"
        print(f"\n[Done] status={status}  turns={msg.num_turns}"
              f"  cost={cost}  duration={msg.duration_ms}ms")
    elif isinstance(msg, SystemMessage):
        print(f"\n[System/{msg.subtype}]")
    # StreamEvent: skipped — raw Anthropic API stream data
```

| Message type | Output |
|---|---|
| `AssistantMessage` / `TextBlock` | `[Claude]` prose |
| `AssistantMessage` / `ThinkingBlock` | `[Thinking] ...` placeholder |
| `AssistantMessage` / `ToolUseBlock` | `[Tool] <name>` + summarised args |
| `UserMessage` / `ToolResultBlock` | `[Tool Result/OK\|ERROR]` + truncated content |
| `ResultMessage` | `[Done]` status, turns, cost, duration |
| `SystemMessage` | `[System/<subtype>]` |
| `StreamEvent` | *(skipped)* |

### Verbose Flag

`print_agent_message` is called only when `-v`/`--verbose` is passed to
`main.py`. In non-verbose mode the session runs with logging only.

```python
# main.py
async for msg in client.receive_response():
    logger.info("AssistantMessage — %s", assistant_message_oneliner(msg))
    if verbose:
        print_agent_message(msg)
```

```bash
python main.py           # logging only
python main.py --verbose # logging + full truncated message detail
```
