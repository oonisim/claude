# Get Started — Claude Agent SDK Client

A runnable example of using `ClaudeSDKClient` to drive a stateful multi-turn
conversation with Claude, with human-in-the-loop (HITL) tool approval,
Oracle-style execution plan tracking, sentence-boundary log summaries, and
structured response printing.

---

## What This Example Demonstrates

| Concept | Where |
|---|---|
| Stateful session via `ClaudeSDKClient` | `main.py` — `run()` |
| Configuring the agent with `ClaudeAgentOptions` | `main.py` — `run()` |
| Restricting available tools via `allowed_tools` | `main.py` — `run()` |
| Controlling permissions via `permission_mode` | `main.py` — `run()` |
| HITL approval: approve / ask / review / reject / abort | `util_claude/agent/human_in_the_loop.py` — `make_approval_callback()` |
| Oracle-style execution plan display and step tracking | `util_claude/agent/human_in_the_loop.py` — `AgentExecutionPlanTracker` |
| One-line log summaries with sentence-boundary truncation (`pysbd`) | `util_claude/common.py` — `assistant_message_oneliner`, `user_message_oneliner` |
| Message content extractors and tool input summariser | `util_claude/common.py` |
| Structured, truncated response printing | `util_claude/agent/dialog.py` — `print_agent_message()` |
| Verbose flag to enable full message printing | `main.py` — `-v`/`--verbose` |
| Sample code under review/test | `utils.py` |

---

## Files

```
.
├── main.py                              # Entry point: ClaudeSDKClient, argparse, logging
├── utils.py                             # Sample module Claude reviews and tests
├── test_utils.py                        # pytest file created/run by Claude at runtime
├── requirements.txt                     # Python dependencies (includes pysbd)
├── pytest.ini                           # pytest configuration
├── conftest.py                          # sys.path setup for tests
├── util_claude/
│   ├── common.py                        # Message extractors, tool summariser, log oneliners
│   └── agent/
│       ├── human_in_the_loop.py         # HITL approval menu + AgentExecutionPlanTracker
│       └── dialog.py                    # Agent message display (print_agent_message)
├── test/
│   └── claude/
│       ├── test_common.py               # Tests for util_claude/common.py
│       └── agent/
│           ├── test_callback.py         # Tests for human_in_the_loop.py
│           └── test_dialog.py           # Tests for dialog.py
└── doc/
    └── IMPLEMENTATION.md                # Design notes
```

---

## How It Works

`main.py` launches a `ClaudeSDKClient` session and asks Claude to:

1. Review `utils.py` and make it robust.
2. Create and run `pytest` tests.
3. Summarise any failures.
4. Update the file after user approval.

Claude executes the task autonomously, calling tools (`read`, `edit`, `bash`,
`pytest`, etc.) as needed. Four mechanisms give the programmer control:

### 1. Progress logging — informative one-line summaries

`main.py` emits a `logger.info` line for every message received from
`receive_response()`. The log format uses sentence-boundary truncation
(`pysbd`) so snippets always end at a natural sentence boundary rather
than mid-word:

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
...
09:42:10  Session closed.
```

`"AssistantMessage —"` and `"UserMessage      —"` (5-space pad) are the
same width so columns align cleanly.

### 2. `permission_mode` — coarse-grained policy

```python
permission: Union[Literal["default", "acceptEdits",
                          "bypassPermissions", "plan"], None] = "default"
```

`"default"` requires explicit approval for **every** tool including file edits.

### 3. `make_approval_callback()` — HITL approval menu with plan tracking

Defined in `util_claude/agent/human_in_the_loop.py`. A factory that returns a
`CanUseTool` callback registered on `ClaudeAgentOptions`:

```python
clarification_options = ClaudeAgentOptions(cwd=directory)  # no tools
options = ClaudeAgentOptions(
    ...
    permission_mode="default",
    can_use_tool=make_approval_callback(clarification_options),
)
```

#### Execution plan display

When Claude calls `ExitPlanMode` to commit to a plan, the callback prints
an Oracle-style numbered table before the permission menu:

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

For every subsequent tool call a compact context bar shows where in the plan
the current step falls:

```
  ────────────────────────────────────────────────────────────────────────────────
  Plan: Harden utils.py
  ▶ Step 1/2: run pytest on test_utils.py
  ────────────────────────────────────────────────────────────────────────────────
```

The `[Permission Request]` header embeds the step label inline:

```
[Permission Request — Step 1/2: run pytest on test_utils.py]
  Tool : Bash
  Input:
    run pytest on test_utils.py
    $ pytest test_utils.py -v
```

#### Five-option menu

| Choice | Action | `interrupt` |
|---|---|---|
| `1` Approve | Allow the tool — `PermissionResultAllow()`; advances plan step counter | — |
| `2` Ask | Type a question; model answers in a short-lived session; menu repeats. Press Enter for auto-explain. | — |
| `3` Review | Show command / diff / file details; menu repeats | — |
| `4` Reject | Skip this tool; Claude may try another approach — `PermissionResultDeny(interrupt=False)` | `False` |
| `5` Abort | Terminate the entire session — `PermissionResultDeny(interrupt=True)` | `True` |

**Reject vs Abort**: Reject feeds the denial back to Claude as a tool error so it
can try a different strategy. Abort stops the run immediately.

**Ask — two paths**:
- *User types a question* → a `_CLARIFICATION_PROMPT` wrapping their question is sent to a short-lived session.
- *User presses Enter (empty)* → an auto-generated `_EXPLAIN_PROMPT` asks the model to justify the action.

```
[Permission Request — Step 1/2: run pytest on test_utils.py]
  Tool : Bash
  Input:
    $ pytest test_utils.py -v

  1) Approve — proceed with execution
  2) Ask     — type a question for the agent before deciding
  3) Review  — show command / file details
  4) Reject  — skip this tool, let agent try another approach
  5) Abort   — terminate the session (agent may re-propose a new plan)

  Choice > 2
  Your question (Enter for auto-explain): why not use unittest instead?

[Claude]
pytest is already installed and provides a richer output format…

[Permission Request — Step 1/2: run pytest on test_utils.py]
  ...
  Choice > 1
```

**Option 3 — Review** shows context tailored to the tool type:

| Tool | What is shown |
|---|---|
| `bash` | Full command and description |
| `edit` / `write` / `multiedit` | File path + `git diff -- <file>` (truncated to 30 lines) |
| `read` | File path, byte size, line count |
| Any other | All `tool_input` key/value pairs |

Clarification sessions are **separate** `ClaudeSDKClient` instances so querying
the model during a paused permission request does not affect the main session.

### 4. `print_agent_message()` — readable, truncated response output

Defined in `util_claude/agent/dialog.py`. Enabled with `-v`/`--verbose`:

```bash
python main.py --verbose
```

Long tool result content is truncated to 10 lines with a `... (+N more lines)`
marker.

| Message type | Printed as |
|---|---|
| `AssistantMessage` / `TextBlock` | `[Claude]` — prose response |
| `AssistantMessage` / `ToolUseBlock` | `[Tool]` — tool name and args |
| `UserMessage` / `ToolResultBlock` | `[Tool Result/OK\|ERROR]` — truncated output |
| `ResultMessage` | `[Done]` — turns, cost, duration |
| `SystemMessage` | `[System/<subtype>]` |
| `StreamEvent` | *(skipped)* |

See [`doc/IMPLEMENTATION.md`](doc/IMPLEMENTATION.md) for full design notes.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

## Running

```bash
# Approval prompts and progress logs; no message detail
python main.py

# Full verbose message log with truncated tool results
python main.py --verbose
python main.py -v
```

## Running Tests

```bash
pytest test/ -v
```
