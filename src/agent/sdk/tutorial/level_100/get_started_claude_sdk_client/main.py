"""
# How to use the Claude Agent SDK

* [Agent SDK reference - Python]
(https://platform.claude.com/docs/en/agent-sdk/python)

---
# ClaudeSDKClient

* [Class ClaudeSDKClient]
(https://platform.claude.com/docs/en/agent-sdk/python#claude-sdk-client)

Stateful client to maintain a conversation session across multiple exchanges
in a continuous conversations.

*  [Class - Methods](https://platform.claude.com/docs/en/agent-sdk/python#methods)
for available methods.

* [ClaudeAgentOptions]
(https://platform.claude.com/docs/en/agent-sdk/python#claude-agent-options) for
available options to configure the client.

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient

async with ClaudeSDKClient() as client:
    await client.query("Hello Claude")
    async for message in client.receive_response():
        print(message)
```

### Initialization
```python
def __init__(
    self,
    options: ClaudeAgentOptions | None = None,
    transport: Transport | None = None
)
```

"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import (
    Literal,
    Union,
)

# Resolve util_claude from the shared lib submodule so this script always uses
# the canonical library rather than a local copy.
# Layout: <repo_root>/lib/code/python/lib/util_claude
#   main.py lives 6 levels below the repo root (parents[6]).
_LIB_PATH = str(Path(__file__).parents[6] / "lib" / "code" / "python" / "lib")
if _LIB_PATH not in sys.path:
    sys.path.insert(0, _LIB_PATH)

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    UserMessage,
)

from util_claude.agent.human_in_the_loop import make_approval_callback
from util_claude.agent.dialog import print_agent_message
from util_claude.common import (
    assistant_message_oneliner,
    tool_name_map,
    user_message_oneliner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with a ``verbose`` bool attribute.
    """
    parser = argparse.ArgumentParser(
        description="Run ClaudeSDKClient with human-in-the-loop tool approval."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print all agent messages (tool calls, results, summaries).",
    )
    return parser.parse_args()


async def run(verbose: bool = False) -> None:
    """Run the ClaudeSDKClient with interactive permission approval.

    Args:
        verbose: When True, print every agent message via
            ``print_agent_message``.
    """
    directory: Union[str, Path, None] = str(Path(__file__).parent)

    # "default" routes every tool — including file edits — through the
    # can_use_tool callback so the user must approve or deny each one.
    permission: Union[
        Literal["default", "acceptEdits", "bypassPermissions", "plan"],
        None
    ] = "default"

    logger.info("Configuring agent (cwd=%s, permission_mode=%s)", directory, permission)

    # Clarification sessions are tool-free: Claude answers conversationally
    # without being able to run any tools.
    clarification_options = ClaudeAgentOptions(cwd=directory)

    options = ClaudeAgentOptions(
        cwd=directory,
        allowed_tools=[
            "git",
            "python",
            "pytest",
            "grep",
            "read",
            "edit",
            "bash",
        ],
        permission_mode=permission,
        # make_approval_callback returns a CanUseTool callback that supports
        # y/N/q, where q opens a short-lived clarification session.
        can_use_tool=make_approval_callback(clarification_options),
    )

    logger.info("Starting Claude Code session — launching subprocess and connecting...")
    async with ClaudeSDKClient(options=options) as client:
        logger.info("Session established.")

        logger.info("Sending query to cloud-hosted model...")
        await client.query(
            "Review utils.py and make it robust. "
            "Then create and run pytest and summarize failures."
            "Then update the file after the user approval."
        )

        logger.info("Query sent. Awaiting model response stream...")
        # Maps ToolUseBlock.id → ToolUseBlock.name from the most recent
        # AssistantMessage so user_message_oneliner can label each result.
        _last_tool_names: dict[str, str] = {}

        async for msg in client.receive_response():
            # Log a one-line summary so the user can track progress
            # even without --verbose.
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
            if verbose:
                print_agent_message(msg)

    logger.info("Session closed.")


args = _parse_args()
asyncio.run(run(verbose=args.verbose))
