"""
How to use the Claude Agent SDK.
"""
import asyncio
from pathlib import Path
from claude_agent_sdk import ClaudeSDKClient

async def run():
    repo = Path(__file__).parent

    async with ClaudeSDKClient(
            workspace=str(repo),
            allowed_commands=[
                "git",
                "python",
                "pytest",
                "grep",
                "read",
                "edit",
                "bash"
            ]
    ) as client:

        await client.query(
            "Review utils.py and make it robust. Then create and run pytest and summarize failures."
        )

        async for msg in client.receive_response():
            print(msg)

asyncio.run(run())


#
#from claude_agent_sdk import query, ClaudeAgentOptions
#
#async def __main():
#    async for message in query(
#            prompt="Find and fix the bug in auth.py",
#            options=ClaudeAgentOptions(allowed_tools=["Read", "Edit", "Bash"]),
#    ):
#        print(message)  # Claude reads the file, finds the bug, edits it

