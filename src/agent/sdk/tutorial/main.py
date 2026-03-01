import asyncio
from claude_agent_sdk import ClaudeSDKClient
from pathlib import Path

async def run():
    repo = Path(__file__).parent

    async with ClaudeSDKClient(
            workspace=str(repo),
            allowed_commands=[
                "git",
                "python",
                "pytest",
                "grep"
            ]
    ) as client:

        await client.query(
            "Run git status. Then run pytest and summarize failures."
        )

        async for msg in client.receive_response():
            print(msg)

asyncio.run(run())