"""
For building an agent that can find and fix bugs in code.
This file has simple functions that the agent will review and improve.

[Quickstart]
(https://platform.claude.com/docs/en/agent-sdk/quickstart#build-an-agent-that-finds-and-fixes-bugs)
"""
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)


def get_user_name(user):
    return user["name"].upper()