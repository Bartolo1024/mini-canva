"""Emit project context for a Codex SessionStart hook; no application logic."""

import json


def main() -> None:
    context = (
        "Read AGENTS.md and docs/PROJECT_PLAN.md before working on MarketCanvas. "
        "The initial request is preparation only; a later user request to implement supersedes it. "
        "Keep simulator behavior deterministic and RL/MCP semantics shared. "
        "Use the project agents and skills for bounded work when useful."
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context,
                }
            }
        )
    )


if __name__ == "__main__":
    main()
