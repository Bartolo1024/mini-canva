"""Run a benchmark through a real MCP stdio client and print its final state and reward."""

import argparse
import asyncio
import json
import sys
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from marketcanvas_env.task_data import load_task, load_trajectory, tasks_directory, trajectory_paths


def _result(response):
    if response.isError:
        raise RuntimeError(response.content)
    return response.structuredContent


async def run(name: str, trajectory: str = "well_done") -> None:
    path = next((p for p in trajectory_paths(name) if p.stem == trajectory), None)
    if path is None:
        raise ValueError(f"Unknown trajectory {name}/{trajectory}")
    run_data = load_trajectory(path)
    task = load_task(run_data["task"] + ".yaml")
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "marketcanvas_env.mcp_server"]
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=15)
        ) as session:
            await session.initialize()
            print("Tools:", ", ".join(tool.name for tool in (await session.list_tools()).tools))
            _result(
                await session.call_tool(
                    "reset_canvas",
                    {"seed": run_data["seed"], **task.model_dump(mode="json")},
                )
            )
            for action in run_data["actions"]:
                final = _result(await session.call_tool("execute_action", {"action": action}))
                if not final["info"]["action_applied"]:
                    raise RuntimeError(final["info"])
            print(json.dumps(final, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=[p.stem for p in sorted(tasks_directory().glob("*.yaml"))],
        default="default_task",
    )
    parser.add_argument("--trajectory", default="well_done", help="trajectory filename stem")
    args = parser.parse_args()
    asyncio.run(run(args.benchmark, args.trajectory))


if __name__ == "__main__":
    main()
