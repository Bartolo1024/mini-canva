"""Protocol tests use the installed official SDK client and real stdio server processes."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from marketcanvas_env.env import MarketCanvasEnv
from marketcanvas_env.models import NewElement
from marketcanvas_env.task_data import load_task, tasks_directory
from tests.reward_support.benchmarks import BENCHMARKS


@asynccontextmanager
async def client(cwd=None):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "marketcanvas_env.mcp_server"], cwd=cwd
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=15)
        ) as session:
            await session.initialize()
            yield session


def structured(result):
    assert not result.isError, result
    assert isinstance(result.structuredContent, dict), result
    return result.structuredContent


def error_code(result):
    assert result.isError, result
    return json.loads(result.content[0].text)["code"]


def test_yaml_constraints_reset_through_stdio(tmp_path):
    async def run():
        direct = MarketCanvasEnv()
        try:
            async with client(tmp_path) as session:
                for path in sorted(tasks_directory().glob("*.yaml")):
                    task = load_task(path)
                    _, info = direct.reset(seed=7, options={"target": task})
                    result = structured(
                        await session.call_tool(
                            "reset_canvas", {"seed": 7, **task.model_dump(mode="json")}
                        )
                    )
                    assert result["state"] == direct.get_canvas_state()
                    assert result["info"] == {"target_source": "structured"}
                    assert info == {"target_source": "structured"}
                    assert structured(await session.call_tool("get_current_reward", {})) == (
                        direct.get_current_reward()
                    )
                    assert structured(
                        await session.call_tool("execute_action", {"action": {"op": "finish"}})
                    ) == direct.execute_action({"op": "finish"})
        finally:
            direct.close()

    asyncio.run(run())


def test_discovery_complete_episode_and_direct_parity(tmp_path, caplog):
    async def run():
        direct = MarketCanvasEnv()
        direct.reset()
        async with client(tmp_path) as session:
            tools = {tool.name: tool for tool in (await session.list_tools()).tools}
            assert set(tools) == {
                "get_canvas_state",
                "get_current_reward",
                "execute_action",
                "reset_canvas",
            }
            action_schema = json.dumps(tools["execute_action"].inputSchema)
            for operation in (
                "add_element",
                "move_element",
                "update_element",
                "delete_element",
                "finish",
            ):
                assert operation in action_schema
            assert "constraints" in json.dumps(tools["reset_canvas"].inputSchema)
            for tool in tools.values():
                assert tool.outputSchema is not None
            assert (
                structured(await session.call_tool("get_canvas_state", {}))
                == direct.get_canvas_state()
            )
            actions = [
                {
                    "op": "add_element",
                    "element": {
                        "type": "text",
                        "role": "headline",
                        "content": "Summer Sale",
                        "width": 400,
                        "height": 80,
                    },
                },
                {
                    "op": "add_element",
                    "element": {
                        "type": "shape",
                        "subtype": "button",
                        "role": "cta",
                        "content": "Shop Now",
                        "color": "#FFFF00",
                        "x": 300,
                        "y": 250,
                    },
                },
                {"op": "add_element", "element": {"type": "image", "x": 650, "width": 100}},
                {"op": "move_element", "id": 1, "new_x": 200, "new_y": 80},
                {
                    "op": "update_element",
                    "id": 3,
                    "properties": NewElement(
                        type="image", x=650, width=100, color="#ABCDEF"
                    ).model_dump(exclude={"type", "subtype"}),
                },
                {"op": "delete_element", "id": 3},
                {"op": "delete_element", "id": 3},  # semantic failure still consumes one attempt
                {"op": "finish"},
            ]
            for action in actions:
                expected = direct.execute_action(action)
                actual = structured(await session.call_tool("execute_action", {"action": action}))
                assert actual == expected
                for _ in range(2):
                    assert (
                        structured(await session.call_tool("get_canvas_state", {}))
                        == expected["state"]
                    )
                    assert (
                        structured(await session.call_tool("get_current_reward", {}))
                        == direct.get_current_reward()
                    )
            assert (
                error_code(await session.call_tool("execute_action", {"action": {"op": "finish"}}))
                == "episode_done"
            )
            reset = structured(await session.call_tool("reset_canvas", {}))
            _, info = direct.reset()
            assert reset == {"state": direct.get_canvas_state(), "info": info}

    asyncio.run(run())
    # The SDK logs any non-protocol stdout it cannot parse as JSON-RPC.
    assert "Failed to parse" not in caplog.text


def test_malformed_arguments_are_atomic_and_have_stable_codes():
    async def run():
        async with client() as session:
            before = structured(await session.call_tool("get_canvas_state", {}))
            invalid = [
                ("get_canvas_state", {"extra": 1}),
                ("get_current_reward", {"extra": 1}),
                ("execute_action", {}),
                ("execute_action", {"action": None}),
                ("execute_action", {"action": '{"op":"finish"}'}),
                ("execute_action", {"action": {"op": "unknown"}}),
                ("execute_action", {"action": {"op": "finish"}, "extra": 1}),
                (
                    "execute_action",
                    {"action": {"op": "move_element", "id": True, "new_x": 0, "new_y": 0}},
                ),
                (
                    "execute_action",
                    {"action": {"op": "move_element", "id": 1, "new_x": "0", "new_y": 0}},
                ),
                (
                    "execute_action",
                    {"action": {"op": "move_element", "id": 1, "new_x": 0.0, "new_y": 0}},
                ),
                ("reset_canvas", {"prompt": None}),
                ("reset_canvas", {"target": None}),
                ("reset_canvas", {"seed": True}),
                ("reset_canvas", {"seed": "7"}),
                ("reset_canvas", {"seed": 7.0}),
                ("reset_canvas", {"target": {}}),
                ("reset_canvas", {"constraints": None}),
                ("reset_canvas", {"constraints": [], "target": {}}),
                ("reset_canvas", {"prompt": "anything", "target": {}}),
                ("reset_canvas", {"extra": 1}),
            ]
            for name, args in invalid:
                assert error_code(await session.call_tool(name, args)) == "invalid_request", (
                    name,
                    args,
                )
                assert structured(await session.call_tool("get_canvas_state", {})) == before
            assert (
                error_code(await session.call_tool("reset_canvas", {"prompt": "anything"}))
                == "not_implemented"
            )
            assert structured(await session.call_tool("get_canvas_state", {})) == before
            # Unknown tool uses the SDK error surface and never reaches the environment.
            unknown = await session.call_tool("not_a_tool", {})
            assert unknown.isError
            assert structured(await session.call_tool("get_canvas_state", {})) == before

    asyncio.run(run())


def test_tasks_budget_endings_and_process_isolation():
    async def run():
        async with client() as session, client() as other:
            original = structured(await other.call_tool("get_canvas_state", {}))
            task = BENCHMARKS["two_column"].task.model_dump(mode="json")
            direct = MarketCanvasEnv()
            _, info = direct.reset(seed=7, options={"target": task})
            reset = structured(await session.call_tool("reset_canvas", {"seed": 7, "target": task}))
            assert reset == {"state": direct.get_canvas_state(), "info": info}
            assert structured(await other.call_tool("get_canvas_state", {})) == original
            for _ in range(64):
                action = {"op": "delete_element", "id": 64}
                actual = structured(await session.call_tool("execute_action", {"action": action}))
                assert actual == direct.execute_action(action)
            assert actual["terminated"] and not actual["truncated"]
            assert actual["info"]["end_reason"] == "budget_exhausted"
            assert (
                error_code(await session.call_tool("execute_action", {"action": {"op": "finish"}}))
                == "episode_done"
            )
            assert structured(await other.call_tool("get_canvas_state", {})) == original
            reset = structured(await session.call_tool("reset_canvas", {"seed": None}))
            assert reset["state"] == original

    asyncio.run(run())


def test_all_tools_wait_for_shared_lock_and_definitions_are_detached():
    from marketcanvas_env.mcp_server import create_server

    async def run():
        server = create_server()
        definitions = await server.list_tools()
        definitions[0].inputSchema.clear()
        assert (await server.list_tools())[0].inputSchema
        async with server._lock:
            tasks = [
                asyncio.create_task(server.call_tool(name, arguments))
                for name, arguments in [
                    ("get_canvas_state", {}),
                    ("get_current_reward", {}),
                    ("execute_action", {"action": {"op": "finish"}}),
                    ("reset_canvas", {}),
                ]
            ]
            await asyncio.sleep(0)
            assert all(not task.done() for task in tasks)
        results = await asyncio.gather(*tasks)
        assert results[2]["terminated"]
        assert results[3]["state"]["steps_taken"] == 0

    asyncio.run(run())


def test_programming_fault_is_not_disguised_as_invalid_request(monkeypatch):
    from marketcanvas_env.mcp_server import create_server

    server = create_server()

    def broken():
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(server._env, "get_current_reward", broken)
    import pytest

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        asyncio.run(server.call_tool("get_current_reward", {}))
