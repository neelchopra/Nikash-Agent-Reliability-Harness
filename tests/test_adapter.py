import json
from types import SimpleNamespace

import litellm
import pytest

from arh.adapters.litellm_tools import mcp_tools_to_openai, run_agent
from arh.errors import InfraError
from arh.models import GraderSpec, Task


def make_task() -> Task:
    return Task(
        id="t1", prompt="rename the files", initial_state=".",
        grader=GraderSpec(kind="file_match"), max_turns=5,
    )


class FakeMCP:
    def __init__(self):
        self.calls = []

    async def list_tools(self):
        return [SimpleNamespace(
            name="write_file",
            description="write a file",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
        )]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return "wrote it", True


def fake_message(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    msg.model_dump = lambda **kw: {
        "role": "assistant",
        "content": content,
        **({"tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls]} if tool_calls else {}),
    }
    return msg


def fake_response(msg):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def test_mcp_tools_to_openai_shape():
    tools = [SimpleNamespace(name="x", description="d", inputSchema={"type": "object"})]
    out = mcp_tools_to_openai(tools)
    assert out == [{
        "type": "function",
        "function": {"name": "x", "description": "d", "parameters": {"type": "object"}},
    }]


async def test_tool_call_then_completion(monkeypatch):
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(
        name="write_file", arguments=json.dumps({"path": "/work/a.md"})))
    responses = iter([
        fake_response(fake_message(tool_calls=[tc])),
        fake_response(fake_message(content="done")),
    ])

    async def fake_acompletion(**kwargs):
        return next(responses)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    mcp = FakeMCP()
    traj = await run_agent("fake/model", make_task(), mcp)
    assert traj.stop_reason == "completed"
    assert len(traj.tool_calls) == 1
    assert traj.tool_calls[0].name == "write_file"
    assert traj.tool_calls[0].ok
    assert mcp.calls == [("write_file", {"path": "/work/a.md"})]
    assert traj.input_tokens == 200 and traj.output_tokens == 40


async def test_max_turns_cap(monkeypatch):
    def always_tool_call():
        tc = SimpleNamespace(id="c1", function=SimpleNamespace(
            name="write_file", arguments="{}"))
        return fake_response(fake_message(tool_calls=[tc]))

    async def fake_acompletion(**kwargs):
        return always_tool_call()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    traj = await run_agent("fake/model", make_task(), FakeMCP())
    assert traj.stop_reason == "max_turns"
    assert len(traj.tool_calls) == 5  # max_turns


async def test_malformed_arguments_recorded_not_raised(monkeypatch):
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(
        name="write_file", arguments="{not json"))
    responses = iter([
        fake_response(fake_message(tool_calls=[tc])),
        fake_response(fake_message(content="giving up")),
    ])

    async def fake_acompletion(**kwargs):
        return next(responses)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    mcp = FakeMCP()
    traj = await run_agent("fake/model", make_task(), mcp)
    assert traj.stop_reason == "completed"
    assert len(traj.tool_calls) == 1
    assert not traj.tool_calls[0].ok
    assert "malformed" in traj.tool_calls[0].error
    assert mcp.calls == []  # tool never actually executed


async def test_provider_rate_limit_raises_infra_error(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise litellm.RateLimitError(
            message="429", llm_provider="fake", model="fake/model")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    with pytest.raises(InfraError):
        await run_agent("fake/model", make_task(), FakeMCP())
