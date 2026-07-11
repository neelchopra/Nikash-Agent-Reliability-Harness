"""Integration: real docker container running mcp/filesystem, real stdio session.

Run explicitly with:  uv run pytest -m docker -v
Requires: Docker Desktop running and `docker pull mcp/filesystem` done once.
"""

import shutil

import pytest

from arh.mcp_client import MCPSession
from arh.models import GraderSpec, Task
from arh.runner.sandbox import TrialSandbox

pytestmark = pytest.mark.docker


def make_task(state_dir):
    return Task(
        id="t-docker", prompt="p", initial_state=state_dir,
        grader=GraderSpec(kind="file_match"),
    )


@pytest.fixture
def sandbox(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not on PATH")
    state = tmp_path / "state"
    state.mkdir()
    (state / "seed.txt").write_text("seeded", encoding="utf-8")
    sb = TrialSandbox(make_task(state), runs_root=tmp_path / "runs")
    sb.seed()
    yield sb
    sb.cleanup()


async def test_list_tools_and_roundtrip_write(sandbox):
    async with MCPSession(sandbox.server_params()) as mcp:
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "write_file" in names
        assert "read_text_file" in names or "read_file" in names

        text, ok = await mcp.call_tool(
            "write_file", {"path": "/work/out.txt", "content": "hello from mcp"}
        )
        assert ok, text
    # after the container exits, the write must be visible on the host side
    assert (sandbox.work_dir / "out.txt").read_text(encoding="utf-8") == "hello from mcp"


async def test_no_state_leakage_between_sandboxes(sandbox, tmp_path):
    async with MCPSession(sandbox.server_params()) as mcp:
        _, ok = await mcp.call_tool(
            "write_file", {"path": "/work/leak.txt", "content": "leak"}
        )
        assert ok
    # a SECOND sandbox from the same task must not see the first trial's write
    sb2 = TrialSandbox(sandbox.task, runs_root=tmp_path / "runs2")
    sb2.seed()
    try:
        assert not (sb2.work_dir / "leak.txt").exists()
        assert (sb2.work_dir / "seed.txt").read_text(encoding="utf-8") == "seeded"
    finally:
        sb2.cleanup()
