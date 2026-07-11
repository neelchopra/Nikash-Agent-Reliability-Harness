from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class GraderSpec(BaseModel):
    kind: Literal["file_match"]
    expect_files: list[str] = Field(default_factory=list)
    forbid_files: list[str] = Field(default_factory=list)  # glob patterns, e.g. "*.txt"
    expect_content: dict[str, str] = Field(default_factory=dict)  # file -> required substring


class Task(BaseModel):
    id: str
    version: int = 1
    prompt: str
    docker_image: str = "mcp/filesystem"
    server_args: list[str] = Field(default_factory=lambda: ["/work"])
    network: str = "none"
    mem_limit: str = "512m"
    initial_state: Path
    grader: GraderSpec
    timeout_s: int = 120
    max_turns: int = 15
    n_trials: int = 10


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    latency_ms: int = 0
    ok: bool = True


class Trajectory(BaseModel):
    task_id: str
    turns: list[dict] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: Literal["completed", "max_turns", "timeout", "error"] = "completed"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0


class TrialResult(BaseModel):
    trial_id: str
    task_id: str
    task_version: int
    model: str
    adapter: str = "litellm-tools"
    success: bool
    failure_source: Literal["agent", "infra", "none"]
    grade_detail: str = ""
    trajectory: Trajectory | None = None
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
