"""Single-trial execution and the sequential n-trial loop.

Failure-source discipline (the reliability-critical invariant):
- InfraError (docker/MCP-transport/provider 429/5xx) -> retry up to 3 attempts;
  if still failing, record failure_source="infra". Infra rows are EXCLUDED from
  pass^k denominators by the report layer.
- Everything else the agent did wrong (wrong final state, timeout, gave up)
  -> failure_source="agent", counted.
"""

import asyncio
import uuid
from pathlib import Path

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from arh.adapters.litellm_tools import run_agent
from arh.errors import InfraError
from arh.grader.file_match import grade_file_match
from arh.mcp_client import MCPSession
from arh.models import Task, Trajectory, TrialResult
from arh.runner.sandbox import TrialSandbox

_MAX_ATTEMPTS = 3
_RETRY_WAIT_S = 2  # base for exponential backoff; monkeypatched to 0 in tests


async def run_trial(task: Task, model: str, runs_root: Path) -> TrialResult:
    sandbox = TrialSandbox(task, runs_root)
    sandbox.seed()
    try:
        try:
            async with MCPSession(sandbox.server_params()) as mcp:
                trajectory = await asyncio.wait_for(
                    run_agent(model, task, mcp), timeout=task.timeout_s
                )
        except TimeoutError:
            return TrialResult(
                trial_id=sandbox.trial_id,
                task_id=task.id,
                task_version=task.version,
                model=model,
                success=False,
                failure_source="agent",
                grade_detail=f"harness timeout after {task.timeout_s}s",
                trajectory=Trajectory(task_id=task.id, stop_reason="timeout"),
            )
        success, detail = grade_file_match(sandbox.work_dir, task.grader)
        return TrialResult(
            trial_id=sandbox.trial_id,
            task_id=task.id,
            task_version=task.version,
            model=model,
            success=success,
            failure_source="none" if success else "agent",
            grade_detail=detail,
            trajectory=trajectory,
        )
    finally:
        sandbox.cleanup()


async def run_task_trials(
    task: Task,
    model: str,
    n_trials: int,
    runs_root: Path,
    store,
    on_result=None,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for _ in range(n_trials):
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(InfraError),
                stop=stop_after_attempt(_MAX_ATTEMPTS),
                wait=wait_exponential(multiplier=_RETRY_WAIT_S, max=30),
                reraise=True,
            ):
                with attempt:
                    result = await run_trial(task, model, runs_root)
        except InfraError as e:
            result = TrialResult(
                trial_id=uuid.uuid4().hex[:12],
                task_id=task.id,
                task_version=task.version,
                model=model,
                success=False,
                failure_source="infra",
                grade_detail=f"infra failure after {_MAX_ATTEMPTS} attempts: {e}",
            )
        store.append(result)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
