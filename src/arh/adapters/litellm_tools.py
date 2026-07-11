"""Native function-calling agent loop over LiteLLM.

Contract: agent mistakes (bad JSON args, tool errors) are recorded in the
Trajectory and fed back to the model - never raised. Provider errors
(429/5xx/connection) raise InfraError so the runner can retry the trial.
"""

import json
import time

import litellm

from arh.errors import InfraError
from arh.models import Task, ToolCall, Trajectory

SYSTEM_PROMPT = (
    "You are an agent that completes tasks in a sandboxed environment using the "
    "available tools. Inspect the environment before acting. When the task is "
    "fully complete, reply with a short summary and make no further tool calls."
)

_INFRA_LLM_ERRORS = (
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
    litellm.Timeout,
)

_RESULT_TRUNCATE = 8000


def mcp_tools_to_openai(tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]


async def run_agent(model: str, task: Task, mcp, temperature: float = 0.7) -> Trajectory:
    t0 = time.monotonic()
    tools = mcp_tools_to_openai(await mcp.list_tools())
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    traj = Trajectory(task_id=task.id)
    stop_reason = "max_turns"

    for _ in range(task.max_turns):
        try:
            resp = await litellm.acompletion(
                model=model, messages=messages, tools=tools, temperature=temperature
            )
        except _INFRA_LLM_ERRORS as e:
            raise InfraError(f"LLM provider error: {e}") from e

        usage = getattr(resp, "usage", None)
        if usage is not None:
            traj.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            traj.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        try:
            traj.cost_usd += litellm.completion_cost(completion_response=resp)
        except Exception:
            pass  # unknown/fake models have no pricing entry

        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            stop_reason = "completed"
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                err = f"malformed JSON arguments: {e}"
                traj.tool_calls.append(ToolCall(name=name, error=err, ok=False))
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": f"ERROR: {err}"}
                )
                continue
            t_call = time.monotonic()
            result_text, ok = await mcp.call_tool(name, arguments)
            traj.tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    result=result_text[:_RESULT_TRUNCATE],
                    error=None if ok else result_text[:1000],
                    ok=ok,
                    latency_ms=int((time.monotonic() - t_call) * 1000),
                )
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_text[:_RESULT_TRUNCATE]}
            )

    traj.stop_reason = stop_reason
    traj.turns = [m if isinstance(m, dict) else dict(m) for m in messages]
    traj.wall_ms = int((time.monotonic() - t0) * 1000)
    return traj
