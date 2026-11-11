"""openJiuwen ReAct coding agent (optional; requires LLM_* env)."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional

from harness.cost import cost_from_usage
from harness.types import DEFAULT_MODEL, ModelName, Result, Status, Task, TokenUsage
from harness.usage import extract_usage


CODING_SYSTEM_PROMPT = """You are a coding agent working ONLY inside the current directory.
Goal: make `python3 test_mylib.py` pass by editing mylib.py only. Do not edit the tests.
When all tests pass, write exactly DONE to answer.txt (e.g. `echo DONE > answer.txt`) and stop.
Prefer short shell/python commands. Do not invent files outside this directory.
"""


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def build_react_agent(
    work_dir: str,
    *,
    model_name: str,
    max_iterations: int = 8,
    system_prompt: str = CODING_SYSTEM_PROMPT,
) -> Any:
    """Construct a configured ReActAgent with local fs/code/shell tools.

    Raises ImportError if openjiuwen is not installed.
    """
    from openjiuwen.core.runner import Runner
    from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig
    from openjiuwen.core.sys_operation import (
        LocalWorkConfig,
        OperationMode,
        SysOperationCard,
    )

    # Ensure runner is up (idempotent enough for contest container).
    try:
        loop_running = False
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False
        if not loop_running:
            # Runner.start is async in 0.1.x; fire if not started.
            try:
                asyncio.run(_ensure_runner_started())
            except RuntimeError:
                pass
    except Exception:
        pass

    sysop = SysOperationCard(
        id=f"sysop_{uuid.uuid4().hex[:8]}",
        mode=OperationMode.LOCAL,
        work_config=LocalWorkConfig(work_dir=work_dir),
    )
    Runner.resource_mgr.add_sys_operation(sysop)

    api_key = _env("LLM_API_KEY", "")
    api_base = _env("LLM_BASE_URL", "")
    provider = _env("LLM_PROVIDER", "OpenAI") or "OpenAI"

    card = AgentCard(id=f"coding_{uuid.uuid4().hex[:8]}", name="coding_agent")
    agent = ReActAgent(card=card)
    config = (
        ReActAgentConfig()
        .configure_model_client(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            verify_ssl=_env("LLM_SSL_VERIFY", "false").lower() == "true",
        )
        .configure_prompt_template([{"role": "system", "content": system_prompt}])
        .configure_max_iterations(max_iterations)
    )
    config.sys_operation_id = sysop.id
    agent.configure(config)

    for operation_name, tool_name in (
        ("fs", "read_file"),
        ("fs", "write_file"),
        ("code", "execute_code"),
        ("shell", "execute_cmd"),
    ):
        try:
            tool_card = Runner.resource_mgr.get_sys_op_tool_cards(
                sys_operation_id=sysop.id,
                operation_name=operation_name,
                tool_name=tool_name,
            )
        except Exception:
            tool_card = None
        if tool_card is not None:
            try:
                agent.ability_manager.add(tool_card)
            except Exception:
                pass

    return agent


async def _ensure_runner_started() -> None:
    from openjiuwen.core.runner import Runner

    start = getattr(Runner, "start", None)
    if start is None:
        return
    maybe = start()
    if asyncio.iscoroutine(maybe):
        await maybe


def run_coding_react(
    task: Task,
    *,
    max_iterations: int = 8,
    model: Optional[str] = None,
) -> Result:
    """Synchronously run ReAct coding agent for one task."""
    if not _env("LLM_API_KEY") or not _env("LLM_BASE_URL"):
        return Result(
            status=Status.FAILED,
            cost=0,
            error="LLM_API_KEY/LLM_BASE_URL not set",
            meta={"solver": "coding_react"},
        )

    model_name = model or DEFAULT_MODEL.value
    work_dir = str(task.work_dir)

    async def _run() -> Result:
        await _ensure_runner_started()
        agent = build_react_agent(
            work_dir,
            model_name=model_name,
            max_iterations=max_iterations,
        )
        query = (
            f"Task id: {task.task_id}\n"
            f"Working directory is already set to the task workspace.\n"
            f"Instruction:\n{task.instruction}\n"
        )
        try:
            result = await agent.ainvoke({"query": query})
        except Exception as exc:  # noqa: BLE001
            err = str(exc).lower()
            status = Status.BUDGET if "budget" in err or "429" in err else Status.ERROR
            return Result(
                status=status,
                cost=0,
                model=model_name,
                error=f"{type(exc).__name__}: {exc}",
                meta={"solver": "coding_react"},
            )

        usage = extract_usage(result)
        cost = cost_from_usage(usage)

        # Prefer answer written by the agent into work_dir or task_dir.
        answer = None
        for candidate in (task.work_dir / "answer.txt", task.task_dir / "answer.txt"):
            if candidate.is_file():
                answer = candidate.read_text(encoding="utf-8").strip()
                break
        if answer is None and isinstance(result, dict):
            content = result.get("content") or result.get("output")
            if isinstance(content, str) and "DONE" in content.upper():
                answer = "DONE"

        # Cheap verification: tests must pass.
        from harness.solvers.coding import _run_tests

        ok, out = _run_tests(task.work_dir)
        if ok:
            return Result(
                status=Status.SOLVED,
                answer="DONE",
                cost=cost,
                model=model_name,
                usage=usage,
                meta={"solver": "coding_react", "test_ok": True},
            )
        return Result(
            status=Status.FAILED,
            answer=answer,
            cost=cost,
            model=model_name,
            usage=usage,
            error="tests_still_failing",
            meta={"solver": "coding_react", "detail": out[:800]},
        )

    return asyncio.run(_run())


def smoke_build_config() -> dict[str, Any]:
    """Unit-test helper: build agent config objects without calling the LLM."""
    api_key = _env("LLM_API_KEY", "test-key") or "test-key"
    api_base = _env("LLM_BASE_URL", "http://127.0.0.1:9") or "http://127.0.0.1:9"
    from openjiuwen.core.single_agent import ReActAgentConfig

    cfg = (
        ReActAgentConfig()
        .configure_model_client(
            provider="OpenAI",
            api_key=api_key,
            api_base=api_base,
            model_name=ModelName.QWEN36_35B_NOREASON.value,
            verify_ssl=False,
        )
        .configure_prompt_template([{"role": "system", "content": CODING_SYSTEM_PROMPT}])
        .configure_max_iterations(8)
    )
    return {
        "model": ModelName.QWEN36_35B_NOREASON.value,
        "has_model_client": getattr(cfg, "model_client_config", None) is not None,
        "max_iterations": 8,
    }
