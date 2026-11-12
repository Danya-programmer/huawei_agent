"""openJiuwen ReAct coding agent (optional; requires LLM_* env)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

from harness.cost import cost_from_usage
from harness.types import DEFAULT_MODEL, ModelName, Result, Status, Task, TokenUsage
from harness.usage import extract_usage


CODING_SYSTEM_PROMPT = """You are a coding agent working ONLY inside the current directory.
Goal: make `python3 test_mylib.py` pass by editing mylib.py only. Do not edit the tests.
When all tests pass, write exactly DONE to answer.txt (e.g. `echo DONE > answer.txt`) and stop.
Prefer short shell/python commands. Do not invent files outside this directory.
"""

GENERAL_SYSTEM_PROMPT = """You are a careful task-solving agent working ONLY inside the current directory.
Read the instruction, use bash/python/file tools as needed, and finish by writing ONLY the final
answer to answer.txt in the current directory (for example: echo 42 > answer.txt).
Do not invent files outside this directory. Prefer short tool calls. If the task is clearly
impossible with the given files, write NOTHING and stop.
"""


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _run_tests(work_dir: Path, timeout: float = 20.0) -> tuple[bool, str]:
    """Run ``python3 test_mylib.py`` in the coding workspace."""
    proc = subprocess.run(
        ["python3", "test_mylib.py"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


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


def _read_answer(task: Task) -> Optional[str]:
    for candidate in (task.work_dir / "answer.txt", task.task_dir / "answer.txt"):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def _openai_chat_answer(task: Task, *, model_name: str) -> Result:
    """Minimal OpenAI-compatible single-shot fallback (no openjiuwen / no tools)."""
    import json
    import urllib.request

    from harness.cost import compute_call_cost

    base = (_env("LLM_BASE_URL") or "").rstrip("/")
    key = _env("LLM_API_KEY") or ""
    body = json.dumps(
        {
            "model": model_name,
            "temperature": 0,
            "max_tokens": 128,
            "messages": [
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task type: {task.task_type.value}\n"
                        f"Instruction:\n{task.instruction}\n\n"
                        "Reply with ONLY the final answer text (no markdown)."
                    ),
                },
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        err = str(exc).lower()
        status = Status.BUDGET if "budget" in err or "429" in err else Status.ERROR
        return Result(
            status=status,
            cost=0,
            model=model_name,
            error=f"{type(exc).__name__}: {exc}",
            meta={"solver": "openai_chat_fallback"},
        )

    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    reasoning_tokens = int(
        details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0
    )
    cost = compute_call_cost(prompt_tokens, completion_tokens, reasoning_tokens)
    content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
    answer = (content or "").strip().splitlines()[-1].strip() if content else ""
    if not answer:
        return Result(
            status=Status.FAILED,
            cost=cost,
            model=model_name,
            error="empty_model_answer",
            meta={"solver": "openai_chat_fallback"},
        )
    # Persist answer.txt for consistency with the contest contract.
    try:
        target = task.work_dir / "answer.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(answer + "\n", encoding="utf-8")
    except OSError:
        pass
    return Result(
        status=Status.SOLVED,
        answer=answer,
        cost=cost,
        model=model_name,
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        meta={"solver": "openai_chat_fallback"},
    )


def run_task_react(
    task: Task,
    *,
    max_iterations: int = 4,
    model: Optional[str] = None,
    system_prompt: str = GENERAL_SYSTEM_PROMPT,
) -> Result:
    """Synchronously run a general ReAct agent; expects answer.txt.

    Falls back to a single OpenAI-compatible chat call if openjiuwen is missing.
    """
    if not _env("LLM_API_KEY") or not _env("LLM_BASE_URL"):
        return Result(
            status=Status.FAILED,
            cost=0,
            error="LLM_API_KEY/LLM_BASE_URL not set",
            meta={"solver": "task_react"},
        )

    model_name = model or DEFAULT_MODEL.value
    try:
        import openjiuwen  # noqa: F401
    except ImportError:
        return _openai_chat_answer(task, model_name=model_name)

    work_dir = str(task.work_dir)

    async def _run() -> Result:
        await _ensure_runner_started()
        agent = build_react_agent(
            work_dir,
            model_name=model_name,
            max_iterations=max_iterations,
            system_prompt=system_prompt,
        )
        query = (
            f"Task id: {task.task_id}\n"
            f"Task type: {task.task_type.value}\n"
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
                meta={"solver": "task_react"},
            )

        usage = extract_usage(result)
        cost = cost_from_usage(usage)
        answer = _read_answer(task)
        if answer is None and isinstance(result, dict):
            content = result.get("content") or result.get("output")
            if isinstance(content, str) and content.strip():
                # Last resort: model replied in chat without writing the file.
                answer = content.strip().splitlines()[-1].strip()

        if answer:
            return Result(
                status=Status.SOLVED,
                answer=answer,
                cost=cost,
                model=model_name,
                usage=usage,
                meta={"solver": "task_react"},
            )
        return Result(
            status=Status.FAILED,
            cost=cost,
            model=model_name,
            usage=usage,
            error="no_answer_txt",
            meta={"solver": "task_react"},
        )

    return asyncio.run(_run())


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
            system_prompt=CODING_SYSTEM_PROMPT,
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

        answer = _read_answer(task)
        if answer is None and isinstance(result, dict):
            content = result.get("content") or result.get("output")
            if isinstance(content, str) and "DONE" in content.upper():
                answer = "DONE"

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
