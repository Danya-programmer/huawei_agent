"""openJiuwen ReAct coding agent (optional; requires LLM_* env)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Awaitable, Optional, TypeVar

from harness.answer import sanitize_answer
from harness.cost import cost_from_usage
from harness.http_retry import call_with_retries, error_is_rate_limit
from harness.types import Result, Status, Task, TokenUsage, resolve_model
from harness.usage import capture_llm_usage, extract_usage, sum_usage


async def _agent_invoke(agent: Any, payload: dict[str, Any]) -> tuple[Any, TokenUsage]:
    """Call openJiuwen ReActAgent.invoke; return (result, TokenUsage).

    openjiuwen 0.1.x invoke() returns only ``{output, result_type}`` — token
    usage must be captured from ``LLM_INVOKE_OUTPUT`` callbacks.
    """
    async with capture_llm_usage() as bucket:
        if hasattr(agent, "invoke"):
            result = await agent.invoke(payload)
        elif hasattr(agent, "ainvoke"):
            result = await agent.ainvoke(payload)
        else:
            raise AttributeError(
                f"{type(agent).__name__} has neither invoke nor ainvoke"
            )
    hooked = sum_usage(bucket)
    fallback = extract_usage(result)
    usage = hooked if (hooked.prompt_tokens or hooked.completion_tokens) else fallback
    return result, usage if isinstance(usage, TokenUsage) else TokenUsage()


CODING_SYSTEM_PROMPT = """You are a coding agent working ONLY inside the current directory.
Goal: make `python3 test_mylib.py` pass by editing mylib.py only. Do not edit the tests.
When all tests pass, write exactly the three characters DONE into answer.txt
using a shell redirect, for example: echo DONE > answer.txt
Never put code, commands, markdown, or explanations into answer.txt — only DONE.
Prefer short shell/python tool calls. Do not invent files outside this directory.
"""

GENERAL_SYSTEM_PROMPT = """You are a careful task-solving agent working ONLY inside the current directory.
Use bash/python/file tools to inspect files and compute the result.
When finished, write ONLY the final answer value into answer.txt via a shell redirect, e.g.:
  echo 42 > answer.txt
  echo Osaka > answer.txt
Rules for answer.txt:
- one short final value only (number, name, date, or short phrase)
- NO markdown fences, NO shell commands, NO Python code, NO reasoning, NO XML/tool tags
- if the task is impossible with the given files, do NOT create answer.txt
"""


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


_T = TypeVar("_T")
_thread_loops: dict[int, asyncio.AbstractEventLoop] = {}
_thread_loops_lock = threading.Lock()


def _run_on_thread_loop(coro: Awaitable[_T]) -> _T:
    """Run ``coro`` on a per-thread event loop that is NOT closed between tasks.

    ``asyncio.run()`` creates a fresh loop and closes it on exit. openjiuwen's
    OpenAI/httpx async client stays bound to that closed loop, so the *next*
    task on the same worker thread fails with ``Event loop is closed``.
    """
    tid = threading.get_ident()
    with _thread_loops_lock:
        loop = _thread_loops.get(tid)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            _thread_loops[tid] = loop
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def _reset_shared_llm_clients() -> None:
    """Drop openjiuwen's process-wide AsyncOpenAI/httpx pools between tasks.

    Persistent loops alone are not enough: after ``run_until_complete`` returns,
    the cached keep-alive client can hang (~360s APITimeoutError) then fail with
    APIConnectionError on the next task.
    """
    try:
        from openjiuwen.core.foundation.llm.model_clients.openai_model_client import (
            OpenAIModelClient,
        )

        await OpenAIModelClient.aclose()
    except Exception:
        pass
    try:
        from openjiuwen.core.foundation.llm.model_clients.anthropic_model_client import (
            AnthropicModelClient,
        )

        await AnthropicModelClient.aclose()
    except Exception:
        pass


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
                _run_on_thread_loop(_ensure_runner_started())
            except RuntimeError:
                pass
    except Exception:
        pass

    # openjiuwen 0.1.x: LocalWorkConfig has no work_dir; CWD is ContextVar via init_cwd.
    sysop = SysOperationCard(
        id=f"sysop_{uuid.uuid4().hex[:8]}",
        mode=OperationMode.LOCAL,
        work_config=LocalWorkConfig(
            sandbox_root=[work_dir],
            restrict_to_sandbox=True,
        ),
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


def _init_agent_cwd(work_dir: str) -> None:
    """Bind openjiuwen tool CWD to the task workspace (required in 0.1.x)."""
    from openjiuwen.core.sys_operation.cwd import init_cwd

    init_cwd(work_dir, project_root=work_dir, workspace=work_dir)


def _read_answer_raw(task: Task) -> Optional[str]:
    for candidate in (task.work_dir / "answer.txt", task.task_dir / "answer.txt"):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def _clear_answer_files(task: Task) -> None:
    for candidate in (task.work_dir / "answer.txt", task.task_dir / "answer.txt"):
        try:
            if candidate.is_file():
                candidate.unlink()
        except OSError:
            pass


def _persist_answer(task: Task, answer: str) -> None:
    """Write cleaned answer next to task.json (contest contract)."""
    try:
        target = task.task_dir / "answer.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(answer.rstrip() + "\n", encoding="utf-8")
    except OSError:
        pass


def _status_for_llm_error(exc: BaseException) -> Status:
    err = str(exc).lower()
    if error_is_rate_limit(exc) or "budget" in err:
        return Status.BUDGET
    return Status.ERROR


def _openai_chat_answer(task: Task, *, model_name: str) -> Result:
    """Minimal OpenAI-compatible single-shot fallback (no openjiuwen / no tools)."""
    import json
    import urllib.error
    import urllib.request

    from harness.cost import compute_call_cost

    base = (_env("LLM_BASE_URL") or "").rstrip("/")
    key = _env("LLM_API_KEY") or ""
    body = json.dumps(
        {
            "model": model_name,
            "temperature": 0,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Reply with ONLY the final answer value. "
                        "No markdown, no code, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task type: {task.task_type.value}\n"
                        f"Instruction:\n{task.instruction}\n\n"
                        "Final answer only:"
                    ),
                },
            ],
        }
    ).encode("utf-8")

    def _once() -> dict:
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
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Surface 429 body into the exception message for retry detection.
            raise RuntimeError(f"HTTPError: HTTP Error {exc.code}: {exc.reason}") from exc

    try:
        payload = call_with_retries(_once, retries=4, base_delay_s=2.0)
    except Exception as exc:  # noqa: BLE001
        return Result(
            status=_status_for_llm_error(exc),
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
    answer = sanitize_answer(content, coding=False)
    if not answer:
        _clear_answer_files(task)
        return Result(
            status=Status.FAILED,
            cost=cost,
            model=model_name,
            error="empty_or_garbage_model_answer",
            meta={"solver": "openai_chat_fallback", "raw": (content or "")[:200]},
        )
    _persist_answer(task, answer)
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

    model_name = resolve_model(model)
    try:
        import openjiuwen  # noqa: F401
    except ImportError:
        return _openai_chat_answer(task, model_name=model_name)

    work_dir = str(task.work_dir)

    async def _run() -> Result:
        await _ensure_runner_started()
        _init_agent_cwd(work_dir)
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
        last_exc: BaseException | None = None
        result = None
        usage = TokenUsage()
        try:
            for attempt in range(5):
                try:
                    result, usage = await _agent_invoke(agent, {"query": query})
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    # Only rate-limits are worth retrying. APITimeoutError /
                    # APIConnectionError after a poisoned shared client burn
                    # ~360s each and feel like a hang.
                    if attempt >= 4 or not error_is_rate_limit(exc):
                        break
                    await asyncio.sleep(2.0 * (2**attempt))
            if last_exc is not None:
                return Result(
                    status=_status_for_llm_error(last_exc),
                    cost=0,
                    model=model_name,
                    error=f"{type(last_exc).__name__}: {last_exc}",
                    meta={"solver": "task_react"},
                )

            cost = cost_from_usage(usage)
            file_raw = _read_answer_raw(task)
            raw = file_raw
            source = "answer_txt" if file_raw else "none"
            if raw is None and isinstance(result, dict):
                content = result.get("content") or result.get("output")
                if isinstance(content, str) and content.strip():
                    raw = content.strip()
                    source = "llm_output"

            answer = sanitize_answer(raw, coding=False)
            # Prefer file-backed answers; reject long LLM prose fallback.
            if source == "llm_output" and answer and (
                " " in answer.strip() and len(answer.split()) > 4
            ):
                answer = None
            if answer:
                _persist_answer(task, answer)
                return Result(
                    status=Status.SOLVED,
                    answer=answer,
                    cost=cost,
                    model=model_name,
                    usage=usage,
                    meta={"solver": "task_react", "answer_source": source},
                )
            _clear_answer_files(task)
            return Result(
                status=Status.FAILED,
                cost=cost,
                model=model_name,
                usage=usage,
                error="no_answer_txt" if not raw else "garbage_answer",
                meta={
                    "solver": "task_react",
                    "raw": (raw or "")[:200],
                    "answer_source": source,
                },
            )
        finally:
            await _reset_shared_llm_clients()

    return _run_on_thread_loop(_run())


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

    try:
        import openjiuwen  # noqa: F401
    except ImportError:
        return Result(
            status=Status.FAILED,
            cost=0,
            error="openjiuwen not installed (run inside Docker image)",
            meta={"solver": "coding_react"},
        )

    model_name = resolve_model(model)
    work_dir = str(task.work_dir)

    async def _run() -> Result:
        await _ensure_runner_started()
        _init_agent_cwd(work_dir)
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
        last_exc: BaseException | None = None
        result = None
        usage = TokenUsage()
        try:
            for attempt in range(5):
                try:
                    result, usage = await _agent_invoke(agent, {"query": query})
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt >= 4 or not error_is_rate_limit(exc):
                        break
                    await asyncio.sleep(2.0 * (2**attempt))
            if last_exc is not None:
                return Result(
                    status=_status_for_llm_error(last_exc),
                    cost=0,
                    model=model_name,
                    error=f"{type(last_exc).__name__}: {last_exc}",
                    meta={"solver": "coding_react"},
                )

            cost = cost_from_usage(usage)

            ok, out = _run_tests(task.work_dir)
            if ok:
                _persist_answer(task, "DONE")
                return Result(
                    status=Status.SOLVED,
                    answer="DONE",
                    cost=cost,
                    model=model_name,
                    usage=usage,
                    meta={"solver": "coding_react", "test_ok": True},
                )
            _clear_answer_files(task)
            return Result(
                status=Status.FAILED,
                cost=cost,
                model=model_name,
                usage=usage,
                error="tests_still_failing",
                meta={"solver": "coding_react", "detail": out[:800]},
            )
        finally:
            await _reset_shared_llm_clients()

    return _run_on_thread_loop(_run())


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
            model_name=resolve_model(),
            verify_ssl=False,
        )
        .configure_prompt_template([{"role": "system", "content": CODING_SYSTEM_PROMPT}])
        .configure_max_iterations(8)
    )
    return {
        "model": resolve_model(),
        "has_model_client": getattr(cfg, "model_client_config", None) is not None,
        "max_iterations": 8,
    }
