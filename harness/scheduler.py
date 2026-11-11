"""Budget-aware scheduler with reward/cost ordering and up to 5 workers."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from harness.estimate import expected_cost, value_score
from harness.io import CostJournal, atomic_write_answer, discover_tasks
from harness.solver import Solver, SolverRegistry
from harness.types import Result, Status, Task


@dataclass
class RunSummary:
    tasks_seen: int = 0
    solved: int = 0
    skipped: int = 0
    failed: int = 0
    total_reward: int = 0
    total_cost: int = 0
    results: list[tuple[Task, Result]] = field(default_factory=list)


class BudgetLedger:
    """Thread-safe spent/remaining token ledger."""

    def __init__(self, budget: Optional[int]) -> None:
        self.budget = budget
        self._spent = 0
        self._lock = threading.Lock()
        self.stop = False
        self.stop_reason: Optional[str] = None

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    def remaining(self) -> Optional[int]:
        if self.budget is None:
            return None
        with self._lock:
            return self.budget - self._spent

    def can_afford(self, estimate: int) -> bool:
        if self.stop:
            return False
        if self.budget is None:
            return True
        with self._lock:
            return (self._spent + max(estimate, 0)) <= self.budget

    def charge(self, cost: int) -> None:
        with self._lock:
            self._spent += int(cost or 0)
            if self.budget is not None and self._spent >= self.budget:
                self.stop = True
                self.stop_reason = "budget_exhausted"

    def request_stop(self, reason: str) -> None:
        with self._lock:
            self.stop = True
            self.stop_reason = reason


class SimpleScheduler:
    """Order by reward/expected_cost; run up to ``max_workers`` tasks in parallel."""

    def __init__(
        self,
        solvers: Optional[SolverRegistry | Solver] = None,
        journal: Optional[CostJournal] = None,
        budget: Optional[int] = None,
        max_workers: int = 5,
    ) -> None:
        if solvers is None:
            self.solvers: SolverRegistry = SolverRegistry.with_stubs()
        elif isinstance(solvers, SolverRegistry):
            self.solvers = solvers
        else:
            registry = SolverRegistry()
            registry._default = solvers  # type: ignore[attr-defined]
            self.solvers = registry

        self.journal = journal if journal is not None else CostJournal.from_env()
        self.budget = budget if budget is not None else _budget_from_env()
        self.max_workers = max(1, min(int(max_workers), 5))

    def order_tasks(self, tasks: list[Task]) -> list[Task]:
        return sorted(tasks, key=value_score, reverse=True)

    def run(self, tasks_dir: str | os.PathLike[str]) -> RunSummary:
        tasks = self.order_tasks(discover_tasks(tasks_dir))
        summary = RunSummary(tasks_seen=len(tasks))
        ledger = BudgetLedger(self.budget)
        lock = threading.Lock()

        def handle(task: Task) -> tuple[Task, Result]:
            if ledger.stop:
                return task, Result(
                    status=Status.BUDGET,
                    cost=0,
                    error=ledger.stop_reason or "stopped",
                    meta={"reason": "stopped"},
                )
            est = expected_cost(task)
            if not ledger.can_afford(est):
                return task, Result(
                    status=Status.BUDGET,
                    cost=0,
                    error="estimate exceeds remaining budget",
                    meta={"reason": "budget", "estimate": est},
                )
            try:
                result = self.solvers.solve(task)
            except Exception as exc:  # noqa: BLE001
                result = Result(
                    status=Status.ERROR,
                    cost=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            ledger.charge(result.cost)
            if result.status == Status.ERROR and _looks_like_proxy_failure(result):
                ledger.request_stop("proxy_failure")
            return task, result

        workers = self.max_workers
        # Cheap deterministic tasks dominate; coding may use LLM — still cap at 5.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(handle, t) for t in tasks]
            for fut in as_completed(futures):
                task, result = fut.result()
                with lock:
                    self._finalize(task, result, summary)

        summary.total_cost = self.journal.total_cost()
        return summary

    def _finalize(self, task: Task, result: Result, summary: RunSummary) -> None:
        answer_written = False
        if result.status == Status.SOLVED and result.answer is not None:
            atomic_write_answer(task.task_dir, result.answer)
            answer_written = True

        self.journal.record_result(task, result, answer_written=answer_written)
        summary.results.append((task, result))

        if result.status == Status.SOLVED and answer_written:
            summary.solved += 1
            summary.total_reward += task.reward
        elif result.status in (Status.SKIPPED, Status.UNSOLVABLE, Status.BUDGET):
            summary.skipped += 1
        else:
            summary.failed += 1


def run_tasks(
    tasks_dir: str | os.PathLike[str],
    *,
    solvers: Optional[SolverRegistry] = None,
    journal: Optional[CostJournal] = None,
    budget: Optional[int] = None,
    max_workers: int = 5,
) -> RunSummary:
    scheduler = SimpleScheduler(
        solvers=solvers,
        journal=journal,
        budget=budget,
        max_workers=max_workers,
    )
    try:
        return scheduler.run(tasks_dir)
    finally:
        scheduler.journal.close()


def _budget_from_env() -> Optional[int]:
    raw = os.environ.get("BUDGET")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _looks_like_proxy_failure(result: Result) -> bool:
    err = (result.error or "").lower()
    return any(
        needle in err
        for needle in ("budget", "proxy", "429", "401", "403", "quota", "exhausted")
    )
