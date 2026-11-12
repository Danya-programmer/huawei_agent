"""Two-phase budget-aware scheduler: cheap first, coding/react later."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from harness.io import CostJournal, atomic_write_answer, discover_tasks
from harness.solver import Solver, SolverRegistry
from harness.triage import TaskProfile, triage
from harness.types import Result, Status, Task

# Cheap/deterministic pool size; coding/react hard-capped separately.
DEFAULT_MAX_WORKERS = 5
CODING_REACT_CONCURRENCY = 2


@dataclass
class PlannedWork:
    """Triage-partitioned task lists for the two-phase run."""

    profiles: dict[str, TaskProfile]
    skips: list[Task]
    phase_a: list[Task]
    phase_b: list[Task]

    @property
    def runnable(self) -> list[Task]:
        return self.phase_a + self.phase_b


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


def plan(tasks: list[Task]) -> PlannedWork:
    """Triage all tasks into skip / phase A (cheap) / phase B (coding+fallback)."""
    profiles: dict[str, TaskProfile] = {}
    skips: list[Task] = []
    phase_a: list[Task] = []
    phase_b: list[Task] = []

    for task in tasks:
        profile = triage(task)
        profiles[task.task_id] = profile
        if profile.lane == "skip" or not profile.feasible:
            skips.append(task)
        elif profile.lane == "react":
            # Short ReAct (math / data_analysis / search) first.
            phase_a.append(task)
        else:
            # coding (longer ReAct)
            phase_b.append(task)

    phase_a.sort(
        key=lambda t: (profiles[t.task_id].expected_cost, -t.reward),
    )
    phase_b.sort(
        key=lambda t: -(t.reward / max(profiles[t.task_id].expected_cost, 1)),
    )
    return PlannedWork(
        profiles=profiles,
        skips=skips,
        phase_a=phase_a,
        phase_b=phase_b,
    )


class SimpleScheduler:
    """Phase A cheap (≤5 workers), then phase B coding/react (concurrency ≤2)."""

    def __init__(
        self,
        solvers: Optional[SolverRegistry | Solver] = None,
        journal: Optional[CostJournal] = None,
        budget: Optional[int] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        coding_concurrency: int = CODING_REACT_CONCURRENCY,
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
        self.max_workers = max(1, min(int(max_workers), DEFAULT_MAX_WORKERS))
        self.coding_concurrency = max(1, int(coding_concurrency))

    def plan(self, tasks: list[Task]) -> PlannedWork:
        return plan(tasks)

    def order_tasks(self, tasks: list[Task]) -> list[Task]:
        """Runnable order: phase A then phase B (skips omitted)."""
        return self.plan(tasks).runnable

    def run(self, tasks_dir: str | os.PathLike[str]) -> RunSummary:
        discovered = discover_tasks(tasks_dir)
        planned = self.plan(discovered)
        summary = RunSummary(tasks_seen=len(discovered))
        ledger = BudgetLedger(self.budget)
        lock = threading.Lock()
        coding_sem = threading.Semaphore(self.coding_concurrency)

        def finalize(task: Task, result: Result) -> None:
            with lock:
                self._finalize(task, result, summary)

        # Unsolvable: skip with cost 0 (no solver / no LLM).
        for task in planned.skips:
            profile = planned.profiles[task.task_id]
            finalize(
                task,
                Result(
                    status=Status.UNSOLVABLE,
                    cost=0,
                    error=profile.fail_reason or "unsolvable",
                    meta={
                        "reason": "triage_skip",
                        "lane": profile.lane,
                        "fail_reason": profile.fail_reason,
                    },
                ),
            )

        def handle(task: Task, *, use_coding_cap: bool) -> tuple[Task, Result]:
            profile = planned.profiles[task.task_id]
            est = profile.expected_cost
            if ledger.stop:
                return task, Result(
                    status=Status.BUDGET,
                    cost=0,
                    error=ledger.stop_reason or "stopped",
                    meta={"reason": "stopped", "estimate": est},
                )
            if not ledger.can_afford(est):
                return task, Result(
                    status=Status.BUDGET,
                    cost=0,
                    error="estimate exceeds remaining budget",
                    meta={"reason": "budget", "estimate": est},
                )

            acquired = False
            try:
                if use_coding_cap:
                    coding_sem.acquire()
                    acquired = True
                    if ledger.stop or not ledger.can_afford(est):
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
            finally:
                if acquired:
                    coding_sem.release()

        def run_phase(tasks: list[Task], *, workers: int, use_coding_cap: bool) -> None:
            if not tasks or ledger.stop:
                return
            n = max(1, min(workers, len(tasks)))
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [
                    pool.submit(handle, t, use_coding_cap=use_coding_cap) for t in tasks
                ]
                for fut in as_completed(futures):
                    task, result = fut.result()
                    finalize(task, result)

        run_phase(planned.phase_a, workers=self.max_workers, use_coding_cap=False)
        run_phase(
            planned.phase_b,
            workers=min(self.max_workers, self.coding_concurrency),
            use_coding_cap=True,
        )

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
    max_workers: int = DEFAULT_MAX_WORKERS,
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
