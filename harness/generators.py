"""Training-set generators for local solver validation (no LLM)."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from harness.solvers.math_solver import MathOp, eval_math_ops


FIRST = [
    "Alice",
    "Boris",
    "Chen",
    "Dmitri",
    "Farid",
    "Grace",
    "Hiro",
    "Irina",
    "Jonas",
    "Kira",
    "Maya",
    "Nikolai",
    "Olga",
    "Pavel",
    "Quinn",
    "Rosa",
    "Sergei",
    "Tanya",
    "Uma",
    "Viktor",
    "Wendy",
    "Xavier",
    "Yara",
    "Zane",
]
LAST = [
    "Garcia",
    "Haddad",
    "Ivanova",
    "Kowalski",
    "Muller",
    "Novak",
    "Petrov",
    "Sato",
    "Smith",
    "Wang",
    "Brown",
    "Nguyen",
    "Silva",
    "Kim",
    "Andersen",
]
COMPANIES = [
    ("Granite AI", "Berlin", 2001),
    ("Arktika Labs", "Oslo", 1995),
    ("Fjord Data", "Osaka", 1964),
    ("Bluefin Systems", "Seattle", 2010),
    ("Juniper Foods", "Toronto", 1988),
    ("Helix Motors", "Munich", 1972),
    ("Ember Bio", "Boston", 2015),
    ("Dune Robotics", "Hanoi", 1989),
    ("Cobalt Works", "Lisbon", 2004),
    ("Iris Optics", "Zurich", 1999),
    ("Maple Forge", "Montreal", 1978),
    ("Nimbus Soft", "Austin", 2012),
    ("Redwood Chip", "Portland", 2006),
    ("Solara Grid", "Madrid", 1991),
    ("Volt Harbor", "Singapore", 2008),
]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]
REGIONS = ["north", "south", "east", "west"]
TASK_TYPES = ("math", "data_analysis", "search", "coding")
DIFFICULTIES = ("easy", "medium", "hard")

# Known coding helpers the heuristic fixer can repair.
_CODING_BUGS: dict[str, tuple[str, str]] = {
    "total": (
        "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
        "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n",
    ),
    "maximum": (
        "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
        "    def test_maximum(self):\n        self.assertEqual(m.maximum([3, 9, 2]), 9)\n",
    ),
    "reverse_words": (
        "def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
        "    def test_reverse_words(self):\n        self.assertEqual(m.reverse_words('c a b'), 'b a c')\n",
    ),
    "count_even": (
        "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
        "    def test_count_even(self):\n        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)\n",
    ),
    "average": (
        "def average(xs):\n    return sum(xs) - len(xs)\n",
        "    def test_average(self):\n        self.assertEqual(m.average([2, 4, 6]), 4.0)\n",
    ),
    "first_n": (
        "def first_n(xs, n):\n    return xs[n:]\n",
        "    def test_first_n(self):\n        self.assertEqual(m.first_n([1, 2, 3, 4], 2), [1, 2])\n",
    ),
}

MATH_FAIL_MODES = ("no_starts_with", "contradictory_ops", "non_integer_trap")
CSV_FAIL_MODES = ("missing_sales_csv", "broken_header", "missing_units_column")
SEARCH_FAIL_MODES = ("person_missing", "employer_missing", "company_missing")
CODING_FAIL_MODES = ("missing_mylib", "foreign_import", "empty_test")


@dataclass(frozen=True)
class SlotSpec:
    task_type: str
    difficulty: str  # easy | medium | hard | unsolvable
    solvable: bool
    fail_mode: Optional[str]


@dataclass
class GenResult:
    task_id: str
    task_type: str
    difficulty: str
    solvable: bool
    expected: str
    reward: int
    fail_mode: Optional[str]
    path: Path


def write_task(
    root: Path,
    task_id: str,
    task_type: str,
    instruction: str,
    reward: int,
    expected: str,
) -> Path:
    """Write task.json. Schema: task_id, task_type, instruction, reward (+ expected for local eval)."""
    task_dir = root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "instruction": instruction,
        "reward": reward,
        "expected": expected,
    }
    (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return task_dir


def _reward(rng: random.Random) -> int:
    """Reward 1..5, intentionally not correlated with difficulty."""
    return rng.randint(1, 5)


def _answer_suffix() -> str:
    return (
        "Work only inside the current directory. When you are done, write ONLY the final answer "
        "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
    )


def _coding_done_suffix() -> str:
    return (
        "Work only inside the current directory. When you are done, write ONLY the word DONE "
        "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
    )


# ---------------------------------------------------------------------------
# Quota planner
# ---------------------------------------------------------------------------


def plan_slots(n: int, unsolvable_ratio: float = 0.2) -> list[SlotSpec]:
    """Build exact type / solvable / difficulty quotas.

    For n=100 and unsolvable_ratio=0.2:
      - 25 per type
      - 20 unsolvable (5 per type)
      - among 80 solvable: easy 40 / medium 25 / hard 15
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4 so types are equal")
    if not 0.0 <= unsolvable_ratio < 1.0:
        raise ValueError("unsolvable_ratio must be in [0, 1)")

    per_type = n // 4
    n_unsolvable = int(round(n * unsolvable_ratio))
    # Prefer equal unsolvable per type; remainder goes to earlier types.
    if n_unsolvable % 4 != 0:
        # Snap to nearest multiple of 4 that stays within [0, n].
        n_unsolvable = int(round(n_unsolvable / 4) * 4)
    n_unsolvable = min(n_unsolvable, n)
    unsolvable_per_type = n_unsolvable // 4
    solvable_per_type = per_type - unsolvable_per_type
    n_solvable = solvable_per_type * 4

    # Difficulty mix among solvable: scale the 40/25/15 ratio.
    if n_solvable == 80:
        easy_n, medium_n, hard_n = 40, 25, 15
    else:
        easy_n = int(round(n_solvable * 40 / 80))
        hard_n = int(round(n_solvable * 15 / 80))
        medium_n = n_solvable - easy_n - hard_n
        if medium_n < 0:
            medium_n = 0
            # Rebalance if rounding overshot.
            while easy_n + hard_n > n_solvable:
                if easy_n >= hard_n and easy_n > 0:
                    easy_n -= 1
                elif hard_n > 0:
                    hard_n -= 1
                else:
                    break
            medium_n = n_solvable - easy_n - hard_n

    easy_per = _spread(easy_n, 4)
    medium_per = _spread(medium_n, 4)
    hard_per = _spread(hard_n, 4)
    # Fix per-type totals to solvable_per_type (spread can leave ±1 drift).
    for i in range(4):
        while easy_per[i] + medium_per[i] + hard_per[i] > solvable_per_type:
            if medium_per[i] > 0:
                medium_per[i] -= 1
            elif easy_per[i] > 0:
                easy_per[i] -= 1
            elif hard_per[i] > 0:
                hard_per[i] -= 1
        while easy_per[i] + medium_per[i] + hard_per[i] < solvable_per_type:
            medium_per[i] += 1

    slots: list[SlotSpec] = []
    fail_banks = {
        "math": MATH_FAIL_MODES,
        "data_analysis": CSV_FAIL_MODES,
        "search": SEARCH_FAIL_MODES,
        "coding": CODING_FAIL_MODES,
    }
    for ti, task_type in enumerate(TASK_TYPES):
        for d, count in (
            ("easy", easy_per[ti]),
            ("medium", medium_per[ti]),
            ("hard", hard_per[ti]),
        ):
            for _ in range(count):
                slots.append(SlotSpec(task_type, d, True, None))
        modes = fail_banks[task_type]
        for j in range(unsolvable_per_type):
            slots.append(
                SlotSpec(task_type, "unsolvable", False, modes[j % len(modes)])
            )

    if n == 100 and abs(unsolvable_ratio - 0.2) < 1e-9:
        _assert_n100_quotas(slots)

    return slots


def _spread(total: int, buckets: int) -> list[int]:
    """Distribute ``total`` as evenly as possible across ``buckets`` (earlier get +1)."""
    base, rem = divmod(total, buckets)
    return [base + (1 if i < rem else 0) for i in range(buckets)]


def _assert_n100_quotas(slots: list[SlotSpec]) -> None:
    assert len(slots) == 100, f"expected 100 slots, got {len(slots)}"
    by_type = {t: 0 for t in TASK_TYPES}
    by_diff = {"easy": 0, "medium": 0, "hard": 0}
    unsolvable = 0
    unsolvable_by_type = {t: 0 for t in TASK_TYPES}
    for s in slots:
        by_type[s.task_type] += 1
        if not s.solvable:
            unsolvable += 1
            unsolvable_by_type[s.task_type] += 1
        else:
            by_diff[s.difficulty] += 1
    assert by_type == {t: 25 for t in TASK_TYPES}, by_type
    assert unsolvable == 20, unsolvable
    assert unsolvable_by_type == {t: 5 for t in TASK_TYPES}, unsolvable_by_type
    assert by_diff == {"easy": 40, "medium": 25, "hard": 15}, by_diff


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def gen_math(
    root: Path,
    idx: int,
    rng: random.Random,
    *,
    difficulty: str = "easy",
    solvable: bool = True,
    fail_mode: Optional[str] = None,
) -> GenResult:
    task_id = f"math_gen_{idx:04d}"
    reward = _reward(rng)

    if not solvable:
        return _gen_math_unsolvable(root, task_id, rng, reward, fail_mode or "no_starts_with")

    if difficulty == "hard":
        start = rng.randint(5000, 50000)
        parts = [MathOp("start", start)]
        text = [f"A warehouse starts with {start} boxes."]
        for _ in range(rng.randint(4, 6)):
            if rng.random() < 0.5:
                d = rng.choice([2, 3, 4, 5])
                parts.append(MathOp("split", d))
                text.append(
                    f"Then the stock is split equally into {d} warehouses and it keeps one share (rounded down)."
                )
            else:
                parts.append(MathOp("double"))
                text.append("Then the stock is doubled by merging with a sister warehouse.")
        ops = parts
        instruction = (
            " ".join(text)
            + " How many boxes does it have at the end? Compute it with the bash tool (python3 -c). "
            + _answer_suffix()
        )
    elif difficulty == "medium":
        start = rng.randint(800, 4000)
        ops = [
            MathOp("start", start),
            MathOp("ship", rng.randint(10, 40), rng.randint(2, 5)),
            MathOp("receive", rng.randint(1, 4), rng.randint(50, 150)),
        ]
        extra = rng.choice(["split", "double"])
        if extra == "split":
            d = rng.choice([2, 3, 4])
            ops.append(MathOp("split", d))
            extra_text = (
                f"Then the stock is split equally into {d} warehouses and it keeps one share (rounded down)."
            )
        else:
            ops.append(MathOp("double"))
            extra_text = "Then the stock is doubled by merging with a sister warehouse."
        instruction = (
            f"A warehouse starts with {start} boxes. "
            f"It ships {ops[1].a} boxes per day for {ops[1].b} days. "
            f"It receives {ops[2].a} deliveries of {ops[2].b} boxes each. "
            f"{extra_text} "
            "How many boxes does it have at the end? Compute it with the bash tool (python3 -c). "
            + _answer_suffix()
        )
    else:  # easy: 1–2 ops (ship/receive)
        start = rng.randint(500, 2500)
        if rng.random() < 0.5:
            ops = [
                MathOp("start", start),
                MathOp("ship", rng.randint(10, 40), rng.randint(2, 5)),
            ]
            instruction = (
                f"A warehouse starts with {start} boxes. "
                f"It ships {ops[1].a} boxes per day for {ops[1].b} days. "
                "How many boxes does it have at the end? Compute it with the bash tool (python3 -c). "
                + _answer_suffix()
            )
        else:
            ops = [
                MathOp("start", start),
                MathOp("ship", rng.randint(10, 40), rng.randint(2, 5)),
                MathOp("receive", rng.randint(1, 3), rng.randint(50, 120)),
            ]
            instruction = (
                f"A warehouse starts with {start} boxes. "
                f"It ships {ops[1].a} boxes per day for {ops[1].b} days. "
                f"It receives {ops[2].a} deliveries of {ops[2].b} boxes each. "
                "How many boxes does it have at the end? Compute it with the bash tool (python3 -c). "
                + _answer_suffix()
            )

    expected = str(eval_math_ops(ops))
    path = write_task(root, task_id, "math", instruction, reward, expected)
    return GenResult(task_id, "math", difficulty, True, expected, reward, None, path)


def _gen_math_unsolvable(
    root: Path,
    task_id: str,
    rng: random.Random,
    reward: int,
    fail_mode: str,
) -> GenResult:
    if fail_mode == "contradictory_ops":
        a, b = rng.randint(100, 500), rng.randint(600, 900)
        instruction = (
            f"A warehouse starts with {a} boxes. Separately it is also said to start with {b} boxes. "
            f"It ships {rng.randint(10, 30)} boxes per day for {rng.randint(2, 4)} days and also "
            f"ships {rng.randint(50, 80)} boxes per day for {rng.randint(5, 9)} days — these cannot both be true. "
            "How many boxes does it have at the end? Compute it with the bash tool (python3 -c). "
            + _answer_suffix()
        )
    elif fail_mode == "non_integer_trap":
        instruction = (
            "A warehouse starts with 100.5 boxes (half a damaged crate counted). "
            "Then the stock is split equally into 3 warehouses and it keeps one share, "
            "but keep the fractional remainder as a float rather than rounding down. "
            "How many boxes does it have at the end? Compute it with the bash tool (python3 -c). "
            + _answer_suffix()
        )
    else:  # no_starts_with
        instruction = (
            f"A warehouse holds roughly {rng.randint(500, 2000)} crates (exact count unknown). "
            f"After several undocumented transfers it ends with some integer amount. "
            "How many boxes does it have at the end? Compute it with the bash tool (python3 -c). "
            + _answer_suffix()
        )
    expected = "UNSOLVABLE"
    path = write_task(root, task_id, "math", instruction, reward, expected)
    return GenResult(task_id, "math", "unsolvable", False, expected, reward, fail_mode, path)


# ---------------------------------------------------------------------------
# Data analysis (CSV)
# ---------------------------------------------------------------------------


def gen_csv(
    root: Path,
    idx: int,
    rng: random.Random,
    *,
    difficulty: str = "easy",
    solvable: bool = True,
    fail_mode: Optional[str] = None,
) -> GenResult:
    task_id = f"data_analysis_gen_{idx:04d}"
    reward = _reward(rng)
    product = rng.choice(PRODUCTS)
    lo, hi = sorted(rng.sample(range(1, 13), 2))

    if difficulty == "hard":
        rows = rng.randint(300, 800)
    elif difficulty == "medium":
        rows = rng.randint(80, 250)
    else:
        rows = rng.randint(20, 50)

    instruction = (
        f"The file sales.csv has columns order_id,region,product,month,units,price ({rows} rows). "
        f"How many units of product '{product}' were sold in months {lo} through {hi} inclusive? "
        "Use bash with python3 to compute it. "
        + _answer_suffix()
    )

    if not solvable:
        return _gen_csv_unsolvable(
            root, task_id, instruction, reward, fail_mode or "broken_header", rows
        )

    task_dir = write_task(root, task_id, "data_analysis", instruction, reward, "0")
    ws = task_dir / "workspace"
    ws.mkdir(exist_ok=True)

    # Hard: make the queried product rare (~5–12% of rows).
    rare = difficulty == "hard"
    total = 0
    path = ws / "sales.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["order_id", "region", "product", "month", "units", "price"]
        )
        writer.writeheader()
        for i in range(1, rows + 1):
            if rare:
                p = product if rng.random() < 0.08 else rng.choice(
                    [x for x in PRODUCTS if x != product] or PRODUCTS
                )
            else:
                p = rng.choice(PRODUCTS)
            month = rng.randint(1, 12)
            units = rng.randint(1, 40)
            writer.writerow(
                {
                    "order_id": i,
                    "region": rng.choice(REGIONS),
                    "product": p,
                    "month": month,
                    "units": units,
                    "price": rng.randint(10, 100),
                }
            )
            if p == product and lo <= month <= hi:
                total += units

    expected = str(total)
    payload = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    payload["expected"] = expected
    (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return GenResult(
        task_id, "data_analysis", difficulty, True, expected, reward, None, task_dir
    )


def _gen_csv_unsolvable(
    root: Path,
    task_id: str,
    instruction: str,
    reward: int,
    fail_mode: str,
    rows: int,
) -> GenResult:
    expected = "UNSOLVABLE"
    task_dir = write_task(root, task_id, "data_analysis", instruction, reward, expected)
    ws = task_dir / "workspace"
    ws.mkdir(exist_ok=True)

    if fail_mode == "missing_sales_csv":
        # Intentionally no sales.csv
        (ws / "readme.txt").write_text("sales data was not shipped\n", encoding="utf-8")
    elif fail_mode == "missing_units_column":
        path = ws / "sales.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["order_id", "region", "product", "month", "price"]
            )
            writer.writeheader()
            for i in range(1, min(rows, 10) + 1):
                writer.writerow(
                    {
                        "order_id": i,
                        "region": "north",
                        "product": "widget",
                        "month": 1,
                        "price": 10,
                    }
                )
    else:  # broken_header
        (ws / "sales.csv").write_text("bad,header\n1,2\n", encoding="utf-8")

    return GenResult(
        task_id, "data_analysis", "unsolvable", False, expected, reward, fail_mode, task_dir
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def gen_search(
    root: Path,
    idx: int,
    rng: random.Random,
    *,
    difficulty: str = "easy",
    solvable: bool = True,
    fail_mode: Optional[str] = None,
) -> GenResult:
    task_id = f"search_gen_{idx:04d}"
    reward = _reward(rng)

    if difficulty == "hard":
        n_people = rng.randint(50, 70)
        n_companies = min(len(COMPANIES), rng.randint(12, len(COMPANIES)))
        mode = "employer_hq"
    elif difficulty == "medium":
        n_people = rng.randint(30, 45)
        n_companies = min(len(COMPANIES), rng.randint(8, 12))
        mode = rng.choice(["born", "employer_hq"])
    else:
        n_people = rng.randint(18, 25)
        n_companies = min(len(COMPANIES), 8)
        mode = "born"

    # Unsolvable employer/company gaps only bite on 2-hop HQ questions.
    if not solvable and fail_mode in {"employer_missing", "company_missing"}:
        mode = "employer_hq"
        n_people = max(n_people, 20)
        n_companies = max(n_companies, 6)

    people: list[str] = []
    seen: set[str] = set()
    while len(people) < n_people:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in seen:
            seen.add(name)
            people.append(name)

    companies = list(COMPANIES)
    rng.shuffle(companies)
    companies = companies[:n_companies]
    target = people[0]
    employer = companies[0][0]
    born = str(rng.randint(1960, 2000))

    if mode == "born":
        instruction = (
            f"The directory kb/ contains a small knowledge base of markdown files. "
            f"In which year was {target} born? Use the file tools or bash (ls, grep, cat) to find it. "
            + _answer_suffix()
        )
        expected_ok = born
    else:
        instruction = (
            f"The directory kb/ contains a small knowledge base of markdown files. "
            f"In which city is the headquarters of {target}'s employer? Use the file tools or bash "
            "(ls, grep, cat) to find it. "
            + _answer_suffix()
        )
        expected_ok = companies[0][1]

    task_dir = write_task(
        root,
        task_id,
        "search",
        instruction,
        reward,
        "UNSOLVABLE" if not solvable else expected_ok,
    )
    kb = task_dir / "workspace" / "kb"
    kb.mkdir(parents=True, exist_ok=True)

    skip_person = not solvable and fail_mode == "person_missing"
    omit_employer = not solvable and fail_mode == "employer_missing"
    omit_company = not solvable and fail_mode == "company_missing"

    for name in people:
        if skip_person and name == target:
            continue
        emp = employer if name == target else rng.choice(companies)[0]
        b = born if name == target else str(rng.randint(1960, 2000))
        if omit_employer and name == target:
            # Omit employer field entirely (UNKNOWN would still look present).
            emp_line = ""
        else:
            emp_line = f"Employer: {emp}\n"
        slug = name.lower().replace(" ", "_")
        (kb / f"person_{slug}.md").write_text(
            f"# {name}\n\nBorn: {b}\n{emp_line}Mentor: {rng.choice(people)}\n",
            encoding="utf-8",
        )

    for cname, city, year in companies:
        if omit_company and cname == employer:
            continue
        slug = cname.lower().replace(" ", "_")
        (kb / f"company_{slug}.md").write_text(
            f"# {cname}\n\nHeadquarters: {city}\nFounded: {year}\n",
            encoding="utf-8",
        )

    if not solvable:
        # Default fail_mode if somehow unset
        fm = fail_mode or "person_missing"
        return GenResult(
            task_id, "search", "unsolvable", False, "UNSOLVABLE", reward, fm, task_dir
        )

    return GenResult(
        task_id, "search", difficulty, True, expected_ok, reward, None, task_dir
    )


# ---------------------------------------------------------------------------
# Coding
# ---------------------------------------------------------------------------


def gen_coding(
    root: Path,
    idx: int,
    rng: random.Random,
    *,
    difficulty: str = "easy",
    solvable: bool = True,
    fail_mode: Optional[str] = None,
) -> GenResult:
    task_id = f"coding_gen_{idx:04d}"
    reward = _reward(rng)
    instruction = (
        "Run `python3 test_mylib.py` in the current directory; some tests fail. "
        "Fix the bugs in mylib.py (do not edit the tests) until all tests pass. "
        + _coding_done_suffix()
    )

    if not solvable:
        return _gen_coding_unsolvable(
            root, task_id, instruction, reward, fail_mode or "missing_mylib"
        )

    names = list(_CODING_BUGS.keys())
    rng.shuffle(names)
    if difficulty == "hard":
        chosen = names[: rng.randint(4, min(6, len(names)))]
    elif difficulty == "medium":
        chosen = names[: rng.randint(2, 3)]
    else:
        chosen = names[: rng.randint(1, 2)]

    lib_parts = [_CODING_BUGS[n][0] for n in chosen]
    test_methods = [_CODING_BUGS[n][1] for n in chosen]
    mylib_src = "\n".join(lib_parts)
    test_src = (
        "import unittest\nimport mylib as m\n\n"
        "class T(unittest.TestCase):\n"
        + "".join(test_methods)
        + "\nif __name__ == '__main__':\n    unittest.main()\n"
    )

    task_dir = write_task(root, task_id, "coding", instruction, reward, "DONE")
    ws = task_dir / "workspace"
    ws.mkdir(exist_ok=True)
    (ws / "mylib.py").write_text(mylib_src, encoding="utf-8")
    (ws / "test_mylib.py").write_text(test_src, encoding="utf-8")
    return GenResult(task_id, "coding", difficulty, True, "DONE", reward, None, task_dir)


def _gen_coding_unsolvable(
    root: Path,
    task_id: str,
    instruction: str,
    reward: int,
    fail_mode: str,
) -> GenResult:
    expected = "UNSOLVABLE"
    task_dir = write_task(root, task_id, "coding", instruction, reward, expected)
    ws = task_dir / "workspace"
    ws.mkdir(exist_ok=True)

    if fail_mode == "foreign_import":
        (ws / "mylib.py").write_text(
            "def total(xs):\n    return sum(xs)\n",
            encoding="utf-8",
        )
        (ws / "test_mylib.py").write_text(
            "import unittest\nimport otherlib as m\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_total(self):\n"
            "        self.assertEqual(m.total([1, 2, 3]), 6)\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n",
            encoding="utf-8",
        )
    elif fail_mode == "empty_test":
        (ws / "mylib.py").write_text(
            "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
            encoding="utf-8",
        )
        (ws / "test_mylib.py").write_text("", encoding="utf-8")
    else:  # missing_mylib
        (ws / "test_mylib.py").write_text(
            "import unittest\nimport mylib as m\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_total(self):\n"
            "        self.assertEqual(m.total([1, 2, 3]), 6)\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n",
            encoding="utf-8",
        )

    return GenResult(
        task_id, "coding", "unsolvable", False, expected, reward, fail_mode, task_dir
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _reset_out_dir(root: Path) -> None:
    """Idempotent wipe/recreate of the output directory contents."""
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        root.mkdir(parents=True, exist_ok=True)


def _print_sanity_counts(results: list[GenResult]) -> None:
    by_type: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    solvable = 0
    unsolvable = 0
    for r in results:
        by_type[r.task_type] = by_type.get(r.task_type, 0) + 1
        if r.solvable:
            solvable += 1
            by_diff[r.difficulty] = by_diff.get(r.difficulty, 0) + 1
        else:
            unsolvable += 1
    print(f"total={len(results)}")
    print("by_type: " + " ".join(f"{k}={by_type.get(k, 0)}" for k in TASK_TYPES))
    print(f"solvable={solvable} unsolvable={unsolvable}")
    print(
        "by_difficulty(solvable): "
        + " ".join(f"{k}={by_diff.get(k, 0)}" for k in DIFFICULTIES)
    )


def generate_training_set(
    out_dir: Path | str,
    n: int = 100,
    seed: int = 42,
    unsolvable_ratio: float = 0.2,
) -> Path:
    """Create a mixed training set with type/difficulty/unsolvable quotas + manifest."""
    root = Path(out_dir)
    _reset_out_dir(root)
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    slots = plan_slots(n, unsolvable_ratio)
    rng.shuffle(slots)

    counters = {t: 0 for t in TASK_TYPES}
    results: list[GenResult] = []

    for slot in slots:
        idx = counters[slot.task_type]
        counters[slot.task_type] = idx + 1
        if slot.task_type == "math":
            if slot.solvable:
                result = gen_math(
                    root, idx, rng, difficulty=slot.difficulty, solvable=True
                )
            else:
                result = gen_math(
                    root, idx, rng, solvable=False, fail_mode=slot.fail_mode
                )
        elif slot.task_type == "data_analysis":
            if slot.solvable:
                result = gen_csv(
                    root, idx, rng, difficulty=slot.difficulty, solvable=True
                )
            else:
                result = gen_csv(
                    root,
                    idx,
                    rng,
                    difficulty="easy",
                    solvable=False,
                    fail_mode=slot.fail_mode,
                )
        elif slot.task_type == "search":
            if slot.solvable:
                result = gen_search(
                    root, idx, rng, difficulty=slot.difficulty, solvable=True
                )
            else:
                result = gen_search(
                    root,
                    idx,
                    rng,
                    difficulty="easy",
                    solvable=False,
                    fail_mode=slot.fail_mode,
                )
        else:
            if slot.solvable:
                result = gen_coding(
                    root, idx, rng, difficulty=slot.difficulty, solvable=True
                )
            else:
                result = gen_coding(
                    root,
                    idx,
                    rng,
                    difficulty="easy",
                    solvable=False,
                    fail_mode=slot.fail_mode,
                )
        results.append(result)

    # Stable manifest order by task_id
    results.sort(key=lambda r: r.task_id)
    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for r in results:
            row = {
                "task_id": r.task_id,
                "task_type": r.task_type,
                "difficulty": r.difficulty,
                "solvable": r.solvable,
                "expected": r.expected,
                "reward": r.reward,
                "fail_mode": r.fail_mode,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if n == 100 and abs(unsolvable_ratio - 0.2) < 1e-9:
        _assert_n100_results(results)

    _print_sanity_counts(results)
    print(f"manifest={manifest_path}")
    return root


def _assert_n100_results(results: list[GenResult]) -> None:
    assert len(results) == 100
    by_type = {t: 0 for t in TASK_TYPES}
    by_diff = {"easy": 0, "medium": 0, "hard": 0}
    unsolvable = 0
    unsolvable_by_type = {t: 0 for t in TASK_TYPES}
    for r in results:
        by_type[r.task_type] += 1
        if not r.solvable:
            unsolvable += 1
            unsolvable_by_type[r.task_type] += 1
            assert r.fail_mode, f"{r.task_id} missing fail_mode"
            assert r.expected == "UNSOLVABLE"
            assert r.difficulty == "unsolvable"
        else:
            by_diff[r.difficulty] += 1
            assert r.fail_mode is None
    assert by_type == {t: 25 for t in TASK_TYPES}, by_type
    assert unsolvable == 20, unsolvable
    assert unsolvable_by_type == {t: 5 for t in TASK_TYPES}, unsolvable_by_type
    assert by_diff == {"easy": 40, "medium": 25, "hard": 15}, by_diff


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate train-set tasks + manifest.jsonl")
    p.add_argument("--out", default="train-100")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unsolvable-ratio", type=float, default=0.2)
    args = p.parse_args()
    path = generate_training_set(
        args.out,
        n=args.n,
        seed=args.seed,
        unsolvable_ratio=args.unsolvable_ratio,
    )
    print(f"wrote {args.n} tasks under {path}")
