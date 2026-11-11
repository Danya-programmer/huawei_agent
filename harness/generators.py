"""Training-set generators for local solver validation (no LLM)."""

from __future__ import annotations

import csv
import json
import random
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
]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]
REGIONS = ["north", "south", "east", "west"]


def write_task(root: Path, task_id: str, task_type: str, instruction: str, reward: int, expected: str) -> Path:
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


def gen_math(root: Path, idx: int, rng: random.Random) -> Path:
    start = rng.randint(500, 2500)
    if rng.random() < 0.5:
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
            "Work only inside the current directory. When you are done, write ONLY the final answer "
            "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
        )
    else:
        parts = [MathOp("start", start)]
        text = [f"A warehouse starts with {start} boxes."]
        for _ in range(rng.randint(2, 4)):
            if rng.random() < 0.5:
                d = rng.choice([2, 3, 4])
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
            "Work only inside the current directory. When you are done, write ONLY the final answer "
            "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
        )
    expected = str(eval_math_ops(ops))
    return write_task(root, f"math_gen_{idx:04d}", "math", instruction, rng.randint(1, 5), expected)


def gen_csv(root: Path, idx: int, rng: random.Random, *, broken: bool = False) -> Path:
    product = rng.choice(PRODUCTS)
    lo, hi = sorted(rng.sample(range(1, 13), 2))
    rows = rng.randint(20, 80)
    task_dir = write_task(
        root,
        f"data_analysis_gen_{idx:04d}",
        "data_analysis",
        (
            f"The file sales.csv has columns order_id,region,product,month,units,price ({rows} rows). "
            f"How many units of product '{product}' were sold in months {lo} through {hi} inclusive? "
            "Use bash with python3 to compute it. Work only inside the current directory. "
            "When you are done, write ONLY the final answer to the file answer.txt in the current directory "
            "(for example with bash: echo 42 > answer.txt)."
        ),
        rng.randint(1, 4),
        "0",
    )
    ws = task_dir / "workspace"
    ws.mkdir(exist_ok=True)
    if broken:
        (ws / "sales.csv").write_text("bad,header\n1,2\n", encoding="utf-8")
        payload = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        payload["expected"] = "UNSOLVABLE"
        (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return task_dir

    total = 0
    path = ws / "sales.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["order_id", "region", "product", "month", "units", "price"]
        )
        writer.writeheader()
        for i in range(1, rows + 1):
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
    payload = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    payload["expected"] = str(total)
    (task_dir / "task.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return task_dir


def gen_search(root: Path, idx: int, rng: random.Random, *, missing: bool = False) -> Path:
    people = []
    for _ in range(25):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        people.append(name)
    companies = list(COMPANIES)
    rng.shuffle(companies)
    target = people[0]
    employer = companies[0][0]
    born = str(rng.randint(1960, 2000))
    mode = rng.choice(["born", "employer_hq"])
    if mode == "born":
        instruction = (
            f"The directory kb/ contains a small knowledge base of markdown files. "
            f"In which year was {target} born? Use the file tools or bash (ls, grep, cat) to find it. "
            "Work only inside the current directory. When you are done, write ONLY the final answer "
            "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
        )
        expected = "MISSING" if missing else born
    else:
        instruction = (
            f"The directory kb/ contains a small knowledge base of markdown files. "
            f"In which city is the headquarters of {target}'s employer? Use the file tools or bash "
            "(ls, grep, cat) to find it. Work only inside the current directory. When you are done, "
            "write ONLY the final answer to the file answer.txt in the current directory "
            "(for example with bash: echo 42 > answer.txt)."
        )
        expected = "MISSING" if missing else companies[0][1]

    task_dir = write_task(
        root,
        f"search_gen_{idx:04d}",
        "search",
        instruction,
        rng.randint(1, 3),
        expected,
    )
    kb = task_dir / "workspace" / "kb"
    kb.mkdir(parents=True, exist_ok=True)
    for name in people:
        emp = employer if name == target else rng.choice(companies)[0]
        b = born if name == target else str(rng.randint(1960, 2000))
        if missing and name == target:
            continue
        slug = name.lower().replace(" ", "_")
        (kb / f"person_{slug}.md").write_text(
            f"# {name}\n\nBorn: {b}\nEmployer: {emp}\nMentor: {rng.choice(people)}\n",
            encoding="utf-8",
        )
    for cname, city, year in companies:
        slug = cname.lower().replace(" ", "_")
        (kb / f"company_{slug}.md").write_text(
            f"# {cname}\n\nHeadquarters: {city}\nFounded: {year}\n",
            encoding="utf-8",
        )
    return task_dir


def gen_coding(root: Path, idx: int, rng: random.Random) -> Path:
    task_dir = write_task(
        root,
        f"coding_gen_{idx:04d}",
        "coding",
        (
            "Run `python3 test_mylib.py` in the current directory; some tests fail. "
            "Fix the bugs in mylib.py (do not edit the tests) until all tests pass. "
            "Work only inside the current directory. When you are done, write ONLY the word DONE "
            "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
        ),
        rng.randint(1, 3),
        "DONE",
    )
    ws = task_dir / "workspace"
    ws.mkdir(exist_ok=True)
    # Intentionally buggy implementations.
    (ws / "mylib.py").write_text(
        "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n\n"
        "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n\n"
        "def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n\n"
        "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
        encoding="utf-8",
    )
    (ws / "test_mylib.py").write_text(
        "import unittest\nimport mylib as m\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n"
        "    def test_maximum(self):\n        self.assertEqual(m.maximum([3, 9, 2]), 9)\n"
        "    def test_reverse_words(self):\n        self.assertEqual(m.reverse_words('c a b'), 'b a c')\n"
        "    def test_count_even(self):\n        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)\n\n"
        "if __name__ == '__main__':\n    unittest.main()\n",
        encoding="utf-8",
    )
    return task_dir


def generate_training_set(
    out_dir: Path | str,
    n: int = 40,
    seed: int = 0,
    unsolvable_ratio: float = 0.2,
) -> Path:
    """Create a mixed training set (~20% unsolvable)."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        kind = rng.choice(["math", "csv", "search", "coding"])
        broken = rng.random() < unsolvable_ratio
        if kind == "math":
            gen_math(root, i, rng)
        elif kind == "csv":
            gen_csv(root, i, rng, broken=broken)
        elif kind == "search":
            gen_search(root, i, rng, missing=broken)
        else:
            gen_coding(root, i, rng)
    return root


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="train-tasks")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    path = generate_training_set(args.out, n=args.n, seed=args.seed)
    print(f"wrote {args.n} tasks under {path}")
