"""Deterministic kb/*.md search solver (1–2 hop lookups, no LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from harness.types import Result, Status, Task


@dataclass
class Person:
    name: str
    born: Optional[str] = None
    employer: Optional[str] = None
    mentor: Optional[str] = None


@dataclass
class Company:
    name: str
    headquarters: Optional[str] = None
    founded: Optional[str] = None


_BORN_RE = re.compile(
    r"In which year was\s+(.+?)\s+born\?",
    re.IGNORECASE,
)
_HQ_EMPLOYER_RE = re.compile(
    r"In which city is the headquarters of\s+(.+?)(?:'s|’s)\s+employer\?",
    re.IGNORECASE,
)
_HQ_COMPANY_RE = re.compile(
    r"In which city is the headquarters of\s+(.+?)\?",
    re.IGNORECASE,
)
_FIELD_RE = {
    "born": re.compile(r"^Born:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE),
    "employer": re.compile(r"^Employer:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE),
    "mentor": re.compile(r"^Mentor:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE),
    "headquarters": re.compile(
        r"^Headquarters:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE
    ),
    "founded": re.compile(r"^Founded:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE),
}
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


def load_kb(kb_dir: Path) -> tuple[dict[str, Person], dict[str, Company]]:
    people: dict[str, Person] = {}
    companies: dict[str, Company] = {}
    if not kb_dir.is_dir():
        return people, companies

    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_m = _TITLE_RE.search(text)
        if not title_m:
            continue
        name = title_m.group(1).strip()
        key = _norm(name)
        if path.name.startswith("person_") or "Born:" in text or "Employer:" in text:
            people[key] = Person(
                name=name,
                born=_field(text, "born"),
                employer=_field(text, "employer"),
                mentor=_field(text, "mentor"),
            )
        elif path.name.startswith("company_") or "Headquarters:" in text:
            companies[key] = Company(
                name=name,
                headquarters=_field(text, "headquarters"),
                founded=_field(text, "founded"),
            )
        else:
            # Heuristic fallback by fields present.
            if _FIELD_RE["born"].search(text) or _FIELD_RE["employer"].search(text):
                people[key] = Person(
                    name=name,
                    born=_field(text, "born"),
                    employer=_field(text, "employer"),
                    mentor=_field(text, "mentor"),
                )
            elif _FIELD_RE["headquarters"].search(text):
                companies[key] = Company(
                    name=name,
                    headquarters=_field(text, "headquarters"),
                    founded=_field(text, "founded"),
                )
    return people, companies


def _field(text: str, name: str) -> Optional[str]:
    m = _FIELD_RE[name].search(text)
    return m.group(1).strip() if m else None


def try_solve_search(instruction: str, kb_dir: Path) -> tuple[Optional[str], str]:
    """Return (answer, reason). answer is None when unsolvable/unrecognized."""
    people, companies = load_kb(kb_dir)
    if not people and not companies:
        return None, "kb_empty_or_missing"

    m = _BORN_RE.search(instruction)
    if m:
        person_name = m.group(1).strip()
        person = people.get(_norm(person_name))
        if person is None:
            return None, "person_missing"
        if not person.born:
            return None, "born_missing"
        return person.born, "born_year"

    m = _HQ_EMPLOYER_RE.search(instruction)
    if m:
        person_name = m.group(1).strip()
        person = people.get(_norm(person_name))
        if person is None:
            return None, "person_missing"
        emp = (person.employer or "").strip()
        if not emp or emp.upper() in {"UNKNOWN", "N/A", "NONE", "?"}:
            return None, "employer_missing"
        company = companies.get(_norm(emp))
        if company is None:
            return None, "company_missing"
        if not company.headquarters:
            return None, "hq_missing"
        return company.headquarters, "employer_hq"

    m = _HQ_COMPANY_RE.search(instruction)
    if m:
        company_name = m.group(1).strip()
        # Avoid matching the employer phrasing twice.
        if "employer" in company_name.lower():
            return None, "parse_failed"
        company = companies.get(_norm(company_name))
        if company is None:
            return None, "company_missing"
        if not company.headquarters:
            return None, "hq_missing"
        return company.headquarters, "company_hq"

    return None, "parse_failed"


class SearchSolver:
    def solve(self, task: Task) -> Result:
        kb_dir = task.work_dir / "kb"
        answer, reason = try_solve_search(task.instruction, kb_dir)
        if answer is None:
            return Result(
                status=Status.UNSOLVABLE,
                cost=0,
                error=f"search failed: {reason}",
                meta={"reason": reason, "solver": "search"},
            )
        return Result(
            status=Status.SOLVED,
            answer=answer,
            cost=0,
            model=None,
            meta={"solver": "search", "method": "deterministic", "reason": reason},
        )
