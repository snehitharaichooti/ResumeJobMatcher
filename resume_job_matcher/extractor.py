"""Rule-based NLP feature extraction: skills, experience, education, contact info."""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass, field

from .skills_db import ALIASES, SKILL_TO_CATEGORY, SPECIAL_PATTERNS

EDU_LABELS = {0: "Not specified", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}

# ----------------------------------------------------------------------------- skills
_L, _R = r"(?<![A-Za-z0-9+#])", r"(?![A-Za-z0-9+#])"


def _term_regex(term: str) -> str:
    out = []
    for ch in term:
        if ch == " ":
            out.append(r"[\s\-]+")
        elif ch == "-":
            out.append(r"[\s\-]?")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _build_skill_patterns() -> dict[str, re.Pattern]:
    terms: dict[str, list[str]] = {s: [s] for s in SKILL_TO_CATEGORY}
    for alias, canon in ALIASES.items():
        if canon in terms:
            terms[canon].append(alias)
    pats = {}
    for skill, variants in terms.items():
        if skill in SPECIAL_PATTERNS:
            pats[skill] = re.compile(SPECIAL_PATTERNS[skill])
        else:
            alt = "|".join(_term_regex(v) for v in sorted(set(variants), key=len, reverse=True))
            pats[skill] = re.compile(f"{_L}(?:{alt}){_R}", re.I)
    return pats


_SKILL_PATTERNS = _build_skill_patterns()


def extract_skills(text: str) -> Counter:
    found: Counter = Counter()
    for skill, rx in _SKILL_PATTERNS.items():
        n = sum(1 for _ in rx.finditer(text))
        if n:
            found[skill] = n
    return found


# ------------------------------------------------------------------------- sections
_HEADINGS = {
    "experience": ["experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "internships", "internship",
                   "internship experience"],
    "education": ["education", "academic background", "academics", "qualifications",
                  "educational qualifications", "academic qualifications"],
    "skills": ["skills", "technical skills", "key skills", "core competencies",
               "skills & tools", "skills and tools"],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "summary": ["summary", "profile", "professional summary", "objective", "about me",
                "career objective"],
    "certifications": ["certifications", "certificates", "achievements", "awards"],
}
_HEAD_LOOKUP = {h: k for k, hs in _HEADINGS.items() for h in hs}


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "header"
    for line in text.splitlines():
        norm = re.sub(r"[^a-z& ]", "", line.lower()).strip()
        if len(line) < 40 and norm in _HEAD_LOOKUP:
            current = _HEAD_LOOKUP[norm]
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# ---------------------------------------------------------------------- experience
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_RANGE = re.compile(
    rf"(?:(?P<m1>{_MON})\.?,?\s+)?(?P<y1>(?:19|20)\d{{2}})\s*(?:-|–|—|to)\s*"
    rf"(?:(?P<m2>{_MON})\.?,?\s+)?(?P<y2>(?:19|20)\d{{2}}|present|current|now|till date)",
    re.I,
)
_EXPLICIT_YEARS = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b[^.\n]{0,40}?experience", re.I)


def _to_float_date(year: int, month: str | None) -> float:
    m = _MONTHS.get(month[:3].lower(), 1) if month else 1
    return year + (m - 1) / 12


def estimate_experience_years(text: str, sections: dict[str, str]) -> float:
    today = dt.date.today()
    now_f = today.year + (today.month - 1) / 12
    src = sections.get("experience") or "\n".join(
        v for k, v in sections.items() if k not in ("education", "certifications"))
    intervals = []
    for m in _RANGE.finditer(src):
        start = _to_float_date(int(m["y1"]), m["m1"])
        y2 = m["y2"].lower()
        end = now_f if not y2.isdigit() else _to_float_date(int(y2), m["m2"])
        if 1970 <= start <= end <= now_f + 0.1 and end - start < 40:
            intervals.append((start, min(end, now_f)))
    intervals.sort()
    merged: list[list[float]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    computed = sum(e - s for s, e in merged)
    explicit = max([float(x) for x in _EXPLICIT_YEARS.findall(text) if float(x) <= 40] or [0.0])
    return round(max(computed, explicit), 1)


def required_years_from_jd(text: str) -> float | None:
    vals = []
    for m in re.finditer(
            r"(\d{1,2})\s*(?:\+|\s*(?:-|–|to)\s*\d{1,2})?\s*(?:years?|yrs?)\b", text, re.I):
        window = text[max(0, m.start() - 60): m.end() + 60].lower()
        if "experience" in window:
            vals.append(min(float(m.group(1)), 15))
    return max(vals) if vals else None


# ------------------------------------------------------------------------ education
_EDU_PATTERNS = {
    4: [re.compile(r"\b(?:ph\.?\s?d|doctorate|doctoral)\b", re.I)],
    3: [re.compile(r"\b(?:master(?:'|’)s|masters?\s+(?:of|degree|in|program)|mba|m\.?tech|m\.?sc|mca)\b", re.I),
        re.compile(r"(?<![A-Za-z])(?:M\.?\s?Tech|M\.?\s?Sc|MBA|MCA|M\.E\.)(?![A-Za-z])")],
    2: [re.compile(r"\b(?:bachelor(?:'|’)?s?|undergraduate degree|b\.?\s?tech|b\.?\s?sc|bca|b\.?\s?com)\b", re.I),
        re.compile(r"(?<![A-Za-z])(?:B\.?\s?Tech|B\.?\s?Sc|BCA|B\.E\.?|B\.S\.?|BS)(?![A-Za-z])")],
    1: [re.compile(r"\bdiploma\b", re.I)],
}


def detect_education_levels(text: str) -> set[int]:
    return {lvl for lvl, pats in _EDU_PATTERNS.items() if any(p.search(text) for p in pats)}


# ---------------------------------------------------------------------- dataclasses
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)\d{3,5}[\s-]?\d{3,5}")


@dataclass
class ResumeInfo:
    filename: str
    text: str
    name: str | None
    email: str | None
    phone: str | None
    skills: Counter
    experience_years: float
    education_level: int
    sections: dict = field(default_factory=dict)

    @property
    def education_label(self) -> str:
        return EDU_LABELS[self.education_level]

    @property
    def skills_by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in sorted(self.skills):
            out.setdefault(SKILL_TO_CATEGORY[s], []).append(s)
        return out


@dataclass
class JobInfo:
    text: str
    title: str | None
    skills: Counter
    required_years: float | None
    required_education: int | None


def _guess_name(text: str) -> str | None:
    for line in text.splitlines()[:6]:
        line = line.strip()
        words = line.split()
        if (2 <= len(words) <= 4 and not re.search(r"[\d@:/]", line)
                and re.fullmatch(r"[A-Za-z.' -]+", line)
                and re.sub(r"[^a-z& ]", "", line.lower()).strip() not in _HEAD_LOOKUP):
            return line.title() if line.isupper() else line
    return None


def parse_resume(text: str, filename: str = "resume.pdf") -> ResumeInfo:
    sections = split_sections(text)
    edu_src = sections.get("education") or text
    levels = detect_education_levels(edu_src)
    email = _EMAIL.search(text)
    phone = _PHONE.search(text[:600])
    return ResumeInfo(
        filename=filename, text=text, name=_guess_name(text),
        email=email.group(0) if email else None,
        phone=phone.group(0).strip() if phone else None,
        skills=extract_skills(text),
        experience_years=estimate_experience_years(text, sections),
        education_level=max(levels) if levels else 0,
        sections=sections,
    )


def parse_job(text: str) -> JobInfo:
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    levels = detect_education_levels(text)
    return JobInfo(
        text=text, title=first if len(first) < 80 else None, skills=extract_skills(text),
        required_years=required_years_from_jd(text),
        required_education=min(levels) if levels else None,
    )
