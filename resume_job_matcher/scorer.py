"""Matching score, ranking and skill-gap recommendations."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embedder import Embedder
from .extractor import EDU_LABELS, JobInfo, ResumeInfo, parse_job, parse_resume
from .skills_db import RELATED, SKILL_TO_CATEGORY

DEFAULT_WEIGHTS = {"semantic": 0.45, "skills": 0.35, "experience": 0.12, "education": 0.08}


def chunk_text(text: str, max_words: int = 70) -> list[str]:
    chunks, cur, n = [], [], 0
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        w = len(line.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def label_for(score: float) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 60:
        return "Good match"
    if score >= 40:
        return "Moderate match"
    return "Weak match"


def _skill_weight(count: int) -> float:
    return 1.0 + math.log(count)


def combine(components: dict, weights: dict) -> float:
    acc = tot = 0.0
    for k, v in components.items():
        if v is None:
            continue
        w = weights.get(k, 0.0)
        acc += w * v
        tot += w
    return acc / tot if tot > 0 else 0.0


@dataclass
class MatchResult:
    filename: str
    name: str | None
    score: float                       # 0-100
    label: str
    components: dict                   # each 0-1 or None if not applicable
    semantic_raw: float
    matched_skills: list
    missing_skills: list               # [{"skill", "category", "gain"}]
    extra_skills: list                 # on resume but not asked for in the JD
    suggestions: list                  # [{"skill", "category", "because"}]
    notes: list
    info: ResumeInfo = field(repr=False, default=None)

    @property
    def display_name(self) -> str:
        return self.name or self.filename


class ResumeJobMatcher:
    def __init__(self, embedder: Embedder | None = None, weights: dict | None = None):
        self.embedder = embedder or Embedder("auto")
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    # ------------------------------------------------------------- parsing helpers
    @staticmethod
    def analyze_resume(text: str, filename: str = "resume.pdf") -> ResumeInfo:
        return parse_resume(text, filename)

    @staticmethod
    def analyze_job(text: str) -> JobInfo:
        return parse_job(text)

    # ------------------------------------------------------------------ components
    def _semantic(self, r_vec, r_chunks, jd_vec, jd_chunks_vec) -> tuple[float, float]:
        if r_vec is None:
            return 0.0, 0.0
        full = float(cosine_similarity(r_vec, jd_vec)[0, 0])
        sims = cosine_similarity(jd_chunks_vec, r_chunks)       # (jd_chunks, resume_chunks)
        coverage = float(sims.max(axis=1).mean())               # how well each JD part is covered
        raw = 0.4 * full + 0.6 * coverage
        lo, hi = self.embedder.calibration
        return float(np.clip((raw - lo) / (hi - lo), 0, 1)), raw

    @staticmethod
    def _skill_score(jd_skills: Counter, have: set) -> float | None:
        if not jd_skills:
            return None
        total = sum(_skill_weight(c) for c in jd_skills.values())
        got = sum(_skill_weight(c) for s, c in jd_skills.items() if s in have)
        return got / total

    @staticmethod
    def _experience_score(res: ResumeInfo, jd: JobInfo) -> float | None:
        if not jd.required_years:
            return None
        return min(1.0, res.experience_years / jd.required_years)

    @staticmethod
    def _education_score(res: ResumeInfo, jd: JobInfo) -> float | None:
        if not jd.required_education:
            return None
        if res.education_level >= jd.required_education:
            return 1.0
        return res.education_level / jd.required_education if res.education_level else 0.25

    # ---------------------------------------------------------------- gap analysis
    def _gap(self, res: ResumeInfo, jd: JobInfo, comps: dict, base_final: float):
        have = set(res.skills)
        missing = []
        for skill in jd.skills:
            if skill in have:
                continue
            what_if = dict(comps)
            what_if["skills"] = self._skill_score(jd.skills, have | {skill})
            gain = (combine(what_if, self.weights) - base_final) * 100
            missing.append({"skill": skill, "category": SKILL_TO_CATEGORY[skill],
                            "gain": round(gain, 1)})
        missing.sort(key=lambda d: (-d["gain"], d["skill"]))

        suggestions, seen = [], set(have) | set(jd.skills)
        for skill in sorted(have, key=lambda s: -res.skills[s]):
            for rel in RELATED.get(skill, []):
                if rel not in seen:
                    seen.add(rel)
                    suggestions.append({"skill": rel, "category": SKILL_TO_CATEGORY[rel],
                                        "because": f"pairs well with {skill}"})
        return missing, suggestions[:8]

    # --------------------------------------------------------------------- ranking
    def rank(self, resumes: list[ResumeInfo], jd: JobInfo) -> list[MatchResult]:
        jd_chunks = chunk_text(jd.text) or [jd.text or " "]
        res_chunks = [chunk_text(r.text) for r in resumes]
        corpus = [jd.text] + [r.text for r in resumes] + jd_chunks + [c for cs in res_chunks for c in cs]
        self.embedder.prepare(corpus)
        jd_vec = self.embedder.encode([jd.text or " "])
        jd_chunks_vec = self.embedder.encode(jd_chunks)

        results = []
        for res, chunks in zip(resumes, res_chunks):
            if chunks:
                r_vec, r_chunks = self.embedder.encode([res.text]), self.embedder.encode(chunks)
            else:
                r_vec = r_chunks = None
            sem, raw = self._semantic(r_vec, r_chunks, jd_vec, jd_chunks_vec)
            comps = {
                "semantic": sem,
                "skills": self._skill_score(jd.skills, set(res.skills)),
                "experience": self._experience_score(res, jd),
                "education": self._education_score(res, jd),
            }
            final = combine(comps, self.weights)
            missing, suggestions = self._gap(res, jd, comps, final)

            notes = []
            if not chunks:
                notes.append("No text could be extracted (scanned/image-only PDF?). Run OCR first.")
            if jd.required_years and res.experience_years < jd.required_years:
                notes.append(f"Job asks for ~{jd.required_years:g} yrs of experience; resume shows "
                             f"~{res.experience_years:g}. Make dates and project impact explicit.")
            if jd.required_education and res.education_level < jd.required_education:
                notes.append(f"Job expects {EDU_LABELS[jd.required_education]} level education; "
                             f"detected: {res.education_label}.")
            results.append(MatchResult(
                filename=res.filename, name=res.name, score=round(final * 100, 1),
                label=label_for(final * 100), components=comps, semantic_raw=raw,
                matched_skills=sorted(s for s in jd.skills if s in res.skills),
                missing_skills=missing,
                extra_skills=sorted(s for s in res.skills if s not in jd.skills),
                suggestions=suggestions, notes=notes, info=res))
        results.sort(key=lambda m: m.score, reverse=True)
        return results

    def match(self, resume: ResumeInfo, jd: JobInfo) -> MatchResult:
        return self.rank([resume], jd)[0]
