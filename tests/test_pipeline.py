import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import fitz
from resume_job_matcher import Embedder, ResumeJobMatcher, extract_text_from_pdf

RESUME_ML = """Priya Raman
priya@example.com | +91 98765 43210

SUMMARY
Data scientist with experience building NLP and machine learning models.

SKILLS
Python, SQL, pandas, NumPy, scikit-learn, TensorFlow, Streamlit, Git, Docker

EXPERIENCE
Data Scientist, Acme Analytics  Jan 2021 - Present
- Built text classification pipelines using BERT and transformers
- Deployed models with Docker and Flask REST APIs

EDUCATION
M.Tech in Data Science, 2020
"""
RESUME_WEB = """Arun Kumar
arun@example.com

SKILLS
JavaScript, React, HTML, CSS, Node.js, Git

EXPERIENCE
Frontend Developer, WebCo  Mar 2022 - Dec 2023
- Built responsive UIs with React and CSS

EDUCATION
B.E. Computer Science, 2021
"""
JD = """Machine Learning Engineer
We need 3+ years of experience in NLP and machine learning. Python, SQL, scikit-learn,
TensorFlow or PyTorch, natural language processing, Docker, AWS. Bachelor's degree required."""


def make_pdf(path, text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 555, 800), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_end_to_end(tmp_path):
    paths = {}
    for n, t in {"ml.pdf": RESUME_ML, "web.pdf": RESUME_WEB}.items():
        paths[n] = tmp_path / n
        make_pdf(paths[n], t)
    m = ResumeJobMatcher(Embedder("tfidf"))
    infos = [m.analyze_resume(extract_text_from_pdf(p), n) for n, p in paths.items()]
    results = m.rank(infos, m.analyze_job(JD))
    assert results[0].filename == "ml.pdf"
    assert results[0].score > results[1].score
    top = results[0]
    assert {"python", "sql", "docker"} <= set(top.matched_skills)
    assert {"pytorch", "aws"} <= {d["skill"] for d in top.missing_skills}
    assert top.info.name == "Priya Raman" and top.info.education_level == 3
    assert top.info.experience_years >= 4
    assert all(d["gain"] > 0 for d in top.missing_skills)
    print([(r.filename, r.score) for r in results])
