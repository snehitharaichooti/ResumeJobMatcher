# ResumeJobMatcher – Intelligent Resume–Job Matching System

Pipeline: **PDF parsing (PyMuPDF) → NLP feature extraction → Sentence-BERT-style semantic similarity →
weighted match score → ranking → skill-gap recommendations → Streamlit UI**.

## Run
```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
streamlit run app.py
pytest -q                                              # run the tests
```
`sentence-transformers` downloads `all-MiniLM-L6-v2` on first use. If it is missing/offline, the system
automatically falls back to TF-IDF cosine similarity (the sidebar shows which engine is active).

## How the score works (0–100)
| Component | Default weight | Description |
|---|---|---|
| Semantic | 0.45 | Blend of whole-document cosine similarity and *JD coverage* (each JD chunk matched to its best resume chunk) using SBERT embeddings |
| Skills | 0.35 | Weighted overlap of JD skills found in the resume (skills mentioned more often in the JD weigh more) |
| Experience | 0.12 | Candidate years (from date ranges / "N years") vs. years asked in the JD |
| Education | 0.08 | Highest degree detected vs. degree asked in the JD |

Components the JD doesn't specify are skipped and the remaining weights renormalised.

## Skill-gap analysis
* **Missing skills** – JD skills absent from the resume, each with the estimated score gain (points) if added.
* **Related skills** – adjacent skills (e.g. Python → pandas, scikit-learn) to learn or highlight.

## Layout
```
app.py                       Streamlit interface
resume_job_matcher/
  pdf_parser.py              PyMuPDF text extraction + cleaning
  skills_db.py               skill taxonomy, aliases, related-skill map  (extend this!)
  extractor.py               skills / experience / education / contact extraction
  embedder.py                SBERT embeddings with TF-IDF fallback
  scorer.py                  scoring, ranking, gap analysis
tests/test_pipeline.py       end-to-end test (generates PDFs on the fly)
```
Reference: Reimers & Gurevych, *Sentence-BERT*, EMNLP-IJCNLP 2019.
