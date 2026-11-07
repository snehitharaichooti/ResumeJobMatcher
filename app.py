"""ResumeJobMatcher - Streamlit UI.   Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from resume_job_matcher import (DEFAULT_WEIGHTS, Embedder, ResumeJobMatcher,
                                extract_text_from_pdf)

st.set_page_config(page_title="ResumeJobMatcher", page_icon="📄", layout="wide")

SAMPLE_JDS = {
    "Custom (paste below)": "",
    "Machine Learning Engineer": """Machine Learning Engineer
We are looking for an ML Engineer with 3+ years of experience building NLP and machine learning systems.
Requirements: Python, SQL, scikit-learn, TensorFlow or PyTorch, natural language processing, transformers,
Docker, AWS, Git. Experience with MLOps, feature engineering and REST APIs is a plus.
Bachelor's degree in Computer Science or related field. Strong communication and teamwork skills.""",
    "Data Analyst": """Data Analyst
2+ years of experience in data analysis and reporting. Must know SQL, Excel, Tableau or Power BI,
Python (pandas, numpy), data visualization and statistics. Familiarity with A/B testing is a plus.
Bachelor's degree required. Good communication and problem solving skills.""",
    "Full-Stack Developer": """Full-Stack Developer
4+ years of experience building web applications. JavaScript, TypeScript, React, Node.js, REST API,
PostgreSQL, Docker, Git, CI/CD and agile practices. AWS experience preferred. Bachelor's degree.""",
}

LABEL_ICON = {"Strong match": "🟢", "Good match": "🟡", "Moderate match": "🟠", "Weak match": "🔴"}


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(backend: str, model_name: str) -> Embedder:
    return Embedder(backend, model_name)


def chips(items):
    return " ".join(f"`{i}`" for i in items) if items else "_none_"


# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    backend_choice = st.radio(
        "Similarity engine",
        ["Auto (Sentence-BERT, falls back to TF-IDF)", "TF-IDF only (fast, no download)"])
    model_name = st.text_input("Sentence-BERT model", "all-MiniLM-L6-v2")
    st.subheader("Score weights")
    w = {k: st.slider(k.capitalize(), 0.0, 1.0, v, 0.05) for k, v in DEFAULT_WEIGHTS.items()}
    st.caption("Weights are normalised automatically. Components that the job description "
               "does not specify (e.g. years of experience) are skipped.")

embedder = load_embedder("auto" if backend_choice.startswith("Auto") else "tfidf", model_name)

# ---------------------------------------------------------------------------- main
st.title("📄 ResumeJobMatcher")
st.caption("Upload PDF resumes, provide a job description, and get ranked matches with skill-gap advice.")
st.info(f"Similarity engine in use: **{embedder.name}**"
        + (f"  \n_SBERT unavailable ({embedder.load_error}); using TF-IDF fallback._"
           if embedder.load_error else ""))

left, right = st.columns(2)
with left:
    files = st.file_uploader("Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
with right:
    choice = st.selectbox("Job description", list(SAMPLE_JDS))
    jd_text = st.text_area("Job description text", value=SAMPLE_JDS[choice], height=220,
                           key=f"jd_{choice}")

if st.button("🔍 Match & rank", type="primary", disabled=not (files and jd_text.strip())):
    matcher = ResumeJobMatcher(embedder, w)
    resumes, problems = [], []
    with st.spinner("Parsing resumes..."):
        for f in files:
            try:
                text = extract_text_from_pdf(f)
                resumes.append(matcher.analyze_resume(text, f.name))
            except Exception as exc:
                problems.append(f"{f.name}: {exc}")
    for p in problems:
        st.error(f"Could not read {p}")
    if resumes:
        with st.spinner("Computing similarity..."):
            jd = matcher.analyze_job(jd_text)
            st.session_state["results"] = matcher.rank(resumes, jd)
            st.session_state["jd"] = jd

results = st.session_state.get("results")
if results:
    jd = st.session_state["jd"]
    st.subheader("Ranked candidates")
    st.caption(f"Job skills detected: {chips(sorted(jd.skills))}"
               + (f" · experience asked: ~{jd.required_years:g} yrs" if jd.required_years else ""))
    df = pd.DataFrame([{
        "Rank": i, "Candidate": r.display_name, "File": r.filename, "Match %": r.score,
        "Verdict": f"{LABEL_ICON[r.label]} {r.label}",
        "Semantic": round((r.components["semantic"] or 0) * 100),
        "Skills": None if r.components["skills"] is None else round(r.components["skills"] * 100),
        "Exp (yrs)": r.info.experience_years, "Education": r.info.education_label,
        "Missing skills": len(r.missing_skills),
    } for i, r in enumerate(results, 1)])
    st.dataframe(df, hide_index=True, use_container_width=True, column_config={
        "Match %": st.column_config.ProgressColumn("Match %", min_value=0, max_value=100, format="%.1f"),
        "Semantic": st.column_config.ProgressColumn("Semantic", min_value=0, max_value=100, format="%d"),
        "Skills": st.column_config.ProgressColumn("Skills", min_value=0, max_value=100, format="%d"),
    })
    st.download_button("⬇️ Download ranking (CSV)", df.to_csv(index=False), "ranking.csv", "text/csv")
    if len(results) > 1:
        st.bar_chart(df.set_index("Candidate")["Match %"])

    st.subheader("Candidate details")
    for i, r in enumerate(results, 1):
        with st.expander(f"#{i}  {r.display_name} — {r.score:.1f}%  {LABEL_ICON[r.label]} {r.label}",
                         expanded=(i == 1)):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Extracted info**")
                st.write(f"📧 {r.info.email or '—'}  \n📞 {r.info.phone or '—'}  \n"
                         f"🎓 {r.info.education_label}  \n💼 ~{r.info.experience_years:g} yrs experience")
                st.markdown("**Score breakdown**")
                for k, v in r.components.items():
                    if v is None:
                        st.caption(f"{k.capitalize()}: not specified in job description")
                    else:
                        st.progress(float(v), text=f"{k.capitalize()}: {v * 100:.0f}%")
            with c2:
                st.markdown(f"**✅ Matched skills ({len(r.matched_skills)})**  \n{chips(r.matched_skills)}")
                if r.missing_skills:
                    st.markdown("**❌ Missing skills** (est. score gain if added)")
                    st.dataframe(pd.DataFrame(r.missing_skills).rename(columns={
                        "skill": "Skill", "category": "Category", "gain": "Gain (pts)"}),
                        hide_index=True, use_container_width=True)
                if r.suggestions:
                    st.markdown("**💡 Also worth learning / highlighting**  \n" + "  \n".join(
                        f"`{s['skill']}` — {s['because']}" for s in r.suggestions))
                st.markdown(f"**Other skills on resume**  \n{chips(r.extra_skills)}")
                for n in r.notes:
                    st.warning(n)
            with st.expander("Skills by category"):
                for cat, skills in r.info.skills_by_category.items():
                    st.markdown(f"**{cat}:** {chips(skills)}")
