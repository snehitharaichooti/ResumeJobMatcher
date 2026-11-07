"""Skill taxonomy, aliases and 'related skill' map used for extraction and gap analysis."""

SKILL_TAXONOMY = {
    "Programming": [
        "python", "java", "c++", "c#", "r", "sql", "javascript", "typescript", "scala",
        "golang", "rust", "matlab", "bash", "kotlin", "swift", "php", "julia",
    ],
    "Machine Learning & AI": [
        "machine learning", "deep learning", "natural language processing", "computer vision",
        "reinforcement learning", "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost",
        "lightgbm", "transformers", "hugging face", "bert", "sentence-bert", "langchain", "llm",
        "feature engineering", "model deployment", "mlops", "time series",
        "recommendation systems", "statistics", "a/b testing", "text classification",
    ],
    "Data & Analytics": [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "tableau", "power bi",
        "excel", "spark", "hadoop", "airflow", "etl", "data visualization", "data analysis",
        "data engineering", "data cleaning", "big data",
    ],
    "Databases": [
        "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "elasticsearch",
        "cassandra", "nosql", "snowflake", "bigquery",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "git",
        "github", "linux", "ansible", "devops",
    ],
    "Web & Backend": [
        "flask", "django", "fastapi", "streamlit", "react", "angular", "vue", "node.js",
        "express.js", "rest api", "graphql", "html", "css", "spring boot", "microservices",
    ],
    "Engineering Practices": [
        "unit testing", "pytest", "oop", "data structures", "algorithms", "system design",
    ],
    "Professional Skills": [
        "agile", "scrum", "jira", "communication", "leadership", "teamwork", "problem solving",
        "project management", "stakeholder management", "mentoring",
    ],
}

SKILL_TO_CATEGORY = {}
for _cat, _skills in SKILL_TAXONOMY.items():
    for _s in _skills:
        SKILL_TO_CATEGORY.setdefault(_s, _cat)

# alias -> canonical skill
ALIASES = {
    "js": "javascript", "ts": "typescript", "ml": "machine learning", "dl": "deep learning",
    "nlp": "natural language processing", "sklearn": "scikit-learn", "tf": "tensorflow",
    "k8s": "kubernetes", "postgres": "postgresql", "mongo": "mongodb", "nodejs": "node.js",
    "large language models": "llm", "large language model": "llm", "llms": "llm",
    "huggingface": "hugging face", "sbert": "sentence-bert", "restful apis": "rest api",
    "restful api": "rest api", "rest apis": "rest api", "cicd": "ci/cd",
    "google cloud": "gcp", "amazon web services": "aws", "object oriented programming": "oop",
    "object-oriented programming": "oop", "ms excel": "excel", "microsoft excel": "excel",
    "ab testing": "a/b testing", "recommender systems": "recommendation systems",
    "recommender system": "recommendation systems", "problem-solving": "problem solving",
    "team work": "teamwork", "vuejs": "vue", "springboot": "spring boot", "pyspark": "spark",
    "go lang": "golang", "computer-vision": "computer vision", "dataviz": "data visualization",
    "exploratory data analysis": "data analysis", "eda": "data analysis",
    "text mining": "text classification",
}

# Skills that are ambiguous common words / letters: matched case-sensitively with custom patterns.
SPECIAL_PATTERNS = {
    "r": r"(?<![\w+#&/.\-])R(?![\w+#&/\-])(?!\.\w)",
    "react": r"\bReact(?:\.?js)?\b",
    "spark": r"(?i:pyspark|apache spark)|\bSpark\b",
    "swift": r"\bSwift\b",
    "rust": r"\bRust\b",
    "julia": r"\bJulia\b",
}

# Adjacent skills worth learning / highlighting once a candidate already has the key skill.
RELATED = {
    "python": ["pandas", "numpy", "scikit-learn"],
    "machine learning": ["scikit-learn", "feature engineering", "mlops"],
    "deep learning": ["pytorch", "tensorflow"],
    "natural language processing": ["transformers", "hugging face", "bert"],
    "sql": ["postgresql", "mysql"],
    "docker": ["kubernetes", "ci/cd"],
    "aws": ["docker", "terraform"],
    "pandas": ["numpy", "data cleaning"],
    "flask": ["rest api", "docker"],
    "streamlit": ["fastapi", "docker"],
    "react": ["typescript", "rest api"],
    "tableau": ["power bi", "sql"],
    "spark": ["airflow", "big data"],
    "git": ["ci/cd", "github"],
    "tensorflow": ["keras", "mlops"],
    "data analysis": ["statistics", "data visualization"],
    "statistics": ["a/b testing", "time series"],
    "javascript": ["typescript", "node.js"],
    "django": ["rest api", "postgresql"],
    "transformers": ["hugging face", "llm"],
}
