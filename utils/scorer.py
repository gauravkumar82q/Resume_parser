"""
ATS scoring utilities using a weighted multi-signal model.

Final score weights:
- 40% keyword match
- 30% skill match
- 20% semantic similarity
- 10% resume structure
"""
import re
import string
import importlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_EMBED_MODEL = None


# ---------------------------------------------------------------------------
# English stopwords (inline — no NLTK required)
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "that", "this", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "their", "which", "who", "whom",
    "what", "as", "if", "not", "no", "so", "also", "than", "then",
    "when", "where", "how", "all", "each", "every", "both", "more",
    "most", "other", "such", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "up", "down", "about",
    "against", "within", "without", "including", "across", "over",
    "any", "few", "some", "many", "well", "just", "only", "even",
    "looking", "seeking", "required", "requirements", "responsibilities",
    "join", "role", "position", "candidate", "must", "strong", "good",
    "great", "excellent", "preferred", "plus", "bonus", "like", "work",
    "working", "team", "company", "new", "make", "use", "using", "used",
    "help", "ensure", "provide", "support", "ability", "skills", "skill",
    "knowledge", "understanding", "familiar", "familiarity", "proficient",
    "proficiency", "minimum", "years", "year", "month",
}

# Synonym / concept normalization map.
# If any alias is found, the canonical phrase is appended to the text so
# lexical matching can still capture concept-level equivalence.
_CONCEPTS = {
    "natural language processing": ["nlp", "natural language processing"],
    "machine learning": ["ml", "machine learning", "scikit-learn", "sklearn"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
    "computer vision": ["computer vision", "cv"],
    "rest api": ["rest api", "restful api", "api development", "apis"],
    "react": ["react", "react.js", "reactjs"],
    "frontend development": ["frontend", "front end", "frontend development"],
    "nodejs": ["nodejs", "node.js"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "ci cd": ["ci/cd", "continuous integration", "continuous delivery"],
}

_SKILL_ONTOLOGY = {
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular"],
    "vue": ["vue"],
    "nodejs": ["nodejs", "node.js"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "sql": ["sql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "azure": ["azure", "microsoft azure"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab"],
    "ci cd": ["ci/cd", "continuous integration", "continuous delivery"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn"],
    "deep learning": ["deep learning", "neural network", "neural networks"],
    "nlp": ["nlp", "natural language processing"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "rest api": ["rest api", "restful api", "api", "apis"],
}

_WEIGHTS = {
    "keyword": 0.40,
    "skill": 0.30,
    "semantic": 0.20,
    "structure": 0.10,
}

# ---------------------------------------------------------------------------
# Simple rule-based stemmer (no external library needed)
# Strips common English suffixes so develop/developer/development all match
# ---------------------------------------------------------------------------
_SUFFIXES = [
    "ization", "isation", "ations", "nesses", "ities",
    "ation", "ness", "ment", "tion", "sion", "ity",
    "ings", "iers", "ies",
    "ing", "ers", "ied", "ier",
    "ed", "er", "ly", "al", "es", "s",
]

def _stem(word: str) -> str:
    """Strip common suffixes to collapse word variants."""
    if len(word) <= 3:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
    return bool(re.search(pattern, text))


def _augment_with_concepts(text: str) -> str:
    """Append canonical concept phrases when any alias is detected."""
    lower = text.lower()
    concepts = []
    for canonical, aliases in _CONCEPTS.items():
        if any(_contains_phrase(lower, alias.lower()) for alias in aliases):
            concepts.append(canonical)
    if concepts:
        return lower + " " + " ".join(concepts)
    return lower


def _clean(text: str) -> str:
    """Lowercase, normalize concepts, remove punctuation/numbers/URLs, strip stopwords."""
    text = _augment_with_concepts(text)
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = [w for w in text.split() if w not in _STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def _stem_set(text: str) -> set[str]:
    """Return the set of stemmed meaningful tokens from text."""
    return {_stem(w) for w in _clean(text).split()}


def _word_set(text: str) -> set[str]:
    """Return raw (unstemmed) meaningful tokens — used for display."""
    return set(_clean(text).split())


def _extract_skill_set(text: str) -> set[str]:
    """Extract normalized skill tags from text using alias matching."""
    lower = text.lower()
    found = set()
    for canonical, aliases in _SKILL_ONTOLOGY.items():
        if any(_contains_phrase(lower, alias.lower()) for alias in aliases):
            found.add(canonical)
    return found


# ---------------------------------------------------------------------------
# Keyword extraction from JD
# ---------------------------------------------------------------------------

def _extract_keywords_from_jd(jd_text: str, top_n: int = 40) -> list[str]:
    """Extract top unigram keywords from JD by term frequency."""
    cleaned = _clean(jd_text)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=top_n,
        use_idf=False,
        stop_words=None,
    )
    vectorizer.fit([cleaned])
    return list(vectorizer.get_feature_names_out())


# ---------------------------------------------------------------------------
# Component scoring
# ---------------------------------------------------------------------------

def _keyword_match_score(resume_text: str, jd_text: str) -> float:
    jd_keywords = _extract_keywords_from_jd(jd_text, top_n=40)
    resume_stems = _stem_set(resume_text)
    jd_kw_stems = {_stem(kw) for kw in jd_keywords}
    if not jd_kw_stems:
        return 0.0
    return len(resume_stems & jd_kw_stems) / len(jd_kw_stems)


def _skill_match_score(resume_text: str, jd_text: str) -> float:
    jd_skills = _extract_skill_set(jd_text)
    resume_skills = _extract_skill_set(resume_text)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)


def _semantic_similarity_score(resume_text: str, jd_text: str) -> tuple[float, str]:
    """
    Semantic similarity in [0, 1].
    - Preferred: SentenceTransformer embeddings.
    - Fallback: character n-gram TF-IDF cosine.
    """
    global _EMBED_MODEL
    resume_norm = _augment_with_concepts(resume_text)
    jd_norm = _augment_with_concepts(jd_text)

    try:
        st_mod = importlib.import_module("sentence_transformers")
        sentence_transformer_cls = getattr(st_mod, "SentenceTransformer")
        if _EMBED_MODEL is None:
            _EMBED_MODEL = sentence_transformer_cls("all-MiniLM-L6-v2")
        vecs = _EMBED_MODEL.encode([resume_norm, jd_norm], normalize_embeddings=True)
        sim = float(vecs[0] @ vecs[1])
        return max(0.0, min(sim, 1.0)), "sentence-transformers"
    except Exception:
        pass

    # Robust fallback: captures partial/semantic-ish overlap in phrasing.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    mat = vectorizer.fit_transform([resume_norm.lower(), jd_norm.lower()])
    sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    return max(0.0, min(sim, 1.0)), "char-ngram-fallback"


def _structure_score(resume_text: str) -> float:
    """Simple resume-format quality score in [0, 1]."""
    text = resume_text
    checks = [
        bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", text)),  # email
        bool(re.search(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)(\d{3}[\s\-]?\d{4})", text)),  # phone
        bool(re.search(r"\b(summary|objective|profile)\b", text, re.I)),
        bool(re.search(r"\b(experience|employment|work history)\b", text, re.I)),
        bool(re.search(r"\b(education|university|college|bachelor|master|phd)\b", text, re.I)),
        bool(re.search(r"\b(skills|technologies|tech stack)\b", text, re.I)),
        bool(re.search(r"\b(projects?)\b", text, re.I)),
        bool(re.search(r"^\s*[-•]\s+", text, re.M)),
        250 <= len(text.split()) <= 1400,
    ]
    return sum(1 for c in checks if c) / len(checks)


def get_score_breakdown(resume_text: str, jd_text: str) -> dict:
    """Return weighted ATS components and final score."""
    if not resume_text.strip() or not jd_text.strip():
        return {
            "keyword_match": 0.0,
            "skill_match": 0.0,
            "semantic_similarity": 0.0,
            "resume_structure": 0.0,
            "final_score": 0.0,
            "semantic_method": "n/a",
        }

    keyword = _keyword_match_score(resume_text, jd_text)
    skill = _skill_match_score(resume_text, jd_text)
    semantic, semantic_method = _semantic_similarity_score(resume_text, jd_text)
    structure = _structure_score(resume_text)

    final = (
        _WEIGHTS["keyword"] * keyword
        + _WEIGHTS["skill"] * skill
        + _WEIGHTS["semantic"] * semantic
        + _WEIGHTS["structure"] * structure
    )

    return {
        "keyword_match": round(keyword * 100, 2),
        "skill_match": round(skill * 100, 2),
        "semantic_similarity": round(semantic * 100, 2),
        "resume_structure": round(structure * 100, 2),
        "final_score": round(final * 100, 2),
        "semantic_method": semantic_method,
    }

def compute_ats_score(resume_text: str, jd_text: str) -> float:
    """
    Returns an ATS match score (0–100).

    Blends two signals:
      1. TF cosine similarity between cleaned resume and JD text.
      2. Stemmed keyword recall: what fraction of the top JD keywords
         (after stemming) are covered by the resume (after stemming).
         Stemming ensures develop/developer/development all match.

    Blend: 40% TF-cosine + 60% stemmed-recall
    """
    return get_score_breakdown(resume_text, jd_text)["final_score"]


# ---------------------------------------------------------------------------
# Keyword gap analysis (also stemmed)
# ---------------------------------------------------------------------------

def keyword_gap_analysis(resume_text: str, jd_text: str, top_n: int = 40) -> dict:
    """
    Compare JD keywords vs resume using stemmed matching so that
    'manage'/'managed'/'management' etc. all count as a match.
    """
    jd_keywords = _extract_keywords_from_jd(jd_text, top_n=top_n)
    resume_stems = _stem_set(resume_text)

    matched = [kw for kw in jd_keywords if _stem(kw) in resume_stems]
    missing = [kw for kw in jd_keywords if _stem(kw) not in resume_stems]

    match_pct = round(len(matched) / len(jd_keywords) * 100, 1) if jd_keywords else 0.0

    return {
        "jd_keywords": jd_keywords,
        "matched": matched,
        "missing": missing,
        "match_pct": match_pct,
    }


# ---------------------------------------------------------------------------
# Improvement suggestions
# ---------------------------------------------------------------------------

_SUGGESTIONS = [
    ("quantify achievements",
     lambda r: not bool(re.search(r"\d+\s*%|\$\s*\d+|\d+\s*(million|billion|k\b)", r, re.I)),
     "Add quantifiable achievements (e.g., 'Improved performance by 30%', 'Managed $1M budget')."),

    ("action verbs",
     lambda r: len(re.findall(
         r"\b(led|built|developed|designed|implemented|improved|reduced|achieved|"
         r"delivered|managed|created|launched|optimized|automated)\b", r, re.I)) < 5,
     "Use strong action verbs (Led, Built, Developed, Optimized…) to start bullet points."),

    ("contact information",
     lambda r: not bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", r)),
     "Ensure your email address is clearly visible."),

    ("linkedin profile",
     lambda r: "linkedin" not in r.lower(),
     "Include your LinkedIn profile URL."),

    ("summary / objective",
     lambda r: not bool(re.search(r"\b(summary|objective|profile|about me)\b", r, re.I)),
     "Add a professional summary or objective at the top of your resume."),

    ("certifications",
     lambda r: not bool(re.search(
         r"\b(certified|certification|certificate|aws|gcp|azure|pmp|cissp|cpa)\b", r, re.I)),
     "Mention relevant certifications to stand out."),

    ("resume length",
     lambda r: len(r.split()) < 200,
     "Your resume seems too short. Add more detail about your experience and projects."),

    ("resume length",
     lambda r: len(r.split()) > 1200,
     "Your resume may be too long. Aim for 1–2 pages (roughly 400–800 words)."),
]


def generate_suggestions(resume_text: str, missing_keywords: list[str]) -> list[str]:
    """Return a list of actionable improvement suggestions."""
    tips = []
    for _label, check_fn, message in _SUGGESTIONS:
        if check_fn(resume_text):
            tips.append(message)

    if missing_keywords:
        sample = ", ".join(f'"{k}"' for k in missing_keywords[:8])
        tips.append(
            f"Add missing JD keywords to your resume: {sample}"
            + (" and more." if len(missing_keywords) > 8 else ".")
        )

    return tips


# ---------------------------------------------------------------------------
# Score band label
# ---------------------------------------------------------------------------

def score_label(score: float) -> tuple[str, str]:
    """Return (label, color) for a given ATS score."""
    if score >= 80:
        return "Excellent Match", "#2ecc71"
    elif score >= 60:
        return "Good Match", "#27ae60"
    elif score >= 40:
        return "Average Match", "#f39c12"
    elif score >= 20:
        return "Below Average", "#e67e22"
    else:
        return "Poor Match", "#e74c3c"
