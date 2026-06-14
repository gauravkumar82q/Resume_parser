"""
Resume parsing utilities using pdfplumber, python-docx, and spaCy NER.
"""
import re
import io
import pdfplumber
import docx
import spacy


# ---------------------------------------------------------------------------
# Load spaCy model (downloaded via: python -m spacy download en_core_web_sm)
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Common skills keyword bank (extend as needed)
# ---------------------------------------------------------------------------
SKILLS_DB = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "r", "scala", "php", "ruby", "perl", "bash",
    # Web / frontend
    "html", "css", "react", "angular", "vue", "nextjs", "nodejs", "express",
    "django", "flask", "fastapi", "spring", "bootstrap", "tailwind",
    # Data / ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "spacy", "nltk",
    "transformers", "bert", "gpt", "llm", "openai", "langchain",
    "data analysis", "data science", "data engineering", "feature engineering",
    "model deployment", "mlops",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "nosql", "cassandra", "dynamodb",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
    "jenkins", "ci/cd", "github actions", "ansible", "linux", "git",
    # Other
    "agile", "scrum", "rest api", "graphql", "microservices", "spark",
    "hadoop", "kafka", "airflow", "tableau", "power bi", "excel",
}

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)(\d{3}[\s\-]?\d{4})"
)
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file (bytes)."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file (bytes)."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    elif ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


# ---------------------------------------------------------------------------
# NER & regex entity extraction
# ---------------------------------------------------------------------------

def extract_name(text: str, doc) -> str:
    """Extract candidate name using spaCy PERSON entities (first found)."""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    # Fallback: first non-empty line
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first_line[:60]  # cap at 60 chars


def extract_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = PHONE_RE.search(text)
    return match.group(0).strip() if match else ""


def extract_linkedin(text: str) -> str:
    match = LINKEDIN_RE.search(text)
    return match.group(0) if match else ""


def extract_github(text: str) -> str:
    match = GITHUB_RE.search(text)
    return match.group(0) if match else ""


def extract_skills(text: str) -> list[str]:
    """Return skills found in the resume text (case-insensitive)."""
    lower = text.lower()
    found = sorted({skill for skill in SKILLS_DB if skill in lower})
    return found


def extract_education(doc) -> list[str]:
    """Extract ORG entities that look like educational institutions."""
    edu_keywords = {"university", "college", "institute", "school", "academy", "b.tech",
                    "m.tech", "b.sc", "m.sc", "bachelor", "master", "phd", "mba"}
    results = []
    for ent in doc.ents:
        if ent.label_ in ("ORG", "GPE"):
            if any(kw in ent.text.lower() for kw in edu_keywords):
                results.append(ent.text.strip())
    return list(dict.fromkeys(results))  # deduplicate, preserve order


def extract_experience_years(text: str) -> str:
    """Extract years of experience mentioned in the resume."""
    pattern = re.compile(
        r"(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:experience|exp\.?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if matches:
        return max(int(m) for m in matches)
    return 0


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_resume(file_bytes: bytes, filename: str) -> dict:
    """
    Parse a resume and return a structured dict with extracted fields.
    """
    text = extract_text(file_bytes, filename)
    doc = nlp(text[:1_000_000])  # spaCy cap

    return {
        "raw_text": text,
        "name": extract_name(text, doc),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "linkedin": extract_linkedin(text),
        "github": extract_github(text),
        "skills": extract_skills(text),
        "education": extract_education(doc),
        "experience_years": extract_experience_years(text),
    }
