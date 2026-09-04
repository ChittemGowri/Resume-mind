"""Transparent rule-based resume insight and role recommendation engine."""
import re
from typing import Dict, List

TECHNICAL_SKILLS = ["Python", "Java", "C++", "C", "JavaScript", "SQL", "HTML", "CSS", "React", "Node.js", "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch", "Scikit-learn", "Git", "GitHub", "Docker", "AWS", "Excel", "Power BI"]
SOFT_SKILLS = ["Communication", "Leadership", "Teamwork", "Problem Solving"]
ROLE_RULES = {
    "Machine Learning Engineer": {"skills": {"Python", "Machine Learning", "NLP", "TensorFlow", "PyTorch", "Scikit-learn"}, "core": {"Python", "Machine Learning", "Scikit-learn", "Git"}},
    "Data Analyst": {"skills": {"Python", "SQL", "Excel", "Power BI", "Machine Learning"}, "core": {"SQL", "Excel", "Power BI", "Python"}},
    "Frontend Developer": {"skills": {"JavaScript", "HTML", "CSS", "React", "Git"}, "core": {"JavaScript", "HTML", "CSS", "React"}},
    "Backend Developer": {"skills": {"Python", "Java", "Node.js", "SQL", "Docker", "AWS"}, "core": {"Python", "SQL", "Git", "Docker"}},
}
EDUCATION_KEYWORDS = ["bachelor", "master", "b.tech", "b.e.", "degree", "university", "college", "gpa", "cgpa"]
EXPERIENCE_KEYWORDS = ["experience", "intern", "engineer", "analyst", "worked", "employment", "professional"]
PROJECT_KEYWORDS = ["project", "developed", "built", "implemented", "deployed", "github"]


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", text.lower()))


def _find(text: str, candidates: List[str]) -> List[str]:
    return [item for item in candidates if _contains(text, item)]


def _years(text: str) -> str:
    match = re.search(r"(\d{1,2})\+?\s*(?:years|yrs?)", text.lower())
    return f"{match.group(1)}+ years mentioned" if match else "No explicit years of experience detected"


def analyze_resume(text: str, traits: Dict[str, float]) -> Dict[str, object]:
    technical = _find(text, TECHNICAL_SKILLS)
    soft = _find(text, SOFT_SKILLS)
    detected = set(technical)
    scored = []
    for role, config in ROLE_RULES.items():
        skill_fit = len(detected.intersection(config["skills"]))
        trait_fit = 0
        if role == "Machine Learning Engineer": trait_fit = traits.get("openness", 0) + traits.get("conscientiousness", 0)
        elif role == "Data Analyst": trait_fit = traits.get("conscientiousness", 0) + traits.get("emotional_stability", 0)
        elif role == "Frontend Developer": trait_fit = traits.get("openness", 0) + traits.get("extraversion", 0)
        else: trait_fit = traits.get("conscientiousness", 0) + traits.get("problem_solving", 50)
        scored.append((role, skill_fit * 20 + trait_fit / 10, config["core"]))
    scored.sort(key=lambda item: item[1], reverse=True)
    recommendations = []
    for role, _, core in scored[:3]:
        missing = sorted(core.difference(detected))
        recommendations.append({"role": role, "missing_skills": missing or ["Core starter skills are present"]})
    return {
        "technical_skills": technical,
        "soft_skills": soft,
        "education_keywords": _find(text, EDUCATION_KEYWORDS),
        "experience_keywords": _find(text, EXPERIENCE_KEYWORDS),
        "project_keywords": _find(text, PROJECT_KEYWORDS),
        "experience_indicator": _years(text),
        "recommendations": recommendations,
    }

