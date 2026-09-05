"""
Transparent rule-based resume insight and role recommendation engine.
Enhanced & Refactored Version.
"""

import re
from typing import Dict, List, Set, Any, Tuple


# ============================================================
# SKILL DATABASE
# ============================================================

TECHNICAL_SKILLS: List[str] = [
    "Python",
    "Java",
    "C++",
    "C#",
    "C",
    "JavaScript",
    "TypeScript",
    "SQL",
    "HTML",
    "CSS",
    "React",
    "Node.js",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "TensorFlow",
    "PyTorch",
    "Scikit-learn",
    "Git",
    "GitHub",
    "Docker",
    "Kubernetes",
    "AWS",
    "Azure",
    "Excel",
    "Power BI",
    "Tableau",
]

SOFT_SKILLS: List[str] = [
    "Communication",
    "Leadership",
    "Teamwork",
    "Problem Solving",
    "Critical Thinking",
    "Time Management",
    "Adaptability",
]


# ============================================================
# CAREER ROLE RULES
# ============================================================

ROLE_RULES: Dict[str, Dict[str, Any]] = {
    "Machine Learning Engineer": {
        "skills": {
            "Python",
            "Machine Learning",
            "NLP",
            "TensorFlow",
            "PyTorch",
            "Scikit-learn",
            "Git",
        },
        "core": {
            "Python",
            "Machine Learning",
            "Scikit-learn",
            "Git",
        },
        "traits": [
            "openness",
            "conscientiousness",
        ],
        "description": (
            "Build machine learning models and intelligent "
            "systems that learn from data and solve practical problems."
        ),
    },
    "Data Analyst": {
        "skills": {
            "Python",
            "SQL",
            "Excel",
            "Power BI",
            "Tableau",
            "Machine Learning",
        },
        "core": {
            "SQL",
            "Excel",
            "Power BI",
            "Python",
        },
        "traits": [
            "conscientiousness",
            "emotional_stability",
        ],
        "description": (
            "Analyze data, discover useful patterns, create "
            "visualizations, and support data-driven decisions."
        ),
    },
    "Frontend Developer": {
        "skills": {
            "JavaScript",
            "TypeScript",
            "HTML",
            "CSS",
            "React",
            "Git",
        },
        "core": {
            "JavaScript",
            "HTML",
            "CSS",
            "React",
        },
        "traits": [
            "openness",
            "extraversion",
        ],
        "description": (
            "Design and develop responsive, interactive user "
            "interfaces and modern web applications."
        ),
    },
    "Backend Developer": {
        "skills": {
            "Python",
            "Java",
            "Node.js",
            "SQL",
            "Docker",
            "AWS",
            "Git",
        },
        "core": {
            "Python",
            "SQL",
            "Git",
            "Docker",
        },
        "traits": [
            "conscientiousness",
            "emotional_stability",
        ],
        "description": (
            "Develop reliable backend services, APIs, databases, "
            "and scalable software systems."
        ),
    },
}


# ============================================================
# KEYWORDS & PATTERNS
# ============================================================

EDUCATION_KEYWORDS: List[str] = [
    "bachelor",
    "master",
    "phd",
    "b.tech",
    "m.tech",
    "b.e.",
    "m.e.",
    "b.s.",
    "m.s.",
    "degree",
    "university",
    "college",
    "gpa",
    "cgpa",
]

EXPERIENCE_KEYWORDS: List[str] = [
    "experience",
    "intern",
    "internship",
    "engineer",
    "developer",
    "analyst",
    "worked",
    "employment",
    "professional",
]

PROJECT_KEYWORDS: List[str] = [
    "project",
    "developed",
    "built",
    "implemented",
    "deployed",
    "created",
    "designed",
    "github",
]


# ============================================================
# TEXT HELPERS
# ============================================================

def _contains(text: str, phrase: str) -> bool:
    """
    Check whether a phrase occurs as a complete word/phrase.
    Uses flexible boundaries to handle special terms like C++, C#, .NET.
    """
    escaped_phrase = re.escape(phrase.lower())
    
    # Custom boundaries for technical terms with special trailing characters
    pattern = r"(?:^|(?<=\W))" + escaped_phrase + r"(?:$|(?=\W))"
    return bool(re.search(pattern, text.lower()))


def _find(text: str, candidates: List[str]) -> List[str]:
    """
    Return all candidate keywords found in the resume.
    """
    return [item for item in candidates if _contains(text, item)]


def _years(text: str) -> str:
    """
    Detect explicitly mentioned years of experience with support for decimals.
    """
    pattern = r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b"
    match = re.search(pattern, text.lower())

    if match:
        return f"{match.group(1)}+ years mentioned"

    # Number words checking
    words_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
    word_pattern = r"\b(one|two|three|four|five)\b\+?\s*(?:years?|yrs?)\b"
    word_match = re.search(word_pattern, text.lower())
    
    if word_match:
        num = words_map[word_match.group(1)]
        return f"{num}+ years mentioned"

    return "No explicit years of experience detected"


# ============================================================
# ROLE TRAIT SCORE
# ============================================================

def _trait_score(role: str, traits: Dict[str, float]) -> float:
    """
    Calculate personality compatibility for a role.
    Defaults safely to 50.0 (neutral) if traits are missing or non-numeric.
    """
    config = ROLE_RULES.get(role, {})
    role_traits = config.get("traits", [])

    if not role_traits or not traits:
        return 50.0

    values: List[float] = []

    for trait in role_traits:
        raw_val = traits.get(trait, 50.0)
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            val = 50.0

        # Clamp between 0.0 and 100.0
        values.append(max(0.0, min(100.0, val)))

    return sum(values) / len(values) if values else 50.0


# ============================================================
# ROLE MATCH SCORE
# ============================================================

def _role_match_score(
    role: str,
    detected_skills: Set[str],
    traits: Dict[str, float]
) -> float:
    """
    Calculate an interpretable role match percentage.
    70% = Technical skill coverage
    30% = Personality trait fit
    """
    config = ROLE_RULES[role]
    role_skills = config["skills"]

    if role_skills:
        # Measure skill fit against specified role skills
        matched = detected_skills.intersection(role_skills)
        skill_fit = len(matched) / len(role_skills)
    else:
        skill_fit = 0.0

    personality_fit = _trait_score(role, traits) / 100.0

    score = (skill_fit * 70.0) + (personality_fit * 30.0)

    return round(max(0.0, min(100.0, score)), 1)


# ============================================================
# RESUME ANALYZER
# ============================================================

def analyze_resume(
    text: str,
    traits: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Analyze resume text and generate actionable career insights.

    Parameters
    ----------
    text:
        Raw or extracted resume text string.
    traits:
        Optional dictionary of personality trait scores (0-100 scale).

    Returns
    -------
    dict:
        Extracted resume signals and top role recommendations.
    """
    if traits is None:
        traits = {}

    # Standardize input string
    cleaned_text = text.strip()

    # --------------------------------------------------------
    # Detect Skills
    # --------------------------------------------------------
    technical = _find(cleaned_text, TECHNICAL_SKILLS)
    soft = _find(cleaned_text, SOFT_SKILLS)
    detected_set = set(technical)

    # --------------------------------------------------------
    # Calculate Role Recommendations
    # --------------------------------------------------------
    scored: List[Tuple[str, float, Set[str], str]] = []

    for role, config in ROLE_RULES.items():
        score = _role_match_score(role, detected_set, traits)
        scored.append((role, score, config["core"], config["description"]))

    # Sort descending by match score
    scored.sort(key=lambda item: item[1], reverse=True)

    recommendations = []
    for role, score, core, description in scored[:3]:
        missing = sorted(list(core.difference(detected_set)))

        recommendations.append({
            "role": role,
            "match_score": score,
            "description": description,
            "missing_skills": missing if missing else ["Core starter skills are present"],
        })

    # --------------------------------------------------------
    # Keyword Signal Indicators
    # --------------------------------------------------------
    education_keywords = _find(cleaned_text, EDUCATION_KEYWORDS)
    experience_keywords = _find(cleaned_text, EXPERIENCE_KEYWORDS)
    project_keywords = _find(cleaned_text, PROJECT_KEYWORDS)

    # --------------------------------------------------------
    # Final Result Construction
    # --------------------------------------------------------
    return {
        "technical_skills": technical,
        "soft_skills": soft,
        "education_keywords": education_keywords,
        "experience_keywords": experience_keywords,
        "project_keywords": project_keywords,
        "experience_indicator": _years(cleaned_text),
        "recommendations": recommendations,
    }


# ============================================================
# EXAMPLE EXECUTION
# ============================================================

if __name__ == "__main__":
    sample_resume = """
    Chittem Gowrishankar
    Computer Science & Technical Student
    
    Summary:
    Software engineer with over 2.5 years of experience building scalable backend 
    systems and intelligent models. Proven history of internship roles.
    
    Education:
    Bachelor of Technology (B.Tech) in Computer Science and Engineering.
    
    Technical Skills:
    Python, C++, Java, SQL, Machine Learning, Scikit-learn, Git, Docker, HTML, CSS.
    
    Soft Skills:
    Problem Solving, Teamwork, Communication.
    
    Projects:
    Developed microservices architecture using Python, SQL, and Docker. 
    Deployed machine learning models using Scikit-learn on GitHub.
    """

    sample_traits = {
        "openness": 85.0,
        "conscientiousness": 90.0,
        "emotional_stability": 75.0,
        "extraversion": 60.0,
    }

    results = analyze_resume(sample_resume, sample_traits)

    print("=== TECHNICAL SKILLS FOUND ===")
    print(results["technical_skills"])

    print("\n=== EXPERIENCE INDICATOR ===")
    print(results["experience_indicator"])

    print("\n=== ROLE RECOMMENDATIONS ===")
    for rec in results["recommendations"]:
        print(f"\nRole: {rec['role']} ({rec['match_score']}% Match)")
        print(f"Summary: {rec['description']}")
        print(f"Missing Core Skills: {', '.join(rec['missing_skills'])}")
