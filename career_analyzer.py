"""Transparent rule-based resume insight and role recommendation engine."""

import re
from typing import Dict, List


# ============================================================
# SKILL DATABASE
# ============================================================

TECHNICAL_SKILLS = [
    "Python",
    "Java",
    "C++",
    "C",
    "JavaScript",
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
    "AWS",
    "Excel",
    "Power BI",
]


SOFT_SKILLS = [
    "Communication",
    "Leadership",
    "Teamwork",
    "Problem Solving",
]


# ============================================================
# CAREER ROLE RULES
# ============================================================

ROLE_RULES = {

    "Machine Learning Engineer": {
        "skills": {
            "Python",
            "Machine Learning",
            "NLP",
            "TensorFlow",
            "PyTorch",
            "Scikit-learn",
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
# KEYWORDS
# ============================================================

EDUCATION_KEYWORDS = [
    "bachelor",
    "master",
    "b.tech",
    "b.e.",
    "degree",
    "university",
    "college",
    "gpa",
    "cgpa",
]


EXPERIENCE_KEYWORDS = [
    "experience",
    "intern",
    "engineer",
    "analyst",
    "worked",
    "employment",
    "professional",
]


PROJECT_KEYWORDS = [
    "project",
    "developed",
    "built",
    "implemented",
    "deployed",
    "github",
]


# ============================================================
# TEXT HELPERS
# ============================================================

def _contains(text: str, phrase: str) -> bool:
    """
    Check whether a phrase occurs as a complete word/phrase.
    """

    return bool(
        re.search(
            r"(?<!\w)"
            + re.escape(phrase.lower())
            + r"(?!\w)",
            text.lower(),
        )
    )


def _find(
    text: str,
    candidates: List[str],
) -> List[str]:
    """
    Return all candidate keywords found in the resume.
    """

    return [
        item
        for item in candidates
        if _contains(text, item)
    ]


def _years(text: str) -> str:
    """
    Detect explicitly mentioned years of experience.
    """

    match = re.search(
        r"(\d{1,2})\+?\s*(?:years|yrs?)",
        text.lower(),
    )

    if match:
        return (
            f"{match.group(1)}+ years mentioned"
        )

    return (
        "No explicit years of experience detected"
    )


# ============================================================
# ROLE TRAIT SCORE
# ============================================================

def _trait_score(
    role: str,
    traits: Dict[str, float],
) -> float:
    """
    Calculate personality compatibility for a role.

    Personality scores are treated as experimental signals
    and should not be interpreted as psychological assessments.
    """

    config = ROLE_RULES[role]

    role_traits = config.get(
        "traits",
        [],
    )

    if not role_traits:
        return 50.0

    values = []

    for trait in role_traits:

        try:
            value = float(
                traits.get(
                    trait,
                    50,
                )
            )
        except (
            ValueError,
            TypeError,
        ):
            value = 50.0

        values.append(
            max(
                0.0,
                min(
                    100.0,
                    value,
                ),
            )
        )

    if not values:
        return 50.0

    return sum(values) / len(values)


# ============================================================
# ROLE MATCH SCORE
# ============================================================

def _role_match_score(
    role: str,
    detected: set,
    traits: Dict[str, float],
) -> float:
    """
    Calculate an interpretable role match percentage.

    70% = resume technical skill fit
    30% = experimental personality-trait fit
    """

    config = ROLE_RULES[role]

    role_skills = config["skills"]

    if role_skills:

        skill_fit = (
            len(
                detected.intersection(
                    role_skills
                )
            )
            / len(role_skills)
        )

    else:

        skill_fit = 0.0

    personality_fit = (
        _trait_score(
            role,
            traits,
        )
        / 100.0
    )

    score = (
        skill_fit * 70
        + personality_fit * 30
    )

    return round(
        max(
            0.0,
            min(
                100.0,
                score,
            ),
        ),
        1,
    )


# ============================================================
# RESUME ANALYZER
# ============================================================

def analyze_resume(
    text: str,
    traits: Dict[str, float],
) -> Dict[str, object]:
    """
    Analyze resume text and generate career insights.

    Parameters
    ----------
    text:
        Extracted resume text.

    traits:
        Personality scores generated by PersonalityPredictor.

    Returns
    -------
    dict:
        Resume signals and career recommendations.
    """

    # --------------------------------------------------------
    # Detect skills
    # --------------------------------------------------------

    technical = _find(
        text,
        TECHNICAL_SKILLS,
    )

    soft = _find(
        text,
        SOFT_SKILLS,
    )

    detected = set(
        technical
    )

    # --------------------------------------------------------
    # Calculate role scores
    # --------------------------------------------------------

    scored = []

    for role, config in ROLE_RULES.items():

        score = _role_match_score(
            role,
            detected,
            traits,
        )

        scored.append(
            (
                role,
                score,
                config["core"],
                config["description"],
            )
        )

    # Highest match first
    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = []

    for (
        role,
        score,
        core,
        description,
    ) in scored[:3]:

        missing = sorted(
            core.difference(
                detected
            )
        )

        if not missing:

            missing = [
                "Core starter skills are present"
            ]

        recommendations.append(
            {
                "role": role,

                "match_score": score,

                "description": description,

                "missing_skills": missing,
            }
        )

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    education_keywords = _find(
        text,
        EDUCATION_KEYWORDS,
    )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    experience_keywords = _find(
        text,
        EXPERIENCE_KEYWORDS,
    )

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    project_keywords = _find(
        text,
        PROJECT_KEYWORDS,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "technical_skills": technical,

        "soft_skills": soft,

        "education_keywords": education_keywords,

        "experience_keywords": experience_keywords,

        "project_keywords": project_keywords,

        "experience_indicator": _years(
            text
        ),

        "recommendations": recommendations,
    }