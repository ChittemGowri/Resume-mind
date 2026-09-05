"""ResumeMind - Resume analysis Streamlit application."""

import streamlit as st

from career_analyzer import analyze_resume
from predictor import ModelUnavailableError, PersonalityPredictor
from resume_parser import ResumeParseError, extract_resume_text


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ResumeMind",
    page_icon="🧠",
    layout="wide",
)


TRAIT_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "emotional_stability": "Emotional Stability",
}


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_predictor():
    """Load and cache the personality prediction model."""
    return PersonalityPredictor("models")


# =========================================================
# DATA VALIDATION
# =========================================================

def validate_resume_text(text):
    """Validate extracted resume text."""

    if text is None:
        return False

    if not isinstance(text, str):
        return False

    if not text.strip():
        return False

    return True


def normalize_scores(scores):
    """Normalize personality scores into a safe dictionary."""

    if not isinstance(scores, dict):
        raise ValueError("Personality model returned invalid scores.")

    result = {}

    for trait, value in scores.items():

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        value = max(0.0, min(100.0, value))

        result[trait] = value

    if not result:
        raise ValueError("No valid personality scores were returned.")

    return result


def normalize_insights(insights):
    """Ensure career analysis has the expected structure."""

    if not isinstance(insights, dict):
        raise ValueError("Career analyzer returned invalid data.")

    return {
        "technical_skills": insights.get(
            "technical_skills",
            [],
        ),
        "soft_skills": insights.get(
            "soft_skills",
            [],
        ),
        "education_keywords": insights.get(
            "education_keywords",
            [],
        ),
        "experience_indicator": insights.get(
            "experience_indicator",
            "No experience information detected.",
        ),
        "project_keywords": insights.get(
            "project_keywords",
            [],
        ),
        "recommendations": insights.get(
            "recommendations",
            [],
        ),
    }


# =========================================================
# RESUME PROCESSING
# =========================================================

def process_resume(uploaded_file):
    """
    Complete resume-processing pipeline.

    Steps:
    1. Read uploaded PDF
    2. Extract text
    3. Load ML model
    4. Predict personality
    5. Analyze career information
    6. Validate results
    """

    if uploaded_file is None:
        raise ValueError("No resume was uploaded.")

    file_data = uploaded_file.getvalue()

    if not file_data:
        raise ValueError("The uploaded file is empty.")

    # -----------------------------------------------------
    # Extract PDF text
    # -----------------------------------------------------

    text = extract_resume_text(file_data)

    if not validate_resume_text(text):
        raise ValueError(
            "The PDF does not contain readable text. "
            "Please upload a text-based resume PDF."
        )

    # -----------------------------------------------------
    # Load predictor
    # -----------------------------------------------------

    predictor = load_predictor()

    # -----------------------------------------------------
    # Personality prediction
    # -----------------------------------------------------

    scores = predictor.predict(text)

    scores = normalize_scores(scores)

    # -----------------------------------------------------
    # Career analysis
    # -----------------------------------------------------

    insights = analyze_resume(text)

    insights = normalize_insights(insights)

    return {
        "text": text,
        "scores": scores,
        "insights": insights,
    }


# =========================================================
# DISPLAY HELPERS
# =========================================================

def display_personality(scores):
    """Display personality scores."""

    st.subheader("Personality Snapshot")

    st.caption(
        "Experimental estimates inferred from resume language. "
        "These are not psychological assessments or hiring decisions."
    )

    traits = list(scores.items())

    columns = st.columns(len(traits))

    for column, (trait, score) in zip(columns, traits):

        label = TRAIT_LABELS.get(
            trait,
            trait.replace("_", " ").title(),
        )

        column.metric(
            label,
            f"{score:.0f}%",
        )

    st.divider()

    for trait, score in traits:

        label = TRAIT_LABELS.get(
            trait,
            trait.replace("_", " ").title(),
        )

        st.write(f"**{label}**")

        st.progress(
            int(score),
            text=f"{score:.0f}%",
        )


def display_skill_section(title, skills):
    """Display a skill list."""

    st.subheader(title)

    if not skills:
        st.info("No skills detected.")
        return

    for skill in skills:
        st.write(f"• {skill}")


def display_resume_signals(insights):
    """Display extracted resume information."""

    st.subheader("Resume Signals")

    col1, col2 = st.columns(2)

    with col1:
        display_skill_section(
            "Technical Skills",
            insights["technical_skills"],
        )

    with col2:
        display_skill_section(
            "Soft Skills",
            insights["soft_skills"],
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Education")

        education = insights["education_keywords"]

        if education:
            for item in education:
                st.write(f"• {item}")
        else:
            st.write("No common education keywords detected.")

    with col2:
        st.write("### Experience")

        st.write(
            insights["experience_indicator"]
        )

    with col3:
        st.write("### Projects")

        projects = insights["project_keywords"]

        if projects:
            for item in projects:
                st.write(f"• {item}")
        else:
            st.write("No common project keywords detected.")


def display_recommendations(recommendations):
    """Display recommended career directions."""

    st.subheader("Career Directions")

    if not recommendations:

        st.info(
            "No career recommendations were generated."
        )

        return

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):

        if not isinstance(recommendation, dict):
            continue

        role = recommendation.get(
            "role",
            "Suggested Career",
        )

        missing_skills = recommendation.get(
            "missing_skills",
            [],
        )

        st.write(
            f"### {index}. {role}"
        )

        if missing_skills:

            st.write(
                "Suggested next skills:"
            )

            for skill in missing_skills:
                st.write(f"• {skill}")

        else:

            st.success(
                "No major starter skills missing."
            )

        st.divider()


def display_resume_text(text):
    """Display extracted resume text."""

    with st.expander(
        "View Extracted Resume Text"
    ):
        st.text(
            text[:10000]
        )


# =========================================================
# RESULTS
# =========================================================

def display_results(results):
    """Display all analysis results."""

    text = results["text"]
    scores = results["scores"]
    insights = results["insights"]

    display_personality(scores)

    st.divider()

    display_resume_signals(insights)

    st.divider()

    display_recommendations(
        insights["recommendations"]
    )

    st.divider()

    display_resume_text(text)


# =========================================================
# APPLICATION
# =========================================================

def main():

    st.title("ResumeMind")

    st.write(
        "AI Resume Personality & Career Analyzer"
    )

    st.write(
        "Upload a PDF resume to analyze its language, "
        "skills, experience, projects, and potential "
        "career directions."
    )

    st.info(
        "Privacy: the resume is processed by the running "
        "application and is not intentionally sent to an "
        "external AI service."
    )

    # -----------------------------------------------------
    # Upload
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
    )

    # -----------------------------------------------------
    # Stop until file exists
    # -----------------------------------------------------

    if uploaded_file is None:

        st.info(
            "Upload a PDF resume to begin."
        )

        return

    # -----------------------------------------------------
    # File information
    # -----------------------------------------------------

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    # -----------------------------------------------------
    # Analyze
    # -----------------------------------------------------

    analyze = st.button(
        "Analyze Resume",
        type="primary",
        use_container_width=True,
    )

    if not analyze:
        return

    # -----------------------------------------------------
    # Processing
    # -----------------------------------------------------

    with st.spinner(
        "Analyzing your resume..."
    ):

        try:

            results = process_resume(
                uploaded_file
            )

        except ResumeParseError as error:

            st.error(
                f"Resume parsing failed: {error}"
            )

            return

        except ModelUnavailableError as error:

            st.error(
                f"Personality model unavailable: {error}"
            )

            st.info(
                "Check that the required model files "
                "exist inside the models folder."
            )

            return

        except FileNotFoundError as error:

            st.error(
                f"Required file not found: {error}"
            )

            return

        except ValueError as error:

            st.error(
                str(error)
            )

            return

        except Exception as error:

            st.error(
                "An unexpected error occurred "
                "while analyzing the resume."
            )

            st.exception(error)

            return

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    st.success(
        "Resume analysis completed successfully."
    )

    display_results(results)


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":
    main()
