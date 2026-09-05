"""ResumeMind Streamlit application."""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from career_analyzer import analyze_resume
from predictor import ModelUnavailableError, PersonalityPredictor
from resume_parser import ResumeParseError, extract_resume_text


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ResumeMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>

    /* -----------------------------
       Global
    ------------------------------ */

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(221, 232, 255, 0.45), transparent 35%),
            radial-gradient(circle at top right, rgba(231, 222, 255, 0.40), transparent 30%),
            #f8fafc;
        color: #1f2937;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                     BlinkMacSystemFont, "Segoe UI", sans-serif;
    }


    /* -----------------------------
       Hero
    ------------------------------ */

    .hero-wrapper {
        padding: 1.5rem 0 1rem 0;
    }

    .eyebrow {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: #eef4ff;
        color: #4169a8;
        border: 1px solid #dbe7ff;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3.6rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.045em;
        margin: 0;
        background: linear-gradient(
            90deg,
            #263f70 0%,
            #5477c4 45%,
            #866cc9 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        line-height: 1.7;
        color: #64748b;
        max-width: 700px;
        margin-top: 1rem;
    }


    /* -----------------------------
       Privacy Banner
    ------------------------------ */

    .privacy-banner {
        margin: 1.2rem 0 1.8rem 0;
        padding: 0.95rem 1.1rem;
        border-radius: 14px;
        background: linear-gradient(90deg, #f2f7ff, #f7f4ff);
        border: 1px solid #dce5f7;
        color: #536276;
        font-size: 0.92rem;
        box-shadow: 0 5px 20px rgba(62, 84, 126, 0.05);
    }

    .privacy-banner strong {
        color: #36527f;
    }


    /* -----------------------------
       Upload Area
    ------------------------------ */

    .upload-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #e3e9f3;
        border-radius: 22px;
        padding: 1.7rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 35px rgba(45, 66, 99, 0.07);
    }

    .upload-heading {
        font-size: 1.2rem;
        font-weight: 750;
        color: #24344d;
        margin-bottom: 0.35rem;
    }

    .upload-description {
        color: #738197;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }


    /* -----------------------------
       Buttons
    ------------------------------ */

    .stButton > button {
        border-radius: 14px;
        border: 0;
        min-height: 3rem;
        font-weight: 700;
        font-size: 0.96rem;
        box-shadow: 0 8px 20px rgba(79, 112, 173, 0.16);
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(79, 112, 173, 0.21);
    }


    /* -----------------------------
       Section Titles
    ------------------------------ */

    .section-header {
        margin-top: 2.2rem;
        margin-bottom: 0.3rem;
        font-size: 1.45rem;
        font-weight: 800;
        color: #253753;
        letter-spacing: -0.02em;
    }

    .section-description {
        color: #77859a;
        margin-bottom: 1.2rem;
        font-size: 0.92rem;
    }


    /* -----------------------------
       Cards
    ------------------------------ */

    .glass-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #e4eaf2;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 8px 25px rgba(42, 61, 92, 0.055);
        height: 100%;
    }

    .card-title {
        color: #334766;
        font-size: 1rem;
        font-weight: 750;
    }

    .card-description {
        color: #7b879a;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-top: 0.45rem;
    }


    /* -----------------------------
       Metric Cards
    ------------------------------ */

    .metric-card {
        background: linear-gradient(
            145deg,
            #ffffff,
            #f6f8fd
        );
        border: 1px solid #e2e8f2;
        border-radius: 18px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(46, 63, 91, 0.055);
    }

    .metric-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #7a879a;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        line-height: 1.25;
    }

    .metric-value {
        margin-top: 0.45rem;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4d6fb1, #846bc5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    /* -----------------------------
       Skill Pills
    ------------------------------ */

    .skills-container {
        line-height: 2.5;
    }

    .skill-pill {
        display: inline-block;
        padding: 0.35rem 0.72rem;
        margin: 0.18rem 0.15rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #edf4ff, #f1efff);
        color: #4b6094;
        border: 1px solid #dfe7f7;
        font-size: 0.83rem;
        font-weight: 650;
    }


    /* -----------------------------
       Info Cards
    ------------------------------ */

    .info-card {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 130px;
        box-shadow: 0 7px 20px rgba(40, 58, 88, 0.045);
    }

    .info-icon {
        font-size: 1.25rem;
        margin-bottom: 0.45rem;
    }

    .info-title {
        color: #344765;
        font-weight: 750;
        font-size: 0.95rem;
    }

    .info-text {
        color: #768397;
        line-height: 1.55;
        font-size: 0.86rem;
        margin-top: 0.35rem;
    }


    /* -----------------------------
       Career Recommendation
    ------------------------------ */

    .career-card {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #f7f8ff 100%
        );
        border: 1px solid #e2e7f2;
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 7px 22px rgba(48, 62, 91, 0.045);
    }

    .career-role {
        color: #33496d;
        font-weight: 800;
        font-size: 1.03rem;
    }

    .career-skills {
        color: #7a8799;
        font-size: 0.87rem;
        margin-top: 0.35rem;
    }


    /* -----------------------------
       Footer
    ------------------------------ */

    .footer-card {
        margin-top: 2.8rem;
        padding: 1.4rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #f5f8ff,
            #faf7ff
        );
        border: 1px solid #e1e7f2;
    }


    /* -----------------------------
       Streamlit tweaks
    ------------------------------ */

    div[data-testid="stFileUploader"] {
        background: #fbfcff;
        border-radius: 16px;
    }

    div[data-testid="stFileUploader"] section {
        border: 2px dashed #cfdaf0;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #fbfdff,
            #f8f7ff
        );
    }

    div[data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid #e2e8f1;
        background: #ffffff;
    }

    [data-testid="stMetric"] {
        background: transparent;
    }

    .stAlert {
        border-radius: 14px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================

TRAIT_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "emotional_stability": "Emotional Stability",
}


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_predictor() -> PersonalityPredictor:
    return PersonalityPredictor("models")


# =========================================================
# HELPERS
# =========================================================

def pills(items, empty="None detected"):
    if not items:
        return f"<span style='color:#8793a6;'>{empty}</span>"

    return (
        "<div class='skills-container'>"
        + "".join(
            f"<span class='skill-pill'>{item}</span>"
            for item in items
        )
        + "</div>"
    )


def personality_chart(scores):
    labels = [TRAIT_LABELS[key] for key in scores]
    values = list(scores.values())

    fig, ax = plt.subplots(figsize=(9, 4.2))

    bars = ax.barh(
        labels,
        values,
        height=0.58,
        color=[
            "#88A8E8",
            "#98B5ED",
            "#A9C0F0",
            "#B8C9F2",
            "#C6D3F5",
        ],
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel(
        "Estimated score (%)",
        fontsize=10,
        color="#7A8799",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        labelsize=10,
        colors="#53627A",
        length=0,
    )

    ax.tick_params(
        axis="x",
        labelsize=9,
        colors="#8793A6",
    )

    ax.grid(
        axis="x",
        alpha=0.18,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            min(value + 1.4, 94),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%",
            va="center",
            fontsize=10,
            fontweight="700",
            color="#4B5D7A",
        )

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    fig.tight_layout()

    return fig


# =========================================================
# RESULTS
# =========================================================

def show_results(text, scores, insights):

    # -----------------------------------------------------
    # Personality
    # -----------------------------------------------------

    st.markdown(
        "<div class='section-header'>🧠 Personality snapshot</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-description'>"
        "Experimental estimates inferred from resume language. "
        "These are not psychological assessments or hiring recommendations."
        "</div>",
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)

    for column, (trait, value) in zip(metric_cols, scores.items()):

        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    {TRAIT_LABELS[trait]}
                </div>
                <div class="metric-value">
                    {value:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    chart_col, summary_col = st.columns(
        [1.65, 1],
        gap="large",
    )

    with chart_col:

        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">
                    Personality profile
                </div>
                <div class="card-description">
                    A visual representation of the estimated trait scores
                    generated from the resume text.
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.pyplot(
            personality_chart(scores),
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with summary_col:

        highest_trait = max(scores, key=scores.get)
        lowest_trait = min(scores, key=scores.get)

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">
                    ✨ Key signals
                </div>

                <div style="margin-top:1rem;">
                    <div style="
                        color:#8994a6;
                        font-size:.78rem;
                        text-transform:uppercase;
                        font-weight:700;
                    ">
                        Strongest signal
                    </div>

                    <div style="
                        color:#465d87;
                        font-size:1.35rem;
                        font-weight:800;
                        margin-top:.25rem;
                    ">
                        {TRAIT_LABELS[highest_trait]}
                    </div>
                </div>

                <div style="
                    height:1px;
                    background:#edf0f5;
                    margin:1rem 0;
                "></div>

                <div>
                    <div style="
                        color:#8994a6;
                        font-size:.78rem;
                        text-transform:uppercase;
                        font-weight:700;
                    ">
                        Area to strengthen
                    </div>

                    <div style="
                        color:#596b87;
                        font-size:1.05rem;
                        font-weight:750;
                        margin-top:.25rem;
                    ">
                        {TRAIT_LABELS[lowest_trait]}
                    </div>
                </div>

                <div style="
                    margin-top:1.1rem;
                    padding:.8rem;
                    border-radius:12px;
                    background:#f7f8fc;
                    color:#7a879a;
                    font-size:.84rem;
                    line-height:1.55;
                ">
                    Resume language can provide signals, but it should
                    not be treated as a definitive measure of personality.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # Resume Signals
    # -----------------------------------------------------

    st.markdown(
        "<div class='section-header'>📌 Resume signals</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-description'>"
        "Visible signals extracted from your resume using transparent,
        rule-based analysis."
        "</div>",
        unsafe_allow_html=True,
    )

    skill_col1, skill_col2 = st.columns(2, gap="large")

    with skill_col1:

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">
                    💻 Technical skills
                </div>
                <div class="card-description">
                    Technologies and technical keywords detected in the resume.
                </div>

                <div style="margin-top:.75rem;">
                    {pills(
                        insights["technical_skills"],
                        "No starter technical skills detected"
                    )}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with skill_col2:

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">
                    🤝 Soft skills
                </div>
                <div class="card-description">
                    Communication, teamwork, leadership and related signals.
                </div>

                <div style="margin-top:.75rem;">
                    {pills(
                        insights["soft_skills"],
                        "No starter soft skills detected"
                    )}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # -----------------------------------------------------
    # Education / Experience / Projects
    # -----------------------------------------------------

    info1, info2, info3 = st.columns(3, gap="large")

    education = (
        ", ".join(insights["education_keywords"])
        or "No common education keyword found"
    )

    projects = (
        ", ".join(insights["project_keywords"])
        or "No project keyword found"
    )

    with info1:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-icon">🎓</div>
                <div class="info-title">Education</div>
                <div class="info-text">
                    {education}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info2:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-icon">💼</div>
                <div class="info-title">Experience</div>
                <div class="info-text">
                    {insights["experience_indicator"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info3:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-icon">🚀</div>
                <div class="info-title">Projects</div>
                <div class="info-text">
                    {projects}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # Career Directions
    # -----------------------------------------------------

    st.markdown(
        "<div class='section-header'>🎯 Career directions</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-description'>"
        "Potential roles based on detected resume skills and signals."
        "</div>",
        unsafe_allow_html=True,
    )

    recommendations = insights.get("recommendations", [])

    if recommendations:

        for index, recommendation in enumerate(recommendations, start=1):

            missing = (
                ", ".join(recommendation["missing_skills"])
                if recommendation["missing_skills"]
                else "Your resume already contains the main starter skills."
            )

            st.markdown(
                f"""
                <div class="career-card">

                    <div style="
                        display:flex;
                        align-items:center;
                        gap:.8rem;
                    ">

                        <div style="
                            width:34px;
                            height:34px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            border-radius:10px;
                            background:#eef3ff;
                            color:#5872ac;
                            font-weight:800;
                        ">
                            {index}
                        </div>

                        <div class="career-role">
                            {recommendation["role"]}
                        </div>

                    </div>

                    <div class="career-skills">
                        <strong>Suggested next skills:</strong> {missing}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.info(
            "No career directions were generated from the current resume signals."
        )

    # -----------------------------------------------------
    # Resume Preview
    # -----------------------------------------------------

    with st.expander("📄 Preview extracted resume text"):

        st.caption(
            "This is the text extracted from your uploaded PDF."
        )

        st.text(text[:5000])


# =========================================================
# MAIN APP
# =========================================================

def main():

    # -----------------------------------------------------
    # Hero
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="hero-wrapper">

            <div class="eyebrow">
                LOCAL MACHINE LEARNING • RESUME INTELLIGENCE
            </div>

            <h1 class="hero-title">
                ResumeMind
            </h1>

            <div class="hero-subtitle">
                Turn a PDF resume into practical career signals,
                personality estimates, skill insights and role directions —
                all processed locally.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Privacy
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="privacy-banner">
            🔐 <strong>Privacy first:</strong>
            Your resume is processed inside the running local application
            and is not sent to an external AI service.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Upload card
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="upload-card">

            <div class="upload-heading">
                📎 Upload your resume
            </div>

            <div class="upload-description">
                Upload a text-based PDF resume to begin the analysis.
                Scanned image-only PDFs may not work correctly.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    upload = st.file_uploader(
        "Resume PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Use a text-based PDF rather than a scanned image.",
    )

    st.write("")

    analyze = st.button(
        "✨ Analyze my resume",
        type="primary",
        disabled=upload is None,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # Analysis
    # -----------------------------------------------------

    if analyze and upload is not None:

        try:

            with st.spinner(
                "Reading your resume and generating local insights..."
            ):

                text = extract_resume_text(
                    upload.getvalue()
                )

                predictor = load_predictor()

                scores = predictor.predict(text)

                insights = analyze_resume(
                    text,
                    scores,
                )

            st.success(
                "Analysis completed successfully."
            )

            show_results(
                text,
                scores,
                insights,
            )

        except ResumeParseError as exc:

            st.error(
                f"Resume parsing failed: {exc}"
            )

        except ModelUnavailableError as exc:

            st.error(str(exc))

            st.markdown(
                """
                <div class="glass-card">
                    <div class="card-title">
                        ⚙️ Model setup required
                    </div>

                    <div class="card-description">
                        The trained model files are not available yet.
                        Generate the dataset and train the model first.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.code(
                "python generate_dataset.py\n"
                "python train_model.py",
                language="bash",
            )

        except Exception:

            st.error(
                "Analysis could not be completed. "
                "Please upload another valid text-based PDF."
            )

    # -----------------------------------------------------
    # Bottom information
    # -----------------------------------------------------

    st.markdown(
        "<div class='section-header'>⚡ Behind ResumeMind</div>",
        unsafe_allow_html=True,
    )

    how_col, limit_col = st.columns(
        2,
        gap="large",
    )

    with how_col:

        st.markdown(
            """
            <div class="glass-card">

                <div class="card-title">
                    🔍 How it works
                </div>

                <div class="card-description">

                    <b>1. Extract</b><br>
                    Text is extracted from the uploaded PDF.

                    <br><br>

                    <b>2. Transform</b><br>
                    Resume language is converted into TF-IDF features.

                    <br><br>

                    <b>3. Predict</b><br>
                    Five saved Random Forest regression models estimate
                    personality-related signals.

                    <br><br>

                    <b>4. Analyze</b><br>
                    Skills, projects, education and career directions
                    are generated using visible rule-based dictionaries.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with limit_col:

        st.markdown(
            """
            <div class="glass-card">

                <div class="card-title">
                    ⚠️ Important limitations
                </div>

                <div class="card-description">

                    The personality model uses synthetic,
                    demonstration-only training data.

                    <br><br>

                    Resume wording cannot reliably measure a person's
                    real psychological characteristics.

                    <br><br>

                    These outputs should never be used as a diagnosis,
                    hiring filter, or sole employment decision.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # Footer
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="footer-card">

            <div style="
                color:#4d6287;
                font-weight:800;
                font-size:1rem;
            ">
                ResumeMind 🧠
            </div>

            <div style="
                color:#7d899c;
                font-size:.84rem;
                margin-top:.35rem;
                line-height:1.55;
            ">
                Local-first resume intelligence for learning,
                experimentation and career exploration.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    main()
