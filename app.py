"""ResumeMind - Modern AI Resume Personality & Career Analyzer."""

import json
import matplotlib.pyplot as plt
import streamlit as st

from career_analyzer import analyze_resume
from predictor import ModelUnavailableError, PersonalityPredictor
from resume_parser import ResumeParseError, extract_resume_text


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResumeMind | AI Resume Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM / CSS
# ============================================================

st.markdown(
    """
    <style>

    /* GLOBAL */
    .stApp {
        background: #f7f9fc;
        color: #172033;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e6ebf2;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.2rem 1rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.5rem 0.35rem 1.5rem 0.35rem;
        border-bottom: 1px solid #edf0f5;
        margin-bottom: 1.3rem;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(135deg, #dbeafe, #ede9fe);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        box-shadow: 0 5px 18px rgba(99, 102, 241, 0.12);
    }

    .brand-name {
        font-size: 1.05rem;
        font-weight: 800;
        color: #172033;
        line-height: 1.1;
    }

    .brand-subtitle {
        color: #718096;
        font-size: 0.68rem;
        margin-top: 3px;
    }

    .side-section {
        color: #98a2b3;
        font-size: 0.67rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.1rem 0 0.55rem 0.45rem;
    }

    .side-info {
        background: linear-gradient(135deg, #eff6ff, #f5f3ff);
        border: 1px solid #dce7fb;
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        margin-top: 1rem;
    }

    .side-info-icon {
        font-size: 28px;
        margin-bottom: 5px;
    }

    .side-info-title {
        font-weight: 800;
        color: #172033;
        font-size: 0.82rem;
    }

    .side-info-text {
        color: #667085;
        font-size: 0.7rem;
        line-height: 1.5;
        margin-top: 5px;
    }

    .side-footer {
        color: #98a2b3;
        font-size: 0.68rem;
        text-align: center;
        margin-top: 1.3rem;
    }

    /* TOP BAR */
    .topbar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.7rem;
    }

    .top-pill {
        border: 1px solid #e3e8f0;
        background: white;
        border-radius: 999px;
        padding: 0.5rem 0.8rem;
        color: #5d6b82;
        font-size: 0.8rem;
        box-shadow: 0 3px 12px rgba(31, 48, 78, 0.035);
    }

    /* HERO */
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid #dfe8f5;
        border-radius: 24px;
        padding: 2.1rem 2.2rem;
        margin-bottom: 1.2rem;
        background:
            radial-gradient(circle at 90% 20%, rgba(124, 58, 237, 0.09), transparent 30%),
            radial-gradient(circle at 60% 100%, rgba(37, 99, 235, 0.08), transparent 35%),
            linear-gradient(135deg, #f0f9ff, #ffffff 52%, #f5f3ff);
        box-shadow: 0 8px 30px rgba(40, 72, 120, 0.05);
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 0.38rem 0.75rem;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #cfe0fa;
        border-radius: 999px;
        color: #28539c;
        font-weight: 750;
        font-size: 0.72rem;
        margin-bottom: 0.85rem;
    }

    .hero-title {
        font-size: 2.65rem;
        line-height: 1.08;
        font-weight: 850;
        letter-spacing: -0.04em;
        color: #15264a;
        margin: 0;
        max-width: 720px;
    }

    .hero-description {
        color: #5d6b82;
        font-size: 0.98rem;
        line-height: 1.65;
        max-width: 650px;
        margin-top: 0.8rem;
    }

    .hero-features {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 1.3rem;
    }

    .hero-feature {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #e7edf6;
        border-radius: 999px;
        padding: 0.52rem 0.8rem;
        font-size: 0.72rem;
        color: #43516a;
    }

    .hero-feature-icon {
        font-size: 16px;
    }

    .hero-art {
        position: absolute;
        right: 3%;
        top: 16%;
        font-size: 6.5rem;
        opacity: 0.9;
    }

    /* SECTION HEADERS */
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1.65rem 0 0.8rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #172033;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-subtitle {
        color: #8994a6;
        font-size: 0.73rem;
    }

    /* UPLOAD CARD */
    .upload-card {
        background: white;
        border: 1.5px dashed #c9d8ee;
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 6px 22px rgba(31, 48, 78, 0.035);
        margin-bottom: 0.9rem;
    }

    .upload-title {
        font-weight: 800;
        font-size: 1rem;
        color: #172033;
    }

    .upload-subtitle {
        color: #718096;
        font-size: 0.76rem;
        margin-top: 4px;
    }

    .upload-icon {
        width: 48px;
        height: 48px;
        background: #edf5ff;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    /* BUTTONS */
    .stButton > button {
        border-radius: 12px !important;
        min-height: 2.8rem;
        font-weight: 750 !important;
        border: none !important;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 18px rgba(37, 99, 235, 0.18);
    }

    /* PERSONALITY METRICS */
    .trait-card {
        background: white;
        border: 1px solid #e5eaf2;
        border-radius: 16px;
        padding: 1rem;
        min-height: 142px;
        box-shadow: 0 5px 18px rgba(31, 48, 78, 0.035);
        position: relative;
        overflow: hidden;
    }

    .trait-card.blue { border-top: 3px solid #3b82f6; }
    .trait-card.green { border-top: 3px solid #22c55e; }
    .trait-card.orange { border-top: 3px solid #f59e0b; }
    .trait-card.pink { border-top: 3px solid #ec4899; }
    .trait-card.purple { border-top: 3px solid #8b5cf6; }

    .trait-icon {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.55rem;
        font-size: 18px;
    }

    .trait-name {
        color: #172033;
        font-weight: 700;
        font-size: 0.75rem;
    }

    .trait-score {
        color: #172033;
        font-size: 1.5rem;
        font-weight: 850;
        margin-top: 3px;
    }

    .progress-track {
        width: 100%;
        height: 6px;
        border-radius: 99px;
        background: #e9edf3;
        overflow: hidden;
        margin-top: 0.65rem;
    }

    .progress-fill {
        height: 100%;
        border-radius: 99px;
    }

    /* GENERAL CARDS */
    .info-card {
        background: white;
        border: 1px solid #e4e9f1;
        border-radius: 17px;
        padding: 1.05rem 1.1rem;
        min-height: 118px;
        box-shadow: 0 5px 18px rgba(31, 48, 78, 0.03);
    }

    .info-card-title {
        font-size: 0.78rem;
        font-weight: 800;
        color: #172033;
        margin-bottom: 0.7rem;
    }

    .info-card-text {
        font-size: 0.76rem;
        color: #65758b;
        line-height: 1.55;
    }

    /* SKILLS */
    .skills-card {
        background: white;
        border: 1px solid #e3e9f3;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        min-height: 145px;
        box-shadow: 0 5px 18px rgba(31, 48, 78, 0.03);
    }

    .skills-title {
        font-size: 0.82rem;
        font-weight: 800;
        color: #172033;
        margin-bottom: 0.65rem;
    }

    .skill-pill {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        margin: 0.16rem;
        font-size: 0.7rem;
        font-weight: 650;
    }

    .technical-pill {
        background: #edf4ff;
        color: #28539c;
        border: 1px solid #dbe8ff;
    }

    .soft-pill {
        background: #fff0f7;
        color: #b22b69;
        border: 1px solid #f9d9e9;
    }

    /* CAREER CARDS */
    .career-card {
        background: white;
        border: 1px solid #e3e9f2;
        border-radius: 18px;
        padding: 1.1rem;
        min-height: 205px;
        box-shadow: 0 5px 18px rgba(31, 48, 78, 0.035);
        margin-bottom: 0.8rem;
    }

    .career-role {
        font-size: 0.94rem;
        font-weight: 800;
        color: #172033;
    }

    .match-badge {
        float: right;
        background: #eaf8ef;
        color: #18864b;
        border: 1px solid #d3f0de;
        border-radius: 999px;
        padding: 0.25rem 0.55rem;
        font-size: 0.65rem;
        font-weight: 800;
    }

    .career-description {
        color: #68768a;
        font-size: 0.73rem;
        line-height: 1.5;
        margin: 0.65rem 0;
    }

    .career-label {
        font-size: 0.67rem;
        color: #475467;
        font-weight: 800;
        margin-top: 0.6rem;
    }

    .career-skill {
        display: inline-block;
        background: #f0f4fa;
        color: #526078;
        border-radius: 999px;
        padding: 0.25rem 0.52rem;
        font-size: 0.63rem;
        margin: 0.15rem;
    }

    /* PRIVACY CARD */
    .privacy-card {
        background: linear-gradient(135deg, #eff6ff, #f7f3ff);
        border: 1px solid #dce6fa;
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
        margin-top: 1rem;
    }

    .privacy-icon {
        font-size: 25px;
    }

    .privacy-title {
        font-weight: 800;
        font-size: 0.8rem;
        color: #172033;
        margin-top: 0.35rem;
    }

    .privacy-text {
        color: #667085;
        font-size: 0.68rem;
        line-height: 1.5;
        margin-top: 0.25rem;
    }

    /* FOOTER */
    .app-footer {
        margin-top: 2rem;
        border-radius: 16px;
        padding: 0.85rem;
        text-align: center;
        background: linear-gradient(90deg, #f8efff, #eef6ff, #fff2f8);
        color: #68768a;
        font-size: 0.73rem;
        border: 1px solid #ebe5f3;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: transparent;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #fbfcff !important;
        border: 1px dashed #d5deec !important;
        border-radius: 14px !important;
    }

    /* EXPANDER */
    .streamlit-expanderHeader {
        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TRAIT LABELS & METADATA
# ============================================================

TRAIT_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "emotional_stability": "Emotional Stability",
}

TRAIT_META = {
    "openness": {"icon": "💡", "class": "blue"},
    "conscientiousness": {"icon": "🎯", "class": "green"},
    "extraversion": {"icon": "👥", "class": "orange"},
    "agreeableness": {"icon": "💗", "class": "pink"},
    "emotional_stability": {"icon": "🛡️", "class": "purple"},
}


# ============================================================
# MODEL CACHING
# ============================================================

@st.cache_resource
def load_predictor() -> PersonalityPredictor:
    """Load the personality model once."""
    return PersonalityPredictor("models")


# ============================================================
# HELPERS
# ============================================================

def safe_list(value):
    """Return a safe list."""
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [str(value)]


def render_skill_pills(items, pill_class="technical-pill"):
    """Render skills as rounded pills."""
    items = safe_list(items)
    if not items:
        return "<span style='color:#8994a6;font-size:.72rem;'>No skills detected</span>"

    return "".join(f"<span class='skill-pill {pill_class}'>{item}</span>" for item in items)


def calculate_match(recommendation, index=0):
    """Generate a visual match score with fallback calculation."""
    possible_keys = ["match_score", "match", "score", "fit_score"]
    for key in possible_keys:
        if key in recommendation:
            try:
                return float(recommendation[key])
            except (ValueError, TypeError):
                pass

    # Dynamic fallback scaling to prevent static duplication past index 4
    return max(50.0, round(92.0 - (index * 4.5), 1))


# ============================================================
# PERSONALITY CHART
# ============================================================

def personality_chart(scores):
    """Create a clean personality chart."""
    labels = []
    values = []

    for key, value in scores.items():
        labels.append(TRAIT_LABELS.get(key, key.replace("_", " ").title()))
        try:
            values.append(float(value))
        except (ValueError, TypeError):
            values.append(0)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    chart_colors = ["#3B82F6", "#22C55E", "#F59E0B", "#EC4899", "#8B5CF6"]

    bars = ax.barh(
        labels,
        values,
        color=chart_colors[: len(values)],
        height=0.55,
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Experimental estimated score (%)", fontsize=9, color="#64748B")
    ax.tick_params(axis="both", labelsize=8.5, colors="#475467")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.grid(axis="x", alpha=0.16, linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        x_position = min(value + 1.5, 94)
        ax.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#344054",
        )

    fig.patch.set_alpha(0)
    ax.set_facecolor("white")
    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-icon">🧠</div>
                <div>
                    <div class="brand-name">ResumeMind</div>
                    <div class="brand-subtitle">AI Career Analyzer</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='side-section'>Workspace</div>", unsafe_allow_html=True)

        if st.button("🏠  Dashboard", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if st.button("↻  Analysis History", use_container_width=True):
            st.info("Analysis history can be added here using Streamlit session state or a database.")

        st.markdown("<div class='side-section'>Resources</div>", unsafe_allow_html=True)

        if st.button("💡  Tips & Guide", use_container_width=True):
            st.info("Use a text-based PDF with clear sections such as Skills, Education, Experience and Projects.")

        if st.button("ⓘ  About ResumeMind", use_container_width=True):
            st.info("ResumeMind uses local classical machine learning to analyze resume language.")

        st.markdown(
            """
            <div class="side-info">
                <div class="side-info-icon">🔒</div>
                <div class="side-info-title">Privacy First</div>
                <div class="side-info-text">
                    Your resume is processed by the running application and is not intentionally sent to an external AI service.
                </div>
            </div>
            <div class="side-footer">ResumeMind • Local ML • v1.0.0</div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HERO
# ============================================================

def render_hero():
    st.markdown(
        """
        <div class="topbar">
            <span class="top-pill">☀️ Light</span>
            <span class="top-pill">🔐 Private</span>
        </div>
        <div class="hero">
            <div class="hero-eyebrow">✨ Local Classical Machine Learning</div>
            <div class="hero-title">AI-Powered Resume<br>Personality & Career Analyzer</div>
            <div class="hero-description">
                Upload your resume and discover personality insights, valuable skills, resume signals,
                and career directions — all through transparent local machine learning.
            </div>
            <div class="hero-features">
                <div class="hero-feature">
                    <span class="hero-feature-icon">🧠</span>
                    <span><b>Personality Insights</b><br>5 Big Traits</span>
                </div>
                <div class="hero-feature">
                    <span class="hero-feature-icon">🧩</span>
                    <span><b>Skills Detection</b><br>Technical & Soft</span>
                </div>
                <div class="hero-feature">
                    <span class="hero-feature-icon">🚀</span>
                    <span><b>Career Guidance</b><br>Smart Recommendations</span>
                </div>
            </div>
            <div class="hero-art">📄🔍</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

def render_upload():
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">📄 Analyze your resume</div>
            <div class="section-subtitle">PDF • Text-based resume recommended</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-card">
            <div style="display:flex;align-items:center;gap:14px;">
                <div class="upload-icon">☁️</div>
                <div>
                    <div class="upload-title">Upload your resume</div>
                    <div class="upload-subtitle">Choose a PDF resume to begin your analysis.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload = st.file_uploader(
        "Choose PDF file",
        type=["pdf"],
        help="Use a text-based PDF rather than a scanned image.",
        label_visibility="collapsed",
        key="resume_uploader",
    )

    if upload:
        st.success(f"✓ {upload.name} is ready for analysis")

    analyze = st.button(
        "✨  Analyze Resume",
        type="primary",
        disabled=upload is None,
        use_container_width=True,
    )

    st.markdown(
        """
        <div style="text-align:center;color:#8994a6;font-size:.68rem;margin-top:.35rem;">
            🔒 Your resume stays within the running application.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return upload, analyze


# ============================================================
# PERSONALITY SECTION
# ============================================================

def render_personality(scores):
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">📊 Personality Snapshot</div>
            <div class="section-subtitle">Experimental estimates</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "These scores are inferred from resume language using classical machine learning. "
        "They are not a psychological assessment or hiring recommendation."
    )

    metric_columns = st.columns(5)
    trait_classes = ["blue", "green", "orange", "pink", "purple"]

    for index, (trait, value) in enumerate(scores.items()):
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            numeric_value = 0

        meta = TRAIT_META.get(
            trait,
            {"icon": "✨", "class": trait_classes[min(index, 4)]},
        )

        with metric_columns[index]:
            st.markdown(
                f"""
                <div class="trait-card {meta['class']}">
                    <div class="trait-icon" style="background:#f0f5ff;">{meta['icon']}</div>
                    <div class="trait-name">{TRAIT_LABELS.get(trait, trait.replace("_", " ").title())}</div>
                    <div class="trait-score">{numeric_value:.0f}%</div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width:{max(0, min(100, numeric_value))}%;background:linear-gradient(90deg, #3b82f6, #8b5cf6);"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    chart_left, chart_right = st.columns([2.1, 0.8])

    with chart_left:
        st.markdown(
            """
            <div class="info-card" style="padding-bottom:.3rem;">
                <div class="info-card-title">Personality Trait Distribution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = personality_chart(scores)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)  # Prevents memory leak

    with chart_right:
        st.markdown(
            """
            <div class="privacy-card">
                <div class="privacy-icon">ℹ️</div>
                <div class="privacy-title">About these scores</div>
                <div class="privacy-text">
                    Estimates are generated from resume language using local classical machine learning.
                    They should be interpreted as experimental signals.
                </div>
                <div style="font-size:2rem;margin-top:.6rem;">🧠</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RESUME SIGNALS
# ============================================================

def render_resume_signals(insights):
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">✨ Resume Signals</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    technical = insights.get("technical_skills", [])
    soft = insights.get("soft_skills", [])

    skill_left, skill_right = st.columns(2)

    with skill_left:
        st.markdown(
            f"""
            <div class="skills-card">
                <div class="skills-title">💻 Technical Skills</div>
                {render_skill_pills(technical, "technical-pill")}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with skill_right:
        st.markdown(
            f"""
            <div class="skills-card">
                <div class="skills-title">🤝 Soft Skills</div>
                {render_skill_pills(soft, "soft-pill")}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    education = insights.get("education_keywords", [])
    experience = insights.get("experience_indicator", "No experience indicator detected.")
    projects = insights.get("project_keywords", [])

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-title">🎓 Education</div>
                <div class="info-card-text">
                    {", ".join(safe_list(education)) if education else "No common education keyword found"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-title">💼 Experience</div>
                <div class="info-card-text">{experience}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-title">🚀 Projects</div>
                <div class="info-card-text">
                    {", ".join(safe_list(projects)) if projects else "No project keyword found"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CAREER DIRECTIONS
# ============================================================

def render_career_directions(insights):
    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">🎯 Career Directions</div>
            <div class="section-subtitle">Suggested based on your resume signals</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recommendations = insights.get("recommendations", [])

    if not recommendations:
        st.info("No career recommendations were generated.")
        return

    for start in range(0, len(recommendations), 3):
        row = recommendations[start : start + 3]
        columns = st.columns(3)

        for index, (column, recommendation) in enumerate(zip(columns, row)):
            role = recommendation.get("role", "Suggested Career")
            missing = safe_list(recommendation.get("missing_skills", []))
            description = recommendation.get(
                "description",
                "A career direction that aligns with the signals found in your resume.",
            )

            match = calculate_match(recommendation, start + index)

            if missing:
                skill_html = "".join(f'<span class="career-skill">{skill}</span>' for skill in missing)
            else:
                skill_html = '<span class="career-skill">No major skill gaps detected</span>'

            with column:
                st.markdown(
                    f"""
                    <div class="career-card">
                        <span class="match-badge">{match:.0f}% Match</span>
                        <div class="career-role">{role}</div>
                        <div class="career-description">{description}</div>
                        <div class="career-label">Suggested Next Skills</div>
                        <div style="margin-top:.25rem;">{skill_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def run_analysis(upload):
    progress = st.progress(0)
    status = st.empty()

    try:
        # STEP 1: Text extraction & validation
        status.info("📄 Reading your resume...")
        progress.progress(20)

        text = extract_resume_text(upload.getvalue())

        if not text or len(text.strip()) < 100:
            progress.empty()
            status.empty()
            st.error(
                "Insufficient or readable text found in this PDF (<100 characters). "
                "Please upload a standard text-based PDF instead of a scanned image."
            )
            return

        # STEP 2: Model setup
        status.info("🧠 Loading your personality model...")
        progress.progress(40)
        predictor = load_predictor()

        # STEP 3: Language prediction
        status.info("🔍 Analyzing resume language...")
        progress.progress(60)
        scores = predictor.predict(text)

        # STEP 4: Insights calculation
        status.info("🎯 Finding your career directions...")
        progress.progress(80)
        insights = analyze_resume(text, scores)

        progress.progress(100)
        status.success("✅ Analysis completed successfully!")
        progress.empty()

        # Render outputs
        render_personality(scores)
        render_resume_signals(insights)
        render_career_directions(insights)

        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

        # RESUME TEXT PREVIEW & EXPORT OPTIONS
        col_exp, col_dl = st.columns([3, 1])

        with col_exp:
            with st.expander("📄 Preview extracted resume text"):
                st.caption("First 5,000 characters extracted from your PDF.")
                st.text(text[:5000])

        with col_dl:
            export_payload = json.dumps(
                {
                    "personality_scores": scores,
                    "resume_signals": insights,
                },
                indent=2,
            )

            st.download_button(
                label="📥 Export Report (JSON)",
                data=export_payload,
                file_name="resumemind_analysis.json",
                mime="application/json",
                use_container_width=True,
            )

        # FOOTER
        st.markdown(
            """
            <div class="app-footer">
                💗 Keep learning, keep growing — your career journey starts with one step.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except ResumeParseError as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Could not read the resume: {exc}")

    except ModelUnavailableError as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Personality model unavailable: {exc}")
        st.info("Make sure the required model files are present inside the `models` folder.")

    except FileNotFoundError as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Required file not found: {exc}")

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error("❌ Something went wrong while analyzing the resume.")
        st.exception(exc)


# ============================================================
# MAIN
# ============================================================

def main():
    render_sidebar()
    render_hero()

    upload, analyze = render_upload()

    if analyze and upload:
        run_analysis(upload)
    elif upload is None:
        st.markdown(
            """
            <div style="text-align:center;padding:1.4rem 0 2rem;">
                <div style="font-size:2.4rem;margin-bottom:.35rem;">📄</div>
                <div style="font-weight:800;color:#344054;font-size:.95rem;">
                    Your resume insights will appear here
                </div>
                <div style="color:#98a2b3;font-size:.72rem;margin-top:.3rem;">
                    Upload a PDF above and click “Analyze Resume” to get started.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
