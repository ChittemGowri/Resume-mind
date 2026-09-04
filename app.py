"""ResumeMind Streamlit application."""
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from career_analyzer import analyze_resume
from predictor import ModelUnavailableError, PersonalityPredictor
from resume_parser import ResumeParseError, extract_resume_text

st.set_page_config(page_title="ResumeMind", page_icon="🧠", layout="wide")

st.markdown("""
<style>
  .stApp { background: #f7f9fc; color: #172033; }
  .hero { padding: 1.8rem 0 .8rem; }
  .hero h1 { font-size: 3rem; margin-bottom: .25rem; color: #132238; }
  .eyebrow { color: #4267a8; font-weight: 700; letter-spacing: .08em; font-size: .78rem; text-transform: uppercase; }
  .card { background: white; border: 1px solid #e5eaf2; border-radius: 16px; padding: 1.15rem 1.25rem; min-height: 126px; box-shadow: 0 4px 16px rgba(31,48,78,.04); }
  .skill { display: inline-block; background: #eaf1ff; color: #28539c; border-radius: 999px; padding: .28rem .64rem; margin: .16rem; font-size: .88rem; }
  .muted { color: #65758b; }
</style>
""", unsafe_allow_html=True)

TRAIT_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "emotional_stability": "Emotional Stability",
}


@st.cache_resource
def load_predictor() -> PersonalityPredictor:
    return PersonalityPredictor("models")


def pills(items, empty="None detected"):
    if not items:
        return f"<span class='muted'>{empty}</span>"
    return "".join(f"<span class='skill'>{item}</span>" for item in items)


def personality_chart(scores):
    labels = [TRAIT_LABELS[key] for key in scores]
    values = list(scores.values())
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = ["#4e79c7", "#628ed8", "#7ba5e7", "#94b8ef", "#b1ccf4"]
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Experimental estimated score (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 1.2, bar.get_y() + bar.get_height()/2, f"{value:.0f}%", va="center", fontsize=10)
    fig.tight_layout()
    return fig


def show_results(text, scores, insights):
    st.divider()
    st.subheader("Personality snapshot")
    st.caption("Experimental estimates inferred from resume language; they are not a psychological assessment or hiring recommendation.")
    metric_cols = st.columns(5)
    for column, (trait, value) in zip(metric_cols, scores.items()):
        column.metric(TRAIT_LABELS[trait], f"{value:.0f}%")
    st.pyplot(personality_chart(scores), use_container_width=True)

    st.subheader("Resume signals")
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='card'><b>Technical skills</b><br><br>" + pills(insights["technical_skills"], "No starter technical skills detected") + "</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'><b>Soft skills</b><br><br>" + pills(insights["soft_skills"], "No starter soft skills detected") + "</div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.info("**Education**\n\n" + (", ".join(insights["education_keywords"]) or "No common education keyword found"))
    b.info("**Experience**\n\n" + insights["experience_indicator"])
    c.info("**Projects**\n\n" + (", ".join(insights["project_keywords"]) or "No project keyword found"))

    st.subheader("Career directions")
    for recommendation in insights["recommendations"]:
        missing = ", ".join(recommendation["missing_skills"])
        st.markdown(f"<div class='card'><b>{recommendation['role']}</b><br><span class='muted'>Suggested next skills: {missing}</span></div>", unsafe_allow_html=True)

    with st.expander("Preview extracted resume text"):
        st.text(text[:5000])


def main():
    st.markdown("<div class='hero'><div class='eyebrow'>Local classical machine learning</div><h1>ResumeMind</h1><p class='muted'>AI Resume Personality & Career Analyzer — turn a PDF CV into transparent, practical signals.</p></div>", unsafe_allow_html=True)
    st.warning("Privacy note: uploaded resumes are processed in the running local app and are not sent to an external AI service.")
    upload = st.file_uploader("Upload your resume (PDF)", type=["pdf"], help="Use a text-based PDF, not a scanned image.")
    analyze = st.button("Analyze resume", type="primary", disabled=upload is None, use_container_width=True)

    if analyze and upload is not None:
        try:
            with st.spinner("Reading your resume and generating local insights…"):
                text = extract_resume_text(upload.getvalue())
                predictor = load_predictor()
                scores = predictor.predict(text)
                insights = analyze_resume(text, scores)
            show_results(text, scores, insights)
        except ResumeParseError as exc:
            st.error(str(exc))
        except ModelUnavailableError as exc:
            st.error(str(exc))
            st.code("python generate_dataset.py\npython train_model.py", language="bash")
        except Exception:
            st.error("Analysis could not be completed. Please try another valid text-based PDF.")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("How it works")
        st.write("PDF text is normalized, converted to TF-IDF features, and passed to five saved Random Forest regression models. Skill and role insights come from visible, rule-based dictionaries.")
    with right:
        st.subheader("Important limitations")
        st.write("The training data is synthetic and demonstration-only. Resume wording cannot reliably measure personality. Never use this tool as a diagnosis or as the sole basis for employment decisions.")


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    main()

