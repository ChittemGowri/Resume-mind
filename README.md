# ResumeMind — AI Resume Personality & Career Analyzer

ResumeMind is a local Streamlit portfolio project for **Task 2: Personality Prediction System Through CV Analysis**. Upload a text-based PDF resume to extract its text, estimate five Big Five-style trait scores, and surface transparent skill and career insights.

> **Ethical notice:** Results are experimental estimates based on resume language. The included labels are synthetic, so this project is neither a psychological instrument nor a valid hiring decision tool.

## Features

- PDF text extraction with useful errors for invalid, empty, scanned, and password-protected files
- TF-IDF text features and five independently trained `RandomForestRegressor` models
- Scores for Openness, Conscientiousness, Extraversion, Agreeableness, and Emotional Stability
- Rule-based technical/soft-skill detection, education/project signals, role suggestions, and skill gaps
- Polished, light Streamlit interface; no paid APIs, LLMs, or external services
- Reproducible synthetic-data pipeline and held-out demonstration metrics

## Architecture

```text
PDF resume → pdfplumber extraction → cleaning → TF-IDF vectorizer
                                              ├→ five Random Forest regressors → trait scores
                                              └→ rule dictionaries → skills, role fit, gaps
```

## Project structure

```text
ResumeMind/
├── app.py                  # Streamlit interface
├── generate_dataset.py     # deterministic synthetic data creator
├── train_model.py          # trains/evaluates/saves artifacts
├── predictor.py            # model loading and inference
├── resume_parser.py        # robust PDF extraction
├── feature_extractor.py    # text cleaning and TF-IDF
├── career_analyzer.py      # explainable rule-based insights
├── data/                   # generated CSV (ignored)
├── models/                 # generated joblib artifacts (ignored)
└── sample_resume/          # optional local test PDFs
```

## Setup and run

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

The first two commands build the local demonstration artifacts in `data/` and `models/`. Run them once before launching the app. To reproduce a new dataset, use `python generate_dataset.py --rows 1000 --seed 42`.

## ML methodology

The generator composes fictional resume snippets from role, skills, and trait-language templates, then adds bounded random variation to the labels. `TfidfVectorizer` learns unigram/bigram text importance. One seeded `RandomForestRegressor` is trained per trait, and the same saved vectorizer is used at inference. The trainer reserves 20% of the synthetic rows and writes MAE/R² metrics to `models/metrics.json`.

This deliberate synthetic design makes the pipeline runnable and honest: the metrics indicate template-learning performance only, not real-world personality validity.

## Screenshots

Add screenshots of the upload page and completed analysis here after running the app locally.

## Limitations and responsible use

- Scanned/image-only PDFs need OCR and are intentionally rejected.
- A resume is incomplete, strategic writing—not a reliable personality sample.
- Skills use starter dictionaries and may miss aliases or context.
- Do not use outputs for diagnosis, protected-class inference, or automated employment decisions.

## How I Explain This Project in an Interview

**Problem statement:** I built a local CV analyzer that demonstrates an end-to-end NLP/ML workflow while producing useful, explainable career signals.

**Data:** There is no credible personality-labelled CV dataset bundled with the project, so I created a reproducible, clearly disclosed synthetic dataset instead of claiming scientific validation.

**Preprocessing:** PDF text is extracted with `pdfplumber`, lowercased, whitespace-normalized, and cleaned while preserving technical tokens such as `C++` and `Node.js`.

**TF-IDF:** TF-IDF turns resume words and two-word phrases into sparse numerical features, weighting discriminative terms more strongly than common ones.

**Why classical ML:** TF-IDF plus Random Forest is local, low-cost, explainable, fast to train on a small demo dataset, and easy to maintain without an API.

**Model selection and training:** I train five independent seeded Random Forest regressors, one per Big Five-style target. The vectorizer and models are saved with joblib.

**Prediction:** The app loads the saved artifacts, transforms the uploaded text with the same vectorizer, bounds predictions to 0–100, and presents them as experimental estimates.

**Evaluation:** `train_model.py` uses an 80/20 held-out split and records MAE and R² in `models/metrics.json`; these are only synthetic demonstration metrics.

**Limitations:** The data and labels are synthetic, personality cannot be reliably inferred from a CV, and keyword rules miss semantic context.

**Future improvements:** With consented, validated research data, I would evaluate fairness and calibration, add OCR, broaden skill ontologies, add model cards, and perform human-in-the-loop usability testing.

