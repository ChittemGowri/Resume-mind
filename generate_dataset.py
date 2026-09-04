"""Create a reproducible synthetic dataset for the ResumeMind demonstration model.

The labels are deliberately generated from language templates and are not measures of
real people. They make the full local ML pipeline runnable without claiming a
scientifically validated personality-labelled resume dataset.
"""
from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd


TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]
TRAIT_PHRASES = {
    "openness": ["creative experimentation", "curious research", "innovative ideas", "learning new technologies", "design thinking"],
    "conscientiousness": ["organized planning", "attention to detail", "reliable delivery", "structured documentation", "quality assurance"],
    "extraversion": ["client presentations", "team workshops", "public speaking", "stakeholder engagement", "community events"],
    "agreeableness": ["cross-functional collaboration", "empathetic communication", "team support", "mentoring peers", "conflict resolution"],
    "emotional_stability": ["calm under pressure", "resilient problem solving", "steady decision making", "deadline management", "adaptable execution"],
}
TECH = ["Python", "SQL", "JavaScript", "React", "Node.js", "AWS", "Docker", "Git", "Machine Learning", "NLP", "TensorFlow", "Power BI", "Excel", "Java", "C++"]
ROLES = ["software engineer", "data analyst", "machine learning intern", "product associate", "business analyst", "web developer"]
EDUCATION = ["Bachelor of Technology", "Bachelor of Science", "Master of Computer Applications", "Computer Science degree"]


def make_dataset(rows: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    noise = np.random.default_rng(seed)
    records = []
    for index in range(rows):
        scores = {trait: int(rng.randint(35, 85)) for trait in TRAITS}
        phrases = []
        for trait in TRAITS:
            count = 1 if scores[trait] < 55 else 2 if scores[trait] < 72 else 3
            phrases.extend(rng.sample(TRAIT_PHRASES[trait], count))
        skills = rng.sample(TECH, rng.randint(4, 8))
        text = (
            f"{rng.choice(ROLES).title()} resume. {rng.choice(EDUCATION)}. "
            f"Skills: {', '.join(skills)}. Experience: delivered {rng.randint(1, 5)} projects. "
            f"Highlights: {'; '.join(phrases)}."
        )
        record = {"resume_text": text}
        for trait in TRAITS:
            record[trait] = int(np.clip(scores[trait] + noise.normal(0, 4), 20, 98))
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ResumeMind training data.")
    parser.add_argument("--rows", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic_resume_personality.csv")
    args = parser.parse_args()
    if args.rows < 50:
        raise ValueError("Use at least 50 rows for a meaningful demonstration split.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_dataset(args.rows, args.seed).to_csv(output, index=False)
    print(f"Created {args.rows} synthetic records at {output}")


if __name__ == "__main__":
    main()

