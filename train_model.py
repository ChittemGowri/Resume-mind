"""Train one reproducible classical ML regressor for each Big Five trait."""
from pathlib import Path
import argparse
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from feature_extractor import fit_transform_texts
from generate_dataset import TRAITS


def train(dataset_path: str, model_dir: str, seed: int = 42) -> dict:
    data = pd.read_csv(dataset_path)
    required = {"resume_text", *TRAITS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    data = data.dropna(subset=["resume_text", *TRAITS])
    if len(data) < 50:
        raise ValueError("The dataset needs at least 50 complete rows.")

    vectorizer, features = fit_transform_texts(data["resume_text"])
    indices = list(range(len(data)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output / "tfidf_vectorizer.joblib")
    metrics = {}
    for trait in TRAITS:
        model = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        model.fit(features[train_idx], data.iloc[train_idx][trait])
        predictions = model.predict(features[test_idx])
        joblib.dump(model, output / f"{trait}_model.joblib")
        metrics[trait] = {
            "mae": round(float(mean_absolute_error(data.iloc[test_idx][trait], predictions)), 2),
            "r2": round(float(r2_score(data.iloc[test_idx][trait], predictions)), 3),
        }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResumeMind models.")
    parser.add_argument("--data", default="data/synthetic_resume_personality.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError("Training data not found. Run: python generate_dataset.py")
    metrics = train(args.data, args.model_dir, args.seed)
    print("Training complete. Held-out demonstration metrics:")
    for trait, values in metrics.items():
        print(f"  {trait}: MAE={values['mae']}, R²={values['r2']}")


if __name__ == "__main__":
    main()

