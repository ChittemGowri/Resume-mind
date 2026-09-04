"""Model loading and bounded personality score prediction."""
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from feature_extractor import clean_text
from generate_dataset import TRAITS


class ModelUnavailableError(RuntimeError):
    pass


class PersonalityPredictor:
    def __init__(self, model_dir: str = "models"):
        directory = Path(model_dir)
        vectorizer_path = directory / "tfidf_vectorizer.joblib"
        paths = {trait: directory / f"{trait}_model.joblib" for trait in TRAITS}
        missing = [str(path.name) for path in [vectorizer_path, *paths.values()] if not path.exists()]
        if missing:
            raise ModelUnavailableError("Model files are unavailable. Run `python generate_dataset.py` and `python train_model.py` first.")
        self.vectorizer = joblib.load(vectorizer_path)
        self.models = {trait: joblib.load(path) for trait, path in paths.items()}

    def predict(self, text: str) -> Dict[str, float]:
        if len(clean_text(text).split()) < 5:
            raise ValueError("The resume does not contain enough readable text to analyze.")
        features = self.vectorizer.transform([text])
        return {
            trait: round(float(np.clip(model.predict(features)[0], 0, 100)), 1)
            for trait, model in self.models.items()
        }

