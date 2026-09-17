from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_synthetic_data import create_synthetic_dataset


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "railwaydata_synthetic.csv"
MODEL_PATH = PROJECT_DIR / "models" / "random_forest.joblib"
NUMERIC_FEATURES = [
    "asset_age", "previous_failures", "days_since_maintenance",
    "usage", "condition", "criticality",
]
FEATURES = ["asset_type"] + NUMERIC_FEATURES


def train_and_save_model() -> dict:
    """Create the demo data, train its model, and return evaluation details."""
    data = create_synthetic_dataset()
    data.to_csv(DATA_PATH, index=False)
    X = data[FEATURES]
    y = data["risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("asset_type", OneHotEncoder(handle_unknown="ignore"), ["asset_type"]),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return {
        "rows": len(data),
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions),
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    result = train_and_save_model()
    print(f"Synthetic dataset created: {DATA_PATH} ({result['rows']} rows)")
    print("Accuracy:", f"{result['accuracy']:.3f}")
    print("\nClassification Report:")
    print(result["report"])
    print(f"\nModel saved successfully: {result['model_path']}")
