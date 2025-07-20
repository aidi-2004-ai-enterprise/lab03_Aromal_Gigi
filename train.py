"""
Train an XGBoost classifier on the Seaborn penguins dataset,
evaluate its performance, and save both the model and metadata
for later use in the FastAPI application.
"""

import os
import json
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


def main() -> None:
    """
    Load penguins data, preprocess features/labels, train an XGBoost model,
    evaluate performance, and save both the model and metadata.
    """
    # --- Load & preprocess ---
    df: pd.DataFrame = sns.load_dataset("penguins")  # type: ignore
    df = df.dropna()
    y_raw: pd.Series = df["species"]
    X_raw: pd.DataFrame = df.drop(columns=["species"])

    # One-hot encode categorical features
    X: pd.DataFrame = pd.get_dummies(
        X_raw,
        columns=["sex", "island"],
        prefix=["sex", "island"]
    )

    # Label-encode target
    label_encoder: LabelEncoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Train model ---
    model: XGBClassifier = XGBClassifier(
        max_depth=3,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- Evaluate performance ---
    print("=== Training set performance ===")
    y_train_pred = model.predict(X_train)
    print(classification_report(y_train, y_train_pred))

    print("=== Test set performance ===")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred))

    # --- Save model & metadata ---
    os.makedirs("app/data", exist_ok=True)
    model.save_model("app/data/model.json")
    print("Model saved to app/data/model.json")

    metadata = {
        "feature_columns": X.columns.tolist(),
        "label_classes": label_encoder.classes_.tolist()
    }
    with open("app/data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to app/data/metadata.json")


if __name__ == "__main__":
    main()
