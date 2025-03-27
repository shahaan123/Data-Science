# utils.ipynb

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import json
import os

def train_models_from_config(df, target_col, config):
    selected_cols = config["preprocessing"]["select_columns"]
    X = df[selected_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained_models = {}
    model_data = {}

    for model_config in config["models"]:
        if not model_config["enabled"]:
            continue

        name = model_config["name"]
        params = model_config["params"]

        if name == "random_forest":
            model = RandomForestClassifier(**params)
        elif name == "lgbm":
            model = lgb.LGBMClassifier(**params)
        else:
            continue

        model.fit(X_train, y_train)
        trained_models[name] = model
        model_data[name] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test
        }

    return trained_models, model_data


def evaluate_model(model, X_test, y_test, cv_data=None):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    results = {
        "accuracy": accuracy,
        "report": report
    }

    if cv_data:
        scores = cross_val_score(model, *cv_data, cv=5)
        results["cross_val"] = {
            "mean_score": scores.mean(),
            "scores": scores.tolist()
        }

    return results


def log_metrics(metrics, model_name, output_dir="metrics"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
