import json

import numpy as np
import pandas as pd
import joblib

from paths import DATA_DIR, MODEL_DIR
from logger import logger
from model.data import (
    get_raw_data,
    preprocess,
    enforce_constraints,
    split,
    fit_transform,
)
from model.model import (
    get_model,
    SUPPORTED_MODELS,
    compute_metrics,
    compute_importance_values,
    compute_dependence_values,
)


async def startup(cfg):

    np.random.seed(0)

    load_data(cfg)
    load_models(cfg)


def load_data(cfg):

    logger.debug("Started loading data")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data_train_path = DATA_DIR / "data-train.pkl"
    data_test_path = DATA_DIR / "data-test.pkl"

    correct_example_path = DATA_DIR / "correct-example.csv"
    incorrect_example_path = DATA_DIR / "incorrect-example.csv"

    cached = (
        data_train_path.exists()
        and data_test_path.exists()
        and correct_example_path.exists()
        and incorrect_example_path.exists()
    )

    if not cached:
        # if True:
        raw_data = get_raw_data(cfg)

        correct_example = raw_data.sample(100)
        correct_example = correct_example[
            correct_example["Face Area"].between(cfg.min_face_area, cfg.max_face_area)
        ]
        correct_example = correct_example[
            correct_example["Price"].between(cfg.min_price, cfg.max_price)
        ]

        correct_example.to_csv(DATA_DIR / "correct-example.csv", index=False)
        raw_data = raw_data.drop(index=correct_example.index)

        incorrect_example = raw_data.sample(100)
        incorrect_example = incorrect_example.drop(columns=["Movement"])

        incorrect_example.to_csv(DATA_DIR / "incorrect-example.csv", index=False)
        raw_data = raw_data.drop(index=incorrect_example.index)

        data = preprocess(cfg, raw_data)
        data = enforce_constraints(cfg, data)
        data_train, data_test = split(cfg, data)
        data_train, data_test = fit_transform(cfg, data_train, data_test)

        data_train.to_pickle(data_train_path)
        data_test.to_pickle(data_test_path)


def load_models(cfg):

    logger.debug("Started loading models")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data_train = pd.read_pickle(DATA_DIR / "data-train.pkl")
    data_test = pd.read_pickle(DATA_DIR / "data-test.pkl")

    X_train = data_train[cfg.features]
    y_train = data_train[cfg.target]

    X_test = data_test[cfg.features]
    y_test = data_test[cfg.target]

    for model_type in SUPPORTED_MODELS:

        model_dir = MODEL_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            model = get_model(model_type)
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
        else:
            model = joblib.load(model_path)

        metrics_path = model_dir / "metrics.csv"
        if not metrics_path.exists():
            metrics = [
                {"data_split": "train", **compute_metrics(model, X_train, y_train)},
                {"data_split": "test", **compute_metrics(model, X_test, y_test)},
            ]
            metrics = pd.DataFrame(metrics)
            metrics.to_csv(metrics_path, index=False)

        importance_values_path = model_dir / "importance_values.csv"
        if not importance_values_path.exists():
            importance_values = compute_importance_values(model, X_train, y_train)
            importance_values.to_csv(importance_values_path, index=False)

        dependence_values_path = model_dir / "dependence_values.json"
        if not dependence_values_path.exists():
            dependence_values = compute_dependence_values(model, X_train, y_train)
            with open(dependence_values_path, "w") as file:
                json.dump(
                    dependence_values,
                    file,
                    ensure_ascii=False,
                    indent=4,
                )
