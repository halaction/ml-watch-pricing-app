import os
import json

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from model.paths import CACHE_DIR


SUPPORTED_MODEL_TYPES = {
    # "linear_regression",
    # "decision_tree",
    "gradient_boosting",
}

MIN_PRICE = 0
MAX_PRICE = 1e6

MIN_LOG_PRICE = None
MAX_LOG_PRICE = np.log1p(MAX_PRICE)


def load_models(cfg):

    print("started loading models")

    data_train = pd.read_pickle(CACHE_DIR / "data-train.pkl")
    data_test = pd.read_pickle(CACHE_DIR / "data-test.pkl")

    X_train = data_train[cfg.features]
    y_train = data_train[cfg.target]

    X_test = data_test[cfg.features]
    y_test = data_test[cfg.target]

    print(X_train.shape, X_test.shape)

    for model_type in SUPPORTED_MODEL_TYPES:
        model = get_model(model_type)

        print(X_train.info())

        model.fit(X_train, y_train)

        metrics = {
            "train": compute_metrics(model, X_train, y_train),
            "test": compute_metrics(model, X_test, y_test),
        }

        print(json.dumps(metrics, indent=4))

        importances = {
            "train": compute_importances(model, X_train, y_train),
            "test": compute_importances(model, X_test, y_test),
        }

        # print(json.dumps(importances, indent=4))

        model_dir = CACHE_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_dir / "model.pkl")

        with open(model_dir / "metrics.json", "w") as file:
            json.dump(
                metrics,
                file,
                ensure_ascii=False,
                indent=4,
            )

        with open(model_dir / "importances.json", "w") as file:
            json.dump(
                importances,
                file,
                ensure_ascii=False,
                indent=4,
            )

    print("finished loading models")


def get_model(model_type) -> Pipeline:

    if not model_type in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Specified model type `{model_type}` is not supported.")

    if model_type == "gradient_boosting":
        model = get_gradient_boosting_model()

    return model


def get_gradient_boosting_model():

    # Create and train the model
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=None,
        max_depth=None,
        min_samples_leaf=30,
        max_features=1.0,
        categorical_features="from_dtype",
        warm_start=True,
    )

    model = TransformedTargetRegressor(
        regressor=regressor,
        func=log_transform,
        inverse_func=exp_transform,
    )

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(
        steps=[
            ("model", model),
        ]
    )

    return pipeline


def get_linear_regression_model():

    # Combine transformers using ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            # (
            #     "categorical",
            #     OneHotEncoder(handle_unknown="ignore", drop="first", min_frequency=50),
            #     make_column_selector(dtype_include="category"),
            # ),
            (
                "numerical",
                MinMaxScaler(),
                make_column_selector(dtype_include=np.float64),
            ),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )

    # Create and train the model
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=None,
        max_depth=None,
        min_samples_leaf=30,
        max_features=1.0,
        categorical_features="from_dtype",
        warm_start=True,
    )

    model = TransformedTargetRegressor(
        regressor=regressor,
        func=log_transform,
        inverse_func=exp_transform,
    )

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(
        steps=[
            ("transformer", None),
            ("model", model),
        ]
    )

    return pipeline


def log_transform(array):
    return np.log1p(np.clip(array, MIN_PRICE, MAX_PRICE))


def exp_transform(array):
    return np.expm1(np.clip(array, MIN_LOG_PRICE, MAX_LOG_PRICE))


def compute_metrics(model, X, y):

    y_pred = model.predict(X)

    metrics = {
        "mean_squared_error": mean_squared_error(y, y_pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y, y_pred),
        "r2_score": r2_score(y, y_pred),
    }

    return metrics


def compute_importances(model, X, y):

    outputs = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        max_samples=0.75,
        n_jobs=-1,
        random_state=0,
    )

    importances_mean = outputs.importances_mean
    importances_std = outputs.importances_std

    importances = [
        {
            "feature": model.feature_names_in_[i],
            "importance_mean": importances_mean[i],
            "importance_std": importances_std[i],
        }
        for i in range(len(model.feature_names_in_))
    ]

    importances = sorted(importances, key=lambda item: -item["importance_mean"])

    return importances
