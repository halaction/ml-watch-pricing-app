import os
import json

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence

from paths import DATA_DIR, MODEL_DIR


SUPPORTED_MODELS = [
    "linear_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
]

SUPPORTED_METRICS = [
    "r2_score",
    "mean_absolute_percentage_error",
    "median_absolute_error",
    "root_mean_squared_error",
]

MIN_PRICE = 0
MAX_PRICE = 1e6

MIN_LOG_PRICE = None
MAX_LOG_PRICE = np.log1p(MAX_PRICE)


def log_transform(array):
    return np.log1p(np.clip(array, MIN_PRICE, MAX_PRICE))


def exp_transform(array):
    return np.expm1(np.clip(array, MIN_LOG_PRICE, MAX_LOG_PRICE))


def get_model(model_type) -> Pipeline:

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Specified model type `{model_type}` is not supported.")

    if model_type == "linear_regression":
        model = get_linear_regression_model()

    if model_type == "decision_tree":
        model = get_decision_tree_model()

    if model_type == "gradient_boosting":
        model = get_gradient_boosting_model()

    if model_type == "random_forest":
        model = get_random_forest_model()

    return model


def get_linear_regression_model():

    # Combine transformers using ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    min_frequency=50,
                    max_categories=64,
                ),
                make_column_selector(dtype_include="category"),
            ),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )

    # Create and train the model
    regressor = Ridge(alpha=1, fit_intercept=True)

    model = TransformedTargetRegressor(
        regressor=regressor,
        func=log_transform,
        inverse_func=exp_transform,
    )

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            # ("feature_selection", SelectKBest(score_func=mutual_info_regression, k=50)),
            ("model", model),
        ]
    )

    return pipeline


def get_random_forest_model():

    # Combine transformers using ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    min_frequency=30,
                    max_categories=255,
                ),
                make_column_selector(dtype_include="category"),
            ),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )

    # Create and train the model
    regressor = RandomForestRegressor(
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_leaf=25,
        max_features=None,
        n_jobs=-1,
        random_state=0,
        warm_start=True,
    )

    # Wrap target transformation
    model = TransformedTargetRegressor(
        regressor=regressor,
        func=log_transform,
        inverse_func=exp_transform,
    )

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("model", model),
        ]
    )

    return pipeline


def get_decision_tree_model():

    # Combine transformers using ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            (
                "numerical",
                MinMaxScaler(),
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    min_frequency=50,
                    max_categories=128,
                ),
                make_column_selector(dtype_include="category"),
            ),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )

    # Create and train the model
    regressor = DecisionTreeRegressor(
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_leaf=10,
        max_features=None,
    )

    model = TransformedTargetRegressor(
        regressor=regressor,
        func=log_transform,
        inverse_func=exp_transform,
    )

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("model", model),
        ]
    )

    return pipeline


def get_gradient_boosting_model():

    # Create and train the model
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=None,
        max_depth=None,
        min_samples_leaf=25,
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


def compute_metrics(model, X, y):

    y_pred = model.predict(X)

    metrics = {
        "r2_score": r2_score(y, y_pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(y, y_pred),
        "median_absolute_error": median_absolute_error(y, y_pred),
        "root_mean_squared_error": root_mean_squared_error(y, y_pred),
    }

    return metrics


def compute_importance_values(model, X, y):

    outputs = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        max_samples=0.75,
        n_jobs=-1,
        random_state=0,
    )

    importance_values = pd.DataFrame(
        {
            "feature": model.feature_names_in_,
            "importance_mean": outputs.importances_mean,
            "importance_std": outputs.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)

    return importance_values


def compute_dependence_values(model, X, y):

    feature_names = model.feature_names_in_
    categorical_features = X.select_dtypes("category").columns

    dependence_values = {}
    for feature in feature_names:
        outputs = partial_dependence(
            model,
            X,
            features=[feature],
            feature_names=feature_names,
            categorical_features=categorical_features,
            percentiles=(0.05, 0.95),
            grid_resolution=32,
            method="auto",
            kind="average",
        )

        dependence_values[feature] = {
            "grid_values": outputs["grid_values"][0].tolist(),
            "average": outputs["average"].tolist()[0],
        }

    return dependence_values
