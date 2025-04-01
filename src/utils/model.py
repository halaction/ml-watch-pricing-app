import numpy as np
import pandas as pd
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance, partial_dependence


SUPPORTED_MODELS = [
    "linear_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
]

MIN_PRICE = 0
MAX_PRICE = 1e6

MIN_LOG_PRICE = None
MAX_LOG_PRICE = np.log1p(MAX_PRICE)


def log_transform(array):
    return np.log1p(np.clip(array, MIN_PRICE, MAX_PRICE))


def exp_transform(array):
    return np.expm1(np.clip(array, MIN_LOG_PRICE, MAX_LOG_PRICE))


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


def compute_importance_values(model, X, y):

    outputs = permutation_importance(
        model,
        X,
        y,
        n_repeats=3,
        max_samples=0.5,
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
            grid_resolution=16,
            method="auto",
            kind="average",
        )

        dependence_values[feature] = {
            "grid_values": outputs["grid_values"][0].tolist(),
            "average": outputs["average"].tolist()[0],
        }

    return dependence_values
