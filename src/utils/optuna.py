import json
from hashlib import md5

import optuna
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)
from optuna.storages import RDBStorage
from optuna.trial import Trial
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.pipeline import Pipeline

from src.utils.model import log_transform, exp_transform


def _parse_params(params: dict, trial: Trial) -> dict:

    parsed_params = {}

    for key, value in params.items():
        dist_type = value.__class__
        dist_params = value.__dict__

        dist_params["name"] = key

        if dist_type is IntDistribution:
            suggestion = trial.suggest_int
        elif dist_type is FloatDistribution:
            suggestion = trial.suggest_float
        elif dist_type is CategoricalDistribution:
            suggestion = trial.suggest_categorical
        else:
            raise ValueError(f"Unknown parameter distribution type: {dist_type}")

        parsed_params[key] = suggestion(**dist_params)

    return parsed_params


def optimize_tpe(
    cfg,
    model,
    params,
    data,
    n_trials: int | None = None,
    timeout: int | None = None,
    n_jobs: int | None = None,
):

    params_str = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    params_hash = md5(params_str).hexdigest()

    study_name = f"{model.__class__.__name__}-{params_hash}"
    storage = RDBStorage("sqlite:///optuna.db")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

    X = data[cfg.features]
    y = data[cfg.target]

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

    def objective(trial):

        parsed_params = _parse_params(params, trial)
        model.set_params(**parsed_params)

        wrapped_model = TransformedTargetRegressor(
            regressor=model,
            func=log_transform,
            inverse_func=exp_transform,
        )

        pipeline = Pipeline(
            steps=[
                ("transformer", transformer),
                ("model", wrapped_model),
            ]
        )

        scores = cross_val_score(pipeline, X, y, cv=3, scoring="r2")

        return scores.mean()

    params_str = json.dumps(params, indent=2, default=str)
    print(params_str)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    trial = study.best_trial

    result = {
        "number": trial.number,
        "score": trial.value,
        "params": trial.params,
    }

    result_str = json.dumps(result, indent=2, default=str)
    print(result_str)


def optimize_grid(
    cfg, model, params, data, n_jobs: int | None = None, verbose: int | None = None
):

    X = data[cfg.features]
    y = data[cfg.target]

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

    wrapped_model = TransformedTargetRegressor(
        regressor=model,
        func=log_transform,
        inverse_func=exp_transform,
    )

    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("model", wrapped_model),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="r2",
        cv=3,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    grid_search.fit(X, y)

    result = {
        "best_score": grid_search.best_score_,
        "best_params": grid_search.best_params_,
    }

    result_str = json.dumps(result, indent=2, default=str)
    print(result_str)
