from typing import Literal
from pydantic import BaseModel, Field


ModelType = Literal[
    "linear_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
]

MetricType = Literal[
    "r2_score",
    "mean_absolute_percentage_error",
    "median_absolute_error",
    "root_mean_squared_error",
]

DataSplit = Literal[
    "train",
    "test",
    "inference",
]


class MessageResponse(BaseModel):
    message: str = Field(title="Message")


class SupportedModelsResponse(BaseModel):
    supported_models: list[ModelType]


class SupportedMetricsResponse(BaseModel):
    supported_metrics: list[MetricType]


class LoadModelRequest(BaseModel):
    model_type: ModelType = Field(title="Model Type")


class LoadDataRequest(BaseModel):
    data_split: DataSplit = Field(title="Data Split")


class Metrics(BaseModel):
    data_split: DataSplit
    r2_score: float
    mean_absolute_percentage_error: float
    median_absolute_error: float
    root_mean_squared_error: float


MetricsResponse = list[Metrics]


class ImportanceValues(BaseModel):
    feature: str
    importance_mean: float
    importance_std: float


ImportanceValuesResponse = list[ImportanceValues]


class DependenceValues(BaseModel):
    grid_values: list
    average: list


DependenceValuesResponse = dict[str, DependenceValues]


class ComputeMetricRequest(BaseModel):
    metric_name: str


class ComputeMetricResponse(BaseModel):
    metric_value: float
