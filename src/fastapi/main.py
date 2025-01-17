import json
from contextlib import asynccontextmanager
from io import StringIO

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from http import HTTPStatus
import joblib
import pandas as pd
from sklearn.metrics import get_scorer

from schemas import (
    MessageResponse,
    SupportedModelsResponse,
    SupportedMetricsResponse,
    LoadModelRequest,
    LoadDataRequest,
    Metrics,
    MetricsResponse,
    ImportanceValuesResponse,
    DependenceValuesResponse,
    ComputeMetricRequest,
    ComputeMetricResponse,
)
from model.paths import DATA_DIR, MODEL_DIR
from model.config import get_config
from model.data import preprocess, check_constraints, transform
from model.model import compute_metrics, SUPPORTED_MODELS, SUPPORTED_METRICS
from model.startup import startup


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup events
    cfg = get_config()

    await startup(cfg)

    app.state.config = cfg

    app.state.model_type = None
    app.state.model = None

    app.state.data_split = None
    app.state.data = None

    yield

    # Shutdown events
    ...


app = FastAPI(lifespan=lifespan)


@app.get(
    "/get_status",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
)
async def get_status() -> MessageResponse:
    return {"message": "We are up!"}


@app.get(
    "/get_supported_models",
    status_code=HTTPStatus.OK,
    response_model=SupportedModelsResponse,
)
async def get_supported_models() -> SupportedModelsResponse:
    return {"supported_models": SUPPORTED_MODELS}


@app.get(
    "/get_supported_metrics",
    status_code=HTTPStatus.OK,
    response_model=SupportedMetricsResponse,
)
async def get_supported_metrics() -> SupportedMetricsResponse:
    return {"supported_metrics": SUPPORTED_METRICS}


@app.post(
    "/load_model",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
)
async def load_model(request: LoadModelRequest) -> MessageResponse:

    model_type = request.model_type

    model_path = MODEL_DIR / model_type / "model.pkl"
    if not model_path.exists():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Model is not cached",
        )

    model = joblib.load(model_path)

    app.state.model_type = model_type
    app.state.model = model

    return {"message": f"Model {model_type} is successfully loaded!"}


@app.post(
    "/unload_model",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
)
async def unload_model() -> MessageResponse:

    model_type = app.state.model_type
    model = app.state.model

    if model is None:
        return {"message": "No model is currently loaded."}

    app.state.model_type = None
    app.state.model = None

    return {"message": f"Model {model_type} is unloaded."}


@app.get(
    "/get_model_status",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
)
async def get_model_status() -> MessageResponse:

    model_type = app.state.model_type
    model = app.state.model

    if model is None:
        return {"message": "No model is currently loaded."}

    return {"message": f"Model {model_type} is currently loaded."}


@app.get(
    "/get_metrics",
    status_code=HTTPStatus.OK,
    response_model=MetricsResponse,
)
async def get_metrics() -> MetricsResponse:

    model_type = app.state.model_type

    if model_type is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No model is currently loaded.",
        )

    metrics_path = MODEL_DIR / model_type / "metrics.csv"
    metrics = pd.read_csv(metrics_path)

    return metrics.to_dict(orient="records")


@app.get(
    "/get_importance_values",
    status_code=HTTPStatus.OK,
    response_model=ImportanceValuesResponse,
    description="...",
)
async def get_importance_values() -> ImportanceValuesResponse:

    model_type = app.state.model_type
    if model_type is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No model is currently loaded.",
        )

    importance_values_path = MODEL_DIR / model_type / "importance_values.csv"
    importance_values = pd.read_csv(importance_values_path)

    return importance_values.to_dict(orient="records")


@app.get(
    "/get_dependence_values",
    status_code=HTTPStatus.OK,
    response_model=DependenceValuesResponse,
    description="...",
)
async def get_dependence_values() -> DependenceValuesResponse:

    model_type = app.state.model_type
    if model_type is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No model is currently loaded.",
        )

    dependence_values_path = MODEL_DIR / model_type / "dependence_values.json"
    with open(dependence_values_path, "r") as file:
        dependence_values = json.load(file)

    return dependence_values


@app.get(
    "/get_dtypes",
    status_code=HTTPStatus.OK,
    response_model=dict[str, str],
    description="...",
)
async def get_dtypes() -> dict[str, str]:

    data = app.state.data
    if data is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No data is currently loaded.",
        )

    dtypes = data.dtypes.to_dict()
    dtypes = {feature: dtype.name for feature, dtype in dtypes.items()}

    return dtypes


@app.post(
    "/upload_data",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
    description="Upload inference data.",
)
async def upload_data(file: UploadFile) -> MessageResponse:

    if file.content_type != "text/csv":
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"File type of {file.content_type} is not supported. Provide text/csv file.",
        )

    content = await file.read()

    try:
        raw_data = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Encountered error while reading CSV file: {str(e)}",
        )

    cfg = app.state.config

    try:
        data = preprocess(cfg, raw_data)
        check_constraints(cfg, data)
        data = transform(cfg, data)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Encountered error while processing data: {str(e)}",
        )

    data_path = DATA_DIR / "data-inference.pkl"
    data.to_pickle(data_path)

    return {"message": "Data is successfully uploaded!"}


@app.post(
    "/load_data",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
    description="Load data for inference.",
)
async def load_data(request: LoadDataRequest) -> MessageResponse:

    data_split = request.data_split
    data_path = DATA_DIR / f"data-{data_split}.pkl"

    if not data_path.exists():
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="Data is not uploaded yet.",
        )

    data = pd.read_pickle(data_path)

    app.state.data_split = data_split
    app.state.data = data

    return {"message": f"Data from {data_split} split is successfully loaded!"}


@app.post(
    "/unload_data",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
    description="Unload data from memory.",
)
async def unload_data() -> MessageResponse:

    data_split = app.state.data_split
    data = app.state.data

    if data is None:
        return {"message": "No data is currently loaded."}

    app.state.data_split = None
    app.state.data = None

    return {"message": f"Data from {data_split} split is unloaded."}


@app.get(
    "/get_data_status",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
)
async def get_data_status() -> MessageResponse:

    data_split = app.state.data_split
    data = app.state.data

    if data is None:
        return {"message": "No data is currently loaded."}

    return {"message": f"Data from {data_split} split is currently loaded."}


@app.post(
    "/evaluate",
    status_code=HTTPStatus.OK,
    response_model=Metrics,
    description="...",
)
async def evaluate() -> Metrics:

    cfg = app.state.config
    model = app.state.model
    data_split = app.state.data_split
    data = app.state.data

    if model is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No model is currently loaded.",
        )

    if data is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No data is currently loaded.",
        )

    X = data[cfg.features]
    y = data[cfg.target]

    try:
        metrics = compute_metrics(model, X, y)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Encountered error while computing metrics: {str(e)}",
        )

    return {"data_split": data_split, **metrics}


@app.post(
    "/compute_metric",
    status_code=HTTPStatus.OK,
    response_model=ComputeMetricResponse,
    description="...",
)
async def compute_metric(request: ComputeMetricRequest) -> ComputeMetricResponse:

    metric_name = request.metric_name

    cfg = app.state.config
    model = app.state.model
    data = app.state.data

    if model is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No model is currently loaded.",
        )

    if data is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="No data is currently loaded.",
        )

    X = data[cfg.features]
    y = data[cfg.target]

    try:
        scorer = get_scorer(metric_name)
        metric_value = scorer(model, X, y)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=f"Encountered error while computing metric: {str(e)}",
        )

    return {"metric_value": metric_value}


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        # reload=True,
    )
