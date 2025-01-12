from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from http import HTTPStatus

from schemas import MessageResponse


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup events
    state = {"something": 123}

    yield state

    # Shutdown events
    ...


app = FastAPI(lifespan=lifespan)
# app.include_router(router=analysis_router, prefix="/api")


@app.get(
    "/",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
)
async def root() -> MessageResponse:
    return {"message": "We are up!"}


@app.get(
    "/get_model",
    status_code=HTTPStatus.OK,
    response_model=MessageResponse,
)
async def get_model(request: Request, a: int) -> MessageResponse:
    return {"message": f"as per {a} we got {request.state.something}"}


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
