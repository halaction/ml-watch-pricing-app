from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from http import HTTPStatus

from schemas import MessageResponse


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup events
    ...

    yield

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


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
