from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    message: str = Field(title="Message")
