from pydantic import BaseModel

class Request(BaseModel):
    prompt: str

class Response(BaseModel):
    status: int
    message: str
    data: object