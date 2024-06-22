from pydantic import BaseModel


class ChatReq(BaseModel):
    question: str
