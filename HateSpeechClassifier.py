from pydantic import BaseModel

class TextData(BaseModel):
    text: list[str]
