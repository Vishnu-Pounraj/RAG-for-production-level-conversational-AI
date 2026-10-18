from pydantic import BaseModel , Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="User question")

class SourceReference(BaseModel):
    id: str
    source_type: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    status: str = "success"