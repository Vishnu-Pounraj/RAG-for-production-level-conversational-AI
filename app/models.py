from pydantic import BaseModel , Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="User question")

class SourceReference(BaseModel):
    id: str
    source_type: Optional[str] = None
    score: Optional[float] = None
    document_name: Optional[str] = None
    s3_key: Optional[str] = None
    url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    status: str = "success"