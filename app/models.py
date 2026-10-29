from pydantic import BaseModel , Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="User question")

class SourceReference(BaseModel):
    id: str
    document_name: Optional[str] = None
    source_url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
