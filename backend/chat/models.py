"""
backend/chat/models.py

Pydantic models for the chat API.
"""

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the maternity leave policy?"
            }
        }


class SourceDocument(BaseModel):
    """A single source document returned with the answer for citation."""
    file_name: str
    department: str
    doc_type: str
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    role: str
    sources: list[SourceDocument] = []
    question: str
    log_id: Optional[str] = None  # MongoDB log ID for feedback submission


class TokenData(BaseModel):
    """Data extracted from JWT token — used internally."""
    user_id: str
    email: str
    role: str