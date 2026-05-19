"""
backend/auth/models.py

Auth models — updated for employee-based auth (Option B).
Login uses employee_id + password (no email required).
No signup model — employees are pre-loaded from CSV.
"""

from pydantic import BaseModel, field_validator
from typing import Optional


class LoginRequest(BaseModel):
    employee_id: str
    password: str

    @field_validator("employee_id")
    @classmethod
    def normalize_id(cls, v: str) -> str:
        return v.strip().upper()


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    employee_id: str
    name: str
    department: str
    role: str


class TokenData(BaseModel):
    """Extracted from JWT — used internally by all protected endpoints."""
    user_id: str      # employee_id
    email: str        # employee email (may be empty)
    role: str         # derived from department
    name: str = ""
    department: str = ""