"""
backend/auth/utils.py

Auth utilities:
  1. Password hashing + verification (bcrypt)
  2. JWT create + decode
  3. Department -> Role mapping
  4. Auto password generation for employees
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── Department to Role mapping ─────────────────────────────────────────────
# Departments not listed here default to "GENERAL"
DEPARTMENT_ROLE_MAP = {
    # HR
    "hr":                   "HR",
    "human resources":      "HR",

    # Finance
    "finance":              "FINANCE",
    "accounts":             "FINANCE",
    "accounting":           "FINANCE",

    # Engineering (broad — covers all tech departments)
    "technology":           "ENGINEERING",
    "engineering":          "ENGINEERING",
    "data":                 "ENGINEERING",
    "product":              "ENGINEERING",
    "design":               "ENGINEERING",
    "it":                   "ENGINEERING",

    # Marketing (broad — covers sales and business too)
    "marketing":            "MARKETING",
    "sales":                "MARKETING",
    "business":             "MARKETING",
    "business development": "MARKETING",

    # C-Level is set via is_clevel flag, not department name
}

# Departments that only see general docs
GENERAL_ONLY_DEPARTMENTS = {
    "operations", "compliance", "risk",
    "quality assurance", "qa", "legal",
    "administration", "admin", "facilities",
}


def derive_role_from_department(department: str, is_clevel: bool = False) -> str:
    """
    Map employee department to system role.

    Priority:
      1. is_clevel flag in employee record -> C_LEVEL
      2. Exact department match in map -> mapped role
      3. Everything else -> GENERAL (can only see general docs)

    This is the single source of truth for role assignment.
    """
    if is_clevel:
        return "C_LEVEL"

    dept_lower = department.lower().strip() if department else ""
    role = DEPARTMENT_ROLE_MAP.get(dept_lower, "GENERAL")

    logger.debug(f"[AUTH] department='{department}' -> role='{role}'")
    return role


def generate_employee_password(employee_id: str) -> str:
    """
    Generate default password for an employee.
    Pattern: EMP + last 4 chars of ID + @Nexus
    Example: employee_id=FINEMP1012 -> EMP1012@Nexus

    Employees should change this on first login (v2 feature).
    """
    last4 = employee_id[-4:] if len(employee_id) >= 4 else employee_id
    return f"EMP{last4}@Nexus"


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    """
    Create signed JWT token.
    Embeds employee_id, name, department, role.
    Role is embedded so chat API needs no DB lookup per request.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as e:
        logger.warning(f"JWT decode failed: {e}")
        return None