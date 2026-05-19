"""
backend/chat/rbac.py

Role Based Access Control — maps each role to Qdrant collections.

Roles are derived from employee department (see auth/utils.py).
This file maps roles to which Qdrant collections they can search.

GENERAL role = employees in ops, compliance, risk, QA etc.
They can only search the general collection.
"""

from fastapi import HTTPException, status

# Role -> Qdrant filter value
# Used in Qdrant payload filter: allowed_roles contains this value
ROLE_ACCESS: dict[str, str] = {
    "HR":          "HR",
    "FINANCE":     "FINANCE",
    "ENGINEERING": "ENGINEERING",
    "MARKETING":   "MARKETING",
    "C_LEVEL":     "C_LEVEL",
    "GENERAL":     "GENERAL",   # operations, compliance, risk, QA etc
}

VALID_ROLES = set(ROLE_ACCESS.keys())


def get_role_filter(role: str) -> str:
    """
    Returns the role string used in Qdrant allowed_roles filter.

    The filter checks: does allowed_roles list contain this value?

    GENERAL employees only get docs where allowed_roles contains "GENERAL"
    which is set during ingestion for the general/ folder only.
    """
    role_upper = role.upper()
    if role_upper not in VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid role: '{role}'.",
        )
    return ROLE_ACCESS[role_upper]