"""
backend/auth/routes.py

Auth routes — employee-based login only.
No signup endpoint — employees loaded from CSV via migration script.

Login flow:
  1. Employee enters employee_id + password
  2. Lookup employee in MongoDB employees collection
  3. Verify bcrypt password
  4. Derive role from department
  5. Issue JWT with employee_id, name, department, role embedded
"""

import logging
from fastapi import APIRouter, HTTPException, status

from backend.auth.models import LoginRequest, LoginResponse
from backend.auth.utils import (
    verify_password,
    create_access_token,
    derive_role_from_department,
)
from backend.db.mongodb import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Login with employee ID and password",
)
async def login(request: LoginRequest):
    """
    Login using your employee ID and password.

    Default password format: EMP + last 4 digits of your ID + @Nexus
    Example: employee_id=FINEMP1012 → password=EMP1012@Nexus

    Returns a JWT token. Include in all protected requests:
        Authorization: Bearer <token>
    """
    db = get_db()
    employees = db["employees"]

    # Find employee by ID
    employee = employees.find_one(
        {"employee_id": request.employee_id},
        {"_id": 0}
    )

    # Same error for not found and wrong password — don't reveal which
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid employee ID or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password
    stored_password = employee.get("hashed_password", "")
    if not stored_password or not verify_password(request.password, stored_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid employee ID or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Derive role from department
    department = employee.get("department", "")
    is_clevel = employee.get("is_clevel", False)
    role = derive_role_from_department(department, is_clevel)

    # Build JWT payload
    token = create_access_token(data={
        "user_id":    employee["employee_id"],
        "email":      employee.get("email", ""),
        "name":       employee.get("name", ""),
        "department": department,
        "role":       role,
    })

    logger.info(
        f"[AUTH] Login: {employee['employee_id']} | "
        f"dept={department} | role={role}"
    )

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        employee_id=employee["employee_id"],
        name=employee.get("name", ""),
        department=department,
        role=role,
    )


@router.get(
    "/me",
    summary="Get current user info from JWT",
)
async def get_me(token: str):
    """Quick endpoint to verify token and see decoded info."""
    from backend.auth.utils import decode_access_token
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload