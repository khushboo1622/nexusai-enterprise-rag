"""
backend/mcp/hr_db.py

MongoDB CRUD for employee records.

Key design decisions:
  - Salary always redacted unless caller is C_LEVEL (handled in tool layer)
  - Duplicate name handling: returns all matches with disambiguation info
  - Immutable fields cannot be updated via chatbot
  - Full audit trail on every update
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from backend.db.mongodb import get_db

logger = logging.getLogger(__name__)

IMMUTABLE_FIELDS = {"_id", "employee_id", "created_at", "source"}

C_LEVEL_ONLY_FIELDS = {"salary", "compensation", "grade", "role", "department"}

HR_UPDATABLE_FIELDS = {
    "leave_balance", "leaves_taken",
    "phone", "email", "location",
    "manager_id", "emergency_contact",
    "date_of_joining", "status",
}


def get_employees_collection():
    db = get_db()
    col = db["employees"]
    try:
        col.create_index("employee_id", unique=True)
        col.create_index("department")
        col.create_index("status")
        col.create_index("name")
    except Exception:
        pass
    return col


def _redact_sensitive(doc: dict, caller_role: str = "") -> dict:
    """Redact salary unless caller is C_LEVEL."""
    result = dict(doc)
    result.pop("_id", None)
    result.pop("hashed_password", None)
    if caller_role != "C_LEVEL":
        for field in ["salary", "compensation", "wage"]:
            if field in result:
                result[field] = "[REDACTED - C-Level access only]"
    return result


def find_employee(
    name: Optional[str] = None,
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
    caller_role: str = "",
) -> dict:
    """
    Search for employees. Returns structured result with disambiguation support.

    When multiple employees share the same name:
    - Returns all matches
    - Includes employee_id + department for each so user can identify the right one
    - Caller should follow up with specific employee_id for exact record

    Returns:
        {
          "count": int,
          "employees": [...],
          "disambiguation_needed": bool,
          "message": str
        }
    """
    col = get_employees_collection()
    query = {}

    if employee_id:
        query["employee_id"] = employee_id.upper().strip()

    if department:
        query["department"] = {"$regex": department, "$options": "i"}

    if name and not employee_id:
        # Split name to handle partial matches
        # "Ishaan Patel" matches first+last, "Ishaan" matches first name only
        name_parts = name.strip().split()
        if len(name_parts) >= 2:
            # Full name — try exact first, then fuzzy
            query["name"] = {"$regex": name.strip(), "$options": "i"}
        else:
            # Single name — match anywhere in name
            query["name"] = {"$regex": name_parts[0], "$options": "i"}

    results = list(col.find(query, {"_id": 0}).limit(10))

    if not results:
        search_term = name or employee_id or department or "that criteria"
        return {
            "count": 0,
            "employees": [],
            "disambiguation_needed": False,
            "message": f"No employees found matching '{search_term}'.",
        }

    # Redact sensitive fields
    clean_results = [_redact_sensitive(r, caller_role) for r in results]

    # Check for duplicate names within results
    names = [r.get("name", "") for r in clean_results]
    has_duplicates = len(names) != len(set(names))

    if len(clean_results) > 1 and has_duplicates:
        # Multiple employees with same name — need disambiguation
        # Add prominent disambiguation info to each result
        for emp in clean_results:
            emp["_disambiguation_hint"] = (
                f"ID: {emp.get('employee_id')} | "
                f"Dept: {emp.get('department')} | "
                f"Location: {emp.get('location', 'N/A')}"
            )
        return {
            "count": len(clean_results),
            "employees": clean_results,
            "disambiguation_needed": True,
            "message": (
                f"Found {len(clean_results)} employees named '{name}'. "
                f"Please specify the employee ID for the exact record."
            ),
        }

    return {
        "count": len(clean_results),
        "employees": clean_results,
        "disambiguation_needed": False,
        "message": f"Found {len(clean_results)} employee(s).",
    }


def update_employee(
    employee_id: str,
    field: str,
    value: str,
    updated_by: str,
    role: str,
) -> dict:
    """Update a specific field with role-based permission check."""
    field = field.lower().strip()

    if field in IMMUTABLE_FIELDS:
        return {"success": False, "message": f"Field '{field}' cannot be modified."}

    if field in C_LEVEL_ONLY_FIELDS and role != "C_LEVEL":
        return {"success": False,
                "message": f"Field '{field}' can only be updated by C-Level executives."}

    allowed = HR_UPDATABLE_FIELDS | C_LEVEL_ONLY_FIELDS
    if field not in allowed:
        return {
            "success": False,
            "message": (
                f"Field '{field}' is not updatable. "
                f"Updatable fields: {', '.join(sorted(HR_UPDATABLE_FIELDS))}"
            )
        }

    col = get_employees_collection()
    emp = col.find_one({"employee_id": employee_id.upper()}, {"name": 1})
    if not emp:
        return {"success": False, "message": f"Employee ID '{employee_id}' not found."}

    col.update_one(
        {"employee_id": employee_id.upper()},
        {"$set": {
            field: value,
            "last_updated_at": datetime.now(timezone.utc),
            "last_updated_by": updated_by,
        }}
    )

    logger.info(f"[HR_DB] Updated {employee_id}.{field} by {updated_by} ({role})")

    return {
        "success": True,
        "message": f"Successfully updated '{field}' for {emp.get('name', employee_id)}."
    }


def list_employees(
    department: Optional[str] = None,
    role: Optional[str] = None,
    location: Optional[str] = None,
    status: Optional[str] = "active",
    limit: int = 10,
    caller_role: str = "",
) -> list[dict]:
    """List employees with optional filters."""
    col = get_employees_collection()
    query = {}

    if department:
        query["department"] = {"$regex": department, "$options": "i"}
    if role:
        query["role"] = {"$regex": role, "$options": "i"}
    if location:
        query["location"] = {"$regex": location, "$options": "i"}
    if status:
        query["status"] = status

    results = list(col.find(query, {"_id": 0}).limit(min(limit, 20)))
    return [_redact_sensitive(r, caller_role) for r in results]