"""
backend/mcp/hr_tools.py

MCP Tool definitions for HR employee data.

These follow the MCP (Model Context Protocol) specification:
- Each tool has a name, description, and inputSchema (JSON Schema)
- The LLM reads these definitions and decides which tool to call
- Tool calls are structured JSON that we execute against MongoDB

Why formal MCP protocol vs simple functions:
- Interoperable — any MCP-compatible client can use these tools
- Self-documenting — LLM understands tool capabilities from schema
- Type-safe — inputSchema validates parameters before execution
- Industry standard — what production AI systems actually use
"""

from typing import Any

# ── Tool definitions (MCP spec format) ────────────────────────────────────
# These are sent to the LLM so it knows what tools are available
# and what parameters each tool accepts.

HR_TOOLS = [
    {
        "name": "hr_get_employee",
        "description": (
            "Fetch employee details from the HR database. "
            "Use when user asks about a specific employee's information, "
            "details, contact, role, department, leave balance, or status. "
            "At least one of: name, employee_id, or department must be provided."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Employee full name or partial name to search"
                },
                "employee_id": {
                    "type": "string",
                    "description": "Exact employee ID (e.g. FINEMP1012)"
                },
                "department": {
                    "type": "string",
                    "description": "Department name to filter by"
                },
                "field": {
                    "type": "string",
                    "description": "Specific field to retrieve (e.g. leave_balance, email, role). If not provided, returns all basic details."
                }
            },
            "additionalProperties": False,
        }
    },
    {
        "name": "hr_update_employee",
        "description": (
            "Update a specific field in an employee record. "
            "Use when user explicitly asks to update, change, set, or modify "
            "an employee's information such as leave balance, contact details, location. "
            "Requires the employee_id, the field to update, and the new value. "
            "ONLY available to HR and C_LEVEL roles."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "Employee ID to update (e.g. FINEMP1012)"
                },
                "field": {
                    "type": "string",
                    "description": "Field to update (e.g. leave_balance, phone, location, status)"
                },
                "value": {
                    "type": "string",
                    "description": "New value for the field"
                }
            },
            "required": ["employee_id", "field", "value"],
            "additionalProperties": False,
        }
    },
    {
        "name": "hr_list_employees",
        "description": (
            "List employees with optional filters. "
            "Use when user asks to list, show, or find all employees "
            "in a department, role, or location. "
            "Returns basic info only — no salary data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "Filter by department name"
                },
                "role": {
                    "type": "string",
                    "description": "Filter by job role/title"
                },
                "location": {
                    "type": "string",
                    "description": "Filter by office location"
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "on_leave"],
                    "description": "Filter by employment status (default: active)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results to return (default: 10, max: 20)"
                }
            },
            "additionalProperties": False,
        }
    }
]

# ── RBAC: which roles can use which tools ──────────────────────────────────
TOOL_PERMISSIONS = {
    "hr_get_employee":    {"HR", "C_LEVEL"},
    "hr_update_employee": {"HR", "C_LEVEL"},
    "hr_list_employees":  {"HR", "C_LEVEL"},
}


def get_tools_for_role(role: str) -> list[dict]:
    """
    Returns only the tool definitions the user's role is allowed to use.
    These are injected into the LLM prompt so it only knows about permitted tools.
    """
    allowed_tools = []
    for tool in HR_TOOLS:
        allowed_roles = TOOL_PERMISSIONS.get(tool["name"], set())
        if role in allowed_roles:
            allowed_tools.append(tool)
    return allowed_tools


def format_tools_for_prompt(tools: list[dict]) -> str:
    """
    Format tool definitions as a string for injection into the LLM prompt.
    The LLM reads this and understands what tools are available.
    """
    if not tools:
        return ""

    import json
    lines = ["AVAILABLE TOOLS (call these for employee data queries):"]
    for tool in tools:
        lines.append(f"\nTool: {tool['name']}")
        lines.append(f"Description: {tool['description']}")
        lines.append(f"Parameters: {json.dumps(tool['inputSchema']['properties'], indent=2)}")
        if "required" in tool["inputSchema"]:
            lines.append(f"Required: {tool['inputSchema']['required']}")
    lines.append(
        "\nTo call a tool, respond with ONLY this JSON format (no other text):\n"
        '{"tool": "tool_name", "parameters": {"param": "value"}}'
    )
    return "\n".join(lines)