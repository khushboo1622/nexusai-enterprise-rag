"""
backend/mcp/tool_executor.py

MCP Tool Executor — the agent layer.

Flow:
  1. Detect if query needs an MCP tool (hr_read / hr_write intent)
  2. Build prompt with available tools for this role
  3. Ask LLM which tool to call and with what parameters
  4. Validate the tool call (RBAC + schema)
  5. Execute against MongoDB
  6. Format the result as a natural language answer

This is what makes it "agentic" — the LLM decides what action to take,
not hardcoded if/else logic.
"""

import json
import logging
import re
from typing import Optional

from backend.mcp.hr_tools import (
    get_tools_for_role,
    format_tools_for_prompt,
    TOOL_PERMISSIONS,
)
from backend.mcp.hr_db import (
    find_employee,
    update_employee,
    list_employees,
)
from backend.chat.llm_provider import get_llm

logger = logging.getLogger(__name__)

# Write operations that need explicit user confirmation signals
WRITE_TOOLS = {"hr_update_employee"}


def _extract_first_json_object(text: str) -> dict:
    """
    Parse and return the first valid JSON object from model output.

    Some model providers can occasionally return duplicated JSON lines
    or append partial trailing text. This keeps tool selection robust.
    """
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in tool selection response")


def detect_hr_intent(question: str) -> Optional[str]:
    """
    Detect if a query is asking about employee records (not policies).
    Returns "hr_read", "hr_write", or None.

    The distinction matters:
    - "What is our leave policy?" → None (RAG handles this)
    - "What is Ishaan's leave balance?" → "hr_read" (MCP handles this)
    - "Update Ishaan's leave to 15 days" → "hr_write" (MCP handles this)
    """
    q = question.lower()

    write_signals = [
        "update", "change", "set ", "modify", "edit",
        "increase", "decrease", "add leave", "deduct",
        "transfer", "promote", "move to", "assign",
    ]

    read_signals = [
        "who is", "details of", "information about",
        "tell me about", "show me", "find employee",
        "list employees", "who are", "employee id",
        "leave balance", "contact of", "email of",
        "phone of", "manager of", "department of",
        "is still", "still work", "currently work",
        "joining date", "date of joining",
    ]

    hr_field_signals = [
        "leave", "leave balance", "leaves taken", "status",
        "location", "department", "phone", "email",
    ]

    # Check write first (more specific)
    if any(s in q for s in write_signals):
        # Also needs to reference an employee
        employee_signals = [
            "employee", "him", "her", "their", "his", "that", "this", "it",
            "patel", "sharma", "kumar", "singh", "emp", "id ",
        ]
        if (
            any(s in q for s in employee_signals)
            or any(s in q for s in hr_field_signals)
            or re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', question)
        ):
            return "hr_write"

    if any(s in q for s in read_signals):
        return "hr_read"

    # Check for employee name pattern (First Last)
    if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', question):
        return "hr_read"

    return None


def execute_tool_call(
    tool_name: str,
    parameters: dict,
    user_email: str,
    user_role: str,
) -> dict:
    """
    Execute a validated MCP tool call against MongoDB.
    Returns {"success": bool, "data": any, "message": str}
    """
    # RBAC check
    allowed_roles = TOOL_PERMISSIONS.get(tool_name, set())
    if user_role not in allowed_roles:
        return {
            "success": False,
            "data": None,
            "message": f"Your role ({user_role}) does not have permission to use '{tool_name}'."
        }

    try:
        if tool_name == "hr_get_employee":
            field = parameters.get("field")
            emp_id = parameters.get("employee_id")

            # Use find_employee for both specific field and general lookup
            # find_employee returns structured result with disambiguation support
            result = find_employee(
                name=parameters.get("name"),
                employee_id=emp_id,
                department=parameters.get("department"),
                caller_role=user_role,
            )

            if result["count"] == 0:
                return {"success": False, "data": None, "message": result["message"]}

            # If specific field requested, filter response
            if field and result["employees"]:
                filtered = []
                for emp in result["employees"]:
                    filtered.append({
                        "employee_id": emp.get("employee_id"),
                        "name": emp.get("name"),
                        field: emp.get(field, "Field not found"),
                    })
                return {"success": True, "data": filtered, "message": result["message"]}

            return {"success": True, "data": result, "message": result["message"]}

        elif tool_name == "hr_update_employee":
            result = update_employee(
                employee_id=parameters["employee_id"],
                field=parameters["field"],
                value=parameters["value"],
                updated_by=user_email,
                role=user_role,
            )
            return {
                "success": result["success"],
                "data": result,
                "message": result["message"]
            }

        elif tool_name == "hr_list_employees":
            results = list_employees(
                department=parameters.get("department"),
                role=parameters.get("role"),
                location=parameters.get("location"),
                status=parameters.get("status", "active"),
                limit=parameters.get("limit", 10),
            )
            if not results:
                return {"success": False, "data": None,
                        "message": "No employees found matching the specified filters."}
            return {"success": True, "data": results,
                    "message": f"Found {len(results)} employee(s)"}

        else:
            return {"success": False, "data": None,
                    "message": f"Unknown tool: '{tool_name}'"}

    except Exception as e:
        logger.error(f"[MCP] Tool execution error: {tool_name} -> {e}")
        return {"success": False, "data": None,
                "message": "An error occurred while accessing the HR database."}


def run_mcp_agent(
    question: str,
    role: str,
    user_email: str,
    chat_history: list,
) -> dict:
    """
    Main MCP agent function.

    Steps:
    1. Get tools available for this role
    2. Ask LLM which tool to call (tool selection prompt)
    3. Parse LLM response as JSON tool call
    4. Execute the tool
    5. Ask LLM to format result as natural language answer
    """
    llm = get_llm()

    # Get tools this role can use
    available_tools = get_tools_for_role(role)
    if not available_tools:
        return {
            "answer": "You don't have permission to access employee records.",
            "sources": [],
            "tool_used": None,
        }

    tools_prompt = format_tools_for_prompt(available_tools)

    # Build conversation history
    history_str = ""
    if chat_history:
        for msg in chat_history[-4:]:
            label = "User" if "USER" in str(msg.role).upper() else "Assistant"
            history_str += f"{label}: {msg.content}\n"

    history_prefix = f"Conversation history:\n{history_str}\n" if history_str else ""

    # Step 1: Ask LLM to select and parameterize the right tool
    tool_selection_prompt = (
        f"You are an HR assistant agent with access to the following tools.\n\n"
        f"{tools_prompt}\n\n"
        f"{history_prefix}"
        f"User question: {question}\n\n"
        f"Analyze the question and respond with the appropriate tool call as JSON.\n"
        f"If no tool is needed, respond with: {{\"tool\": null}}\n"
        f"Respond ONLY with the JSON, no explanation."
    )

    try:
        tool_response = str(llm.complete(tool_selection_prompt)).strip()
        logger.info(f"[MCP] Tool selection response: {tool_response[:200]}")

        # Extract first valid JSON object (handles duplicated/trailing output)
        tool_call = _extract_first_json_object(tool_response)
        tool_name = tool_call.get("tool")
        parameters = tool_call.get("parameters", {})

        logger.info(f"[MCP] Tool selected: {tool_name} | params: {parameters}")

    except Exception as e:
        logger.error(f"[MCP] Tool selection failed: {e}")
        return {
            "answer": "I had trouble understanding your request. Could you be more specific about which employee you're asking about?",
            "sources": [],
            "tool_used": None,
        }

    # No tool needed
    if not tool_name:
        return {
            "answer": "I couldn't determine what HR data you need. Please specify the employee name or ID.",
            "sources": [],
            "tool_used": None,
        }

    # Step 2: Execute the tool
    execution_result = execute_tool_call(
        tool_name=tool_name,
        parameters=parameters,
        user_email=user_email,
        user_role=role,
    )

    logger.info(
        f"[MCP] Tool executed: {tool_name} | "
        f"success={execution_result['success']} | "
        f"message={execution_result['message']}"
    )

    # Step 3: Ask LLM to format result as natural language
    if not execution_result["success"]:
        return {
            "answer": execution_result["message"],
            "sources": [],
            "tool_used": tool_name,
        }

    format_prompt = (
        f"You are NexusAI, an internal HR assistant.\n"
        f"A database query returned the following result:\n\n"
        f"{json.dumps(execution_result['data'], indent=2, default=str)}\n\n"
        f"User's original question: {question}\n\n"
        f"Instructions:\n"
        f"- Format this data as a clear, natural language response\n"
        f"- Be concise and professional\n"
        f"- For employee lists, use a clean bullet format\n"
        f"- Never mention database, MongoDB, or technical details\n"
        f"- If salary shows [REDACTED], mention it's confidential\n"
        f"- For update confirmations, confirm what was changed clearly\n"
        f"Answer:"
    )

    try:
        formatted_answer = str(llm.complete(format_prompt)).strip()
    except Exception as e:
        logger.error(f"[MCP] Format step failed: {e}")
        formatted_answer = execution_result["message"]

    return {
        "answer": formatted_answer,
        "sources": [],  # MCP results don't have Qdrant sources
        "tool_used": tool_name,
    }