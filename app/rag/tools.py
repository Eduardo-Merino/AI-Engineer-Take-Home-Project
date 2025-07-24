# app/rag/tools.py
"""
Definition and execution helpers for LLM tools.

We expose two simple tools:
1. get_order_status(order_id): returns a dummy shipping status.
2. send_email(recipient_email, message): simulates sending an email.

Both are intentionally lightweight to demonstrate schema design + invocation.
"""

from typing import Any, Dict, Callable, List
from datetime import date
import logging

# ---------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)

def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Return a dummy order status for a given order_id.

    Parameters
    ----------
    order_id : str
        Unique identifier of the order supplied by the user.

    Returns
    -------
    dict
        JSON-like dictionary with static order information.

    Functionality
    -------------
    1. Builds a fake shipping record.
    2. Uses today's date to show an estimated_delivery example.
    3. Returns a dictionary that can be serialized to JSON.
    """
    logger.info("Tool get_order_status called with order_id=%s", order_id)
    return {
        "order_id": order_id,
        "status": "shipped",
        "estimated_delivery": str(date.today()),
        "carrier": "Acme Logistics"
    }


def send_email(recipient_email: str, message: str) -> Dict[str, Any]:
    """
    Simulate sending an email.

    Parameters
    ----------
    recipient_email : str
        Email address of the recipient.
    message : str
        Body content the user wants to send.

    Returns
    -------
    dict
        JSON-like dictionary indicating success.

    Functionality
    -------------
    1. Validates minimal shape (this example trusts the inputs).
    2. Returns a success payload with an echo of the content.
    3. Can be extended to integrate with a real provider.
    """
    logger.info("Tool send_email called to=%s (len=%d)", recipient_email, len(message))
    return {
        "success": True,
        "recipient": recipient_email,
        "message_preview": message[:60],
        "detail": "Email sent (simulated)."
    }


# Map tool names to implementation callables
_TOOL_FUNCTIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "get_order_status": get_order_status,
    "send_email": send_email,
}


def get_tool_specs() -> List[Dict[str, Any]]:
    """
    Build the tool specification list to send to the Anthropic API.

    Parameters
    ----------
    None

    Returns
    -------
    list[dict]
        List of tool specification dictionaries following Anthropic's expected schema.

    Functionality
    -------------
    1. Provides name, description, and JSON schema for each tool.
    2. Used when creating the Claude message with `tools=...`.
    3. Keeps schema centralized; easy to modify or extend.
    """
    return [
        {
            "name": "get_order_status",
            "description": "Look up the current shipping status for a specific order_id.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Opaque identifier of the order."
                    }
                },
                "required": ["order_id"],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email message to a recipient (simulated).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "recipient_email": {
                        "type": "string",
                        "description": "Destination email address."
                    },
                    "message": {
                        "type": "string",
                        "description": "Plain text content to send."
                    }
                },
                "required": ["recipient_email", "message"],
            },
        },
    ]


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the tool implementation specified by name.

    Parameters
    ----------
    name : str
        Name of the tool to execute (must exist in _TOOL_FUNCTIONS).
    arguments : dict
        Parsed arguments from the LLM tool call.

    Returns
    -------
    dict
        Result of the tool execution (serializable).

    Functionality
    -------------
    1. Looks up the callable by name.
    2. Unpacks the dictionary as keyword arguments.
    3. Returns the tool output.
    4. Raises KeyError if the tool is unknown.
    """
    if name not in _TOOL_FUNCTIONS:
        raise KeyError(f"Unknown tool: {name}")
    func = _TOOL_FUNCTIONS[name]
    return func(**arguments)
