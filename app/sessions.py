# app/sessions.py
"""
In‑memory session management.

Each session stores a chronological list of messages:
[{"role": "user"|"assistant", "content": "..."}, ...]

This is intentionally lightweight (no persistence) to satisfy the
"remember previous user queries within the same active session" requirement.
"""

from typing import Dict, List, Any
from threading import Lock
import logging
logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manage conversational state across multiple sessions.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Functionality
    -------------
    1. Stores chat histories in a dictionary keyed by session_id.
    2. Provides thread‑safe creation and retrieval using a Lock.
    3. Appends user/assistant messages for context accumulation.
    """

    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = Lock()

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve (or lazily create) the history list for a session.

        Parameters
        ----------
        session_id : str
            Identifier provided by the client to group chat turns.

        Returns
        -------
        list[dict]
            Mutable list of message objects.

        Functionality
        -------------
        1. Creates an empty list if the session_id is unseen.
        2. Returns the list by reference so callers can read it.
        """
        with self._lock:
            return self._sessions.setdefault(session_id, [])

    def append_user(self, session_id: str, text: str) -> None:
        """
        Append a user message to the session history.

        Parameters
        ----------
        session_id : str
            Session identifier.
        text : str
            Raw user message content.

        Returns
        -------
        None

        Functionality
        -------------
        1. Ensures history list exists.
        2. Appends a dict with role="user".
        """
        logger.debug("Appending user message session=%s", session_id)
        history = self.get_history(session_id)
        history.append({"role": "user", "content": text})

    def append_assistant(self, session_id: str, text: str) -> None:
        """
        Append an assistant message to the session history.

        Parameters
        ----------
        session_id : str
            Session identifier.
        text : str
            Assistant response content (plain text only).

        Returns
        -------
        None

        Functionality
        -------------
        1. Ensures history list exists.
        2. Appends a dict with role="assistant".
        """
        history = self.get_history(session_id)
        history.append({"role": "assistant", "content": text})
