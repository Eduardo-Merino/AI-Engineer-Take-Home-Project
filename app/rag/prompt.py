# app/rag/prompt.py
"""
System prompt construction utilities.

We keep the system prompt logic isolated so it is easy to modify or extend.
The prompt defines the agent persona, behavior rules, and how retrieved
context + chat history should be used.
"""


def build_system_prompt(context_block: str) -> str:
    """
    Build the final system prompt string injected into the LLM call.

    Parameters
    ----------
    context_block : str
        Concatenated retrieved document chunks (may be empty). Passed
        by the RAG pipeline before invoking the model.

    Returns
    -------
    str
        Fully formatted system prompt text.

    Functionality
    -------------
    1. Declares the assistant persona (Customer Support Agent).
    2. Provides explicit instructions on how to use tools and retrieved context.
    3. Embeds the raw `context_block` under a dedicated header.
    4. Instructs the model how to respond when context is missing or insufficient.
    """
    return f"""
You are **HelpBot**, a concise and friendly e‑commerce customer support agent.

Capabilities:
- Answer user questions about orders, returns, shipping, and payments.
- Use the retrieved knowledge base context below when relevant.
- May call tools (get_order_status, send_email) when they help answer the query.

Grounding Rules:
1. Prefer information from the *Retrieved Context* section. If the answer is directly
   present there, summarize it in your own words.
2. If the user requests an action that maps to a tool, call the appropriate tool.
3. If context is empty or does not contain the answer, you may still answer from
   general reasoning, but clearly say when something is not found.
4. Before calling a tool, ensure required parameters are present. If missing,
   ask the user to provide them.
5. After receiving tool output, incorporate it into a natural language reply.

Retrieved Context:
------------------
{context_block or 'NO CONTEXT RETRIEVED'}

Respond helpfully. If clarification is needed, ask a follow‑up question.
Do **not** invent order IDs or email addresses.
"""
