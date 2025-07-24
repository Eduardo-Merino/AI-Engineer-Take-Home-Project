# AI-Engineer-Take-Home-Project

## 1. Use Case
Customer Support **HelpBot** for an e‑commerce store.  
It answers FAQs (shipping, returns, payments) using Retrieval‑Augmented Generation (RAG) over a local text file and can execute two tools:

- `get_order_status(order_id)` – returns a dummy shipping status.  
- `send_email(recipient_email, message)` – simulates sending an email.

The goal is to demonstrate: system prompt design, local retrieval, and Anthropic tool calls.

## 2. Project Structure
```text
├── app/
│ ├── main.py # FastAPI app (chat endpoint)
│ ├── logging_config.py # Logging configuration
│ ├── load_data.py # Build local Chroma vector store
│ ├── sessions.py # In‑memory session history
│ ├── rag/
│ │ ├── retriever.py # Retrieval utilities
│ │ ├── tools.py # Tool definitions + execution
│ │ └── prompt.py # System prompt builder
│ └── init.py
├── knowledge_base.txt # Raw FAQ knowledge source
├── requirements.txt
├── README.md
└── .env # (Not committed) contains ANTHROPIC_API_KEY
```

## 3. Setup

### 3.1 Requirements
- Python 3.11+
- No external services required (Chroma runs locally).

### 3.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.3 Environment Variable
ANTHROPIC_API_KEY=your_real_key_here

### 3.4 Build Vector Store
Populate Chroma with embeddings:
```bash
python -m app.load_data
```
This creates ./chroma_db with the knowledge collection.

## 4. Run the API
```bash
uvicorn app.main:app --reload
```
Endpoint: POST /chat
Request Body
{
  "session_id": "abc123",
  "message": "Where is my order?"
}

You can now test the API through the Swagger UI at [//127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 5. Example Prompts

### 5.1 Pure RAG
User: “What is the standard shipping time?”
The agent retrieves the relevant chunk from knowledge_base.txt and summarizes.

### 5.2 Tool Use: Order Status
User: “Check order status for id=ORD-777 please.”
Model calls get_order_status → returns JSON → agent replies:

“Order ORD-777 is shipped. Estimated delivery: 2025-07-24 …”

### 5.3 Tool Use + Memory Use: Send Email

User: “Send an email to john@example.com stating that I wish to return the product from my last order number.” 
Model calls send_email → responds confirming the simulated send getting the order id from the last prompt.

### 5.4 Missing Parameters
User: “Send an email telling support I’m unhappy.”
Assistant asks for the recipient email before calling the tool (per system rules).

### 5.5 No Context
User: “Do you sell furniture?” (Not in knowledge base)
Agent replies that the information was not found in retrieved context.

## 6. cURL Examples
You can also test the API using the following cURL commands instead of the Swagger UI.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test1","message":"What payment methods are accepted?"}'
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test1","message":"Check order status for ORD-1001"}'
```

## 7. Design Decisions & Trade‑offs

* **In‑memory sessions**: Simplicity; no persistence across restarts. Could swap for Redis.

* **Chroma + SentenceTransformers**: Lightweight, no external infra. A larger model could improve recall but increases latency.

* **Tools are dummy**: Focus is schema + invocation, not real integrations.

* **Single retrieval per turn**: Straightforward. Future work: re‑rankers, multi‑hop retrieval.

* **Security**: API key loaded from environment; key intentionally excluded from repo.

* **Prompting**: System prompt instructs safe tool usage and grounding behavior.

## 8. Future Improvements
Persistence (Redis/DB), authentication, streaming responses, real email/order backends, evaluation scripts (retrieval quality), additional tools.
