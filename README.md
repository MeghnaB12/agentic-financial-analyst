# Agentic Financial Analyst (SEC 10-K)

An autonomous AI agent designed to audit public company Annual Reports (SEC 10-K Filings), extract critical risk factors, and benchmark financial performance against real-time market data.

The system utilizes a **Router-based RAG (Retrieval Augmented Generation)** architecture powered by **Llama 3.3 (70B)** to verify internal claims against external competitors.

---

## 🏗 System Architecture

The agent operates on a dynamic routing logic, deciding autonomously whether to trust internal documents or verify facts with the live web.

```mermaid
graph TD
    A[User Query] -->|Analyzed by Llama 3.3| B{Router Logic}
    B -->|Internal Question| C[<b>Vector Store Tool</b><br/>Search 10-K Filings]
    B -->|External Question| D[<b>Web Search Tool</b><br/>Competitor Data]
    C --> E[Context Retrieval]
    D --> E
    E --> F[<b>LLM Synthesis</b><br/>Reasoning & Citations]
    F --> G[Final Answer]
