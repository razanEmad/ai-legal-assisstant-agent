# AI Legal Assistant

An intelligent agent system for analyzing contracts using LangChain + LangGraph + RAG.

---

## Running

```bash
pip install -r requirements.txt
# Put your API key in .env
streamlit run streamlit_ui.py
```

---

## File Structure

```
legal_assistant/
├── agentic_workflow.py   <- core: LLM + Tools + Graph + Memory
├── streamlit_ui.py       <- UI: Agentic mode + RAG Direct mode
├── mediumblog1.txt       <- legal knowledge base (RAG source)
├── requirements.txt
└── .env                  ← GOOGLE_API_KEY
```

---

## Components and How They Work

### 1. LangChain Tools

Each tool is a regular Python function decorated with `@tool`.
The LLM reads the docstring and knows when to use each tool.

| Tool | When it is used |
|------|------------|
| `pdf_parser` | First step - extracts contract structure and section count |
| `clause_detector` | Detects clauses (payment, termination, confidentiality...) |
| `risk_analyzer` | Classifies risks: 🔴HIGH / 🟡MEDIUM / 🟢LOW |
| `legal_knowledge_retriever` | Searches `mediumblog1.txt` using RAG |
| `web_search` | Searches the internet for current information |

### 2. RAG — Retrieval Augmented Generation

```
mediumblog1.txt
    ↓  TextLoader
documents[]
    ↓  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks[]
    ↓  GoogleGenerativeAIEmbeddings
vectors[]
    ↓  FAISS.from_documents  (RAM only)
vectorstore
    ↓  as_retriever(k=3)
retriever  ->  create_retriever_tool()  ->  Tool ready for the agent
```

**chunk_size=1000**: Each chunk is at most 1000 characters  
**chunk_overlap=200**: 200-character overlap preserves context  
**k=3**: Retrieves the 3 nearest chunks for the question

### 3. LangGraph - Workflow

```
START
  ↓
[agent_node]  <- LLM reasons and decides
  ↓
should_continue()
  ├── tool_calls present? -> [ToolNode] -> execute tool -> [agent_node]
  └── no -> END
```

**AgentState**: TypedDict stores `messages` with `operator.add`
-> Each new message is appended to the previous messages.

**should_continue**: The conditional edge chooses "tools" or END.

**ToolNode**: A LangGraph prebuilt node that executes the requested tool automatically.

### 4. Memory

```python
memory = MemorySaver()          # In RAM; no database
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "session_123"}}
```

- Same `thread_id` = same conversation = same memory
- A different `thread_id` = a new independent conversation
- Closing the application clears the memory (RAM only)

---

## Expected Discussion Questions

**Q: Why chunk_overlap=200?**  
A: Without overlap, information at chunk boundaries could be split and lost. Overlap preserves context.

**Q: How does the LLM know how to use a tool?**  
A: Through `bind_tools(tools)`: the LLM reads each tool's name and docstring and decides when to call it.

**Q: How do you add a new tool?**  
A: Write a Python function, place `@tool` above it, and add it to the `tools` list.

**Q: How do you change k in RAG?**  
A: In `get_retriever()`, change `search_kwargs={"k": 3}` to 5 or 10. Higher k provides more information but takes longer.

**Q: Why MemorySaver instead of SQLite?**  
A: The project does not need persistence across sessions. RAM is faster and simpler.

**Q: What is the difference between Agentic mode and RAG Direct?**  
A: Agentic: The LLM chooses which tools to use and may use three tools in sequence.  
RAG Direct: A fixed path - question -> retriever -> prompt -> LLM -> answer.
