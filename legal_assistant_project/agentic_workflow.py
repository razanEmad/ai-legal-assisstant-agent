"""
agentic_workflow.py
===================
AI Legal Assistant - Main File
Contains:
  1. LangChain Tools  (@tool decorator)
  2. RAG Setup        (TextLoader → Splitter → FAISS → Retriever → Tool)
  3. LLM + bind_tools (Function Calling)
  4. AgentState       (TypedDict + Annotated)
  5. LangGraph Nodes & Edges
  6. MemorySaver      (in-RAM memory)
"""

import operator
import os
import sys
from typing import Annotated, Sequence, TypedDict

# Configure UTF-8 encoding for Windows output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import create_retriever_tool, tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ─── Load environment variables (GOOGLE_API_KEY) ─────────────────────────
load_dotenv()

# Validate the API key to avoid gRPC crashes when using a placeholder
api_key = os.environ.get("GOOGLE_API_KEY", "")
is_api_key_valid = True

if not api_key or api_key.strip() == "":
    is_api_key_valid = False
else:
    # Remove surrounding quotes and compare against common placeholders
    clean_key = api_key.strip().strip("'\"")
    placeholders = ["YOUR_API_KEY", "YOUR_API_KEY_HERE", "your_api_key", "your_api_key_here", "PUT_YOUR_API_KEY_IN_.env"]
    if clean_key in placeholders:
        is_api_key_valid = False

# ═══════════════════════════════════════════════════════════════════════════
# 1. LLM - The primary language model
# ═══════════════════════════════════════════════════════════════════════════
# Gemini-2.5-flash is fast and sufficient for contract analysis
if is_api_key_valid:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
else:
    # Use an English placeholder to avoid gRPC issues during initialization
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="MISSING_OR_INVALID_KEY")



# ═══════════════════════════════════════════════════════════════════════════
# 2. RAG - Legal knowledge base
# ═══════════════════════════════════════════════════════════════════════════
def get_retriever():
    """
    Build a retriever from the legal text file.

    Steps:
    1. TextLoader  -> load text from the file
    2. RecursiveCharacterTextSplitter -> split text into chunks
            chunk_size=1000: each chunk is at most 1000 characters
            chunk_overlap=200: 200-character overlap preserves context
        3. GoogleGenerativeAIEmbeddings -> convert each chunk to a vector
        4. FAISS.from_documents -> store vectors in RAM (no database)
        5. as_retriever(k=3) -> retrieve the 3 nearest chunks
    """
    if not is_api_key_valid:
        raise ValueError("Google API key is missing or invalid. Update the .env file with a valid key.")

    # Resolve the file path next to this script
    #base_dir = "C:\Users\Razan\projects"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir, "mediumblog1.txt")

    if os.path.exists(txt_path):
        loader = TextLoader(txt_path, encoding="utf-8")
        docs = loader.load()
    else:
        # Fallback text if the file is unavailable
        from langchain_core.documents import Document
        docs = [Document(
            page_content=(
                "A non-compete period must not exceed two years under Egyptian labor law. "
                "Termination without notice requires compensation. "
                "Excessive penalties may be invalid. "
                "A lifetime confidentiality clause is excessive."
            )
        )]

    # Split the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # Convert chunks to vectors and store them in FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # k=3 -> retrieve the 3 nearest chunks for each search
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def setup_rag():
    """
    Convert the retriever into a tool the agent can use.
    create_retriever_tool() gives the tool a name and description for the LLM.
    """
    retriever = get_retriever()
    return create_retriever_tool(
        retriever,
        name="legal_knowledge_retriever",
        description=(
            "Search the Egyptian legal knowledge base for statutes, rulings, and definitions. "
            "Use it when you need to reference labor law or determine legal limits."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. LangChain Tools - Custom legal assistant tools
# ═══════════════════════════════════════════════════════════════════════════

@tool
def pdf_parser(contract_text: str) -> str:
    """
        Analyze the contract structure and extract key information:
            - Number of sections (each sentence starting with 'Article' is counted)
            - Preview of the first 150 characters
            - Total text length
        Use it first to understand the contract structure before detailed analysis.
    """
    sections_count = contract_text.count("Article")
    total_chars = len(contract_text)
    preview = contract_text[:150].strip()
    return (
        f"📄 Contract structure analysis:\n"
        f"  • Sections: {sections_count}\n"
        f"  • Total characters: {total_chars}\n"
        f"  • Preview: {preview}..."
    )


@tool
def clause_detector(contract_text: str) -> str:
    """
        Detect the key clauses present in the contract:
            - Payment terms (salary, pounds, dollars)
            - Termination terms
            - Non-compete
            - Confidentiality
            - Dispute resolution / arbitration
        Return the detected clauses or a message if none are found.
    """
    detected = []

    if any(kw in contract_text.lower() for kw in ["salary", "pound", "dollar", "wage", "bonus"]):
        detected.append("💰 Payment and wage terms")

    if any(kw in contract_text.lower() for kw in ["termination", "terminate", "rescission"]):
        detected.append("🔚 Termination terms")

    if any(kw in contract_text.lower() for kw in ["non-compete", "noncompete", "competition"]):
        detected.append("🚫 Non-compete clause")

    if any(kw in contract_text.lower() for kw in ["confidentiality", "confidential", "private"]):
        detected.append("🔒 Confidentiality clause")

    if any(kw in contract_text.lower() for kw in ["arbitration", "court", "litigation", "dispute"]):
        detected.append("⚖️ Dispute resolution")

    if any(kw in contract_text.lower() for kw in ["leave", "sick", "annual"]):
        detected.append("🏖️ Leave")

    if not detected:
        return "⚠️ No clear key clauses were detected. Make sure the contract text is present."

    return "Detected contract clauses:\n" + "\n".join(f"  {b}" for b in detected)


@tool
def risk_analyzer(contract_text: str) -> str:
    """
        Analyze the contract and classify legal risks:
            🔴 HIGH   - serious risks requiring immediate attention
            🟡 MEDIUM - moderate risks worth reviewing
            🟢 LOW    - no clear risks
        Search for excessive non-competes, termination without notice, excessive
        penalties, lifetime confidentiality, and foreign jurisdiction.
    """
    risks = []

    # Check for an excessive non-compete
    if "anywhere in the world" in contract_text.lower():
        risks.append("🔴 HIGH: Open-ended geographic non-compete (anywhere in the world) - legally invalid.")
    if "5 years" in contract_text.lower() or "five years" in contract_text.lower():
        risks.append("🔴 HIGH: Non-compete period exceeds two years - beyond the legal limit.")

    # Check for termination without notice
    if any(kw in contract_text.lower() for kw in ["without notice", "at any time without"]):
        risks.append("🔴 HIGH: Termination at any time without notice or compensation - contrary to labor law.")

    # Check for excessive penalties
    if any(kw in contract_text.lower() for kw in ["500,000", "million", "penalt"]):
        risks.append("🔴 HIGH: Excessive financial penalties - may be deemed abusive and reduced by a court.")

    # Check for excessive confidentiality
    if "lifetime" in contract_text.lower() or "for life" in contract_text.lower():
        risks.append("🟡 MEDIUM: Lifetime confidentiality clause - excessive and open to challenge.")

    # Check for foreign jurisdiction
    if any(kw in contract_text.lower() for kw in ["law of england", "london", "new york", "foreign"]):
        risks.append("🟡 MEDIUM: Foreign jurisdiction - may mean high legal costs for the worker.")

    if not risks:
        return "🟢 LOW: No clear legal risks were detected in the text."

    summary = f"{len(risks)} risks detected:\n"
    summary += "\n".join(f"  {r}" for r in risks)
    return summary


# ─── Internet search tool ─────────────────────────────────────────────────
# DuckDuckGoSearchRun searches the internet without an API key
web_search = DuckDuckGoSearchRun()
web_search.name = "web_search"
web_search.description = (
    "Search the internet for current legal information, court rulings, or any "
    "information missing from the knowledge base. Use it for questions requiring updates."
)

# ─── Assemble all tools ──────────────────────────────────────────────────
try:
    retriever_tool = setup_rag()
    tools = [pdf_parser, clause_detector, risk_analyzer, retriever_tool, web_search]
    print("✅ RAG ready - all tools loaded.")
except Exception as e:
    print(f"⚠️ RAG error: {e} - running without retriever_tool.")
    tools = [pdf_parser, clause_detector, risk_analyzer, web_search]


# ═══════════════════════════════════════════════════════════════════════════
# 4. LLM + Tools - Connect tools to the model (function calling)
# ═══════════════════════════════════════════════════════════════════════════
# bind_tools() tells the LLM about the tools and lets it decide when to use them
llm_with_tools = llm.bind_tools(tools)


# ═══════════════════════════════════════════════════════════════════════════
# 5. AgentState - Agent state (short-term memory)
# ═══════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    """
    TypedDict defining the state shared by all nodes in the graph.

    messages:
      - Annotated[Sequence[BaseMessage], operator.add]
            - operator.add means new messages are appended at each step
                (not replaced), preserving the complete conversation history.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ═══════════════════════════════════════════════════════════════════════════
# 6. LangGraph Nodes
# ═══════════════════════════════════════════════════════════════════════════

def agent_node(state: AgentState) -> dict:
    """
    Main node - invokes the LLM with the complete message history.

    Input:  state["messages"] - all previous messages
    Output: {"messages": [LLM response]}

    The LLM decides:
            • If it wants a tool, it returns an AIMessage with tool_calls
            • If it has a final answer, it returns an AIMessage with content only
    """
    if not is_api_key_valid:
        return {"messages": [AIMessage(content="⚠️ The legal assistant cannot run without a valid API key. Update `.env` with your Gemini API key in `GOOGLE_API_KEY`, then restart the application.")]}
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Conditional edge - decides the next step after agent_node.

    Logic:
            • If the last message has tool_calls, go to the "tools" node
            • Otherwise, go to END (the conversation is complete)

    This is the core agentic loop: Agent -> Tools -> Agent -> ... -> END
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ═══════════════════════════════════════════════════════════════════════════
# 7. LangGraph Workflow - Build the graph
# ═══════════════════════════════════════════════════════════════════════════

def create_workflow() -> StateGraph:
    """
    Build the Agent's StateGraph:

    Structure:
      START
        ↓
      [agent_node]  ← ─────────────────┐
        ↓                               │
      should_continue()                 │
        ├─ "tools" → [ToolNode] ────────┘  (loop)
        └─ END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    # LangGraph's prebuilt ToolNode executes the requested tool automatically
    graph.add_node("tools", ToolNode(tools))

    # Fixed edges
    graph.add_edge(START, "agent")          # Always start at the agent
    graph.add_edge("tools", "agent")        # Return to the agent after a tool

    # Conditional edge - choose tools or END
    graph.add_conditional_edges(
        "agent",
        should_continue,
        ["tools", END],
    )

    return graph


# ═══════════════════════════════════════════════════════════════════════════
# Demo contract for testing
# ═══════════════════════════════════════════════════════════════════════════
# Demo contract for testing
DEMO_CONTRACT = """Employment contract between Advanced Technologies Ltd. and Alex Morgan:
Article 1 - Term: January 2025 to December 2025.
Article 2 - Salary: 15,000 pounds per month.
Article 3 - Non-compete: 5 years anywhere in the world.
Article 4 - Termination: At any time without notice or compensation.
Article 5 - Confidentiality: For life.
Article 6 - Arbitration: London under the law of England.
Article 7 - Penalties: 500,000 pounds for any violation."""


# ═══════════════════════════════════════════════════════════════════════════
# Run directly from the terminal
# Run directly from the terminal
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    workflow = create_workflow()

    # MemorySaver stores checkpoints in RAM only; each thread_id has separate memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # thread_id identifies the conversation session; the same ID reuses its memory
    config = {"configurable": {"thread_id": "terminal_session_1"}}

    print("\n" + "=" * 60)
    print("Demo contract:")
    print(DEMO_CONTRACT)
    print("=" * 60 + "\n")

    user_input = (
        "Analyze the following contract step by step: first extract its structure, "
        "then detect clauses, analyze risks, and compare the results with Egyptian law:\n\n" + DEMO_CONTRACT
    )

    print("🚀 Starting the agentic workflow...\n")

    for event in app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values",
    ):
        msg = event["messages"][-1]
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                names = [tc["name"] for tc in msg.tool_calls]
                print(f"🛠️  Agent tools: {names}")
            elif msg.content:
                print("🤖 Final response:")
                print(msg.content)
                print("-" * 40)
