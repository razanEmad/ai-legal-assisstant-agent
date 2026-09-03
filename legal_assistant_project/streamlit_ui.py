"""
streamlit_ui.py
===============
User interface for the AI Legal Assistant
Supports two modes:
    1. Agentic Workflow (LangGraph) - the agent chooses tools
    2. RAG Direct (LCEL) - direct search in the knowledge base
"""

import uuid
from operator import itemgetter

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver

from agentic_workflow import (
    create_workflow,
    DEMO_CONTRACT,
    get_retriever,
    llm,
    is_api_key_valid,
)


def normalize_content(content) -> str:
    """Extract readable text from plain or block-based model content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            normalize_content(block)
            for block in content
            if normalize_content(block)
        )
    if isinstance(content, dict):
        return normalize_content(content.get("text", ""))
    return str(content) if content else ""


# ─── Page configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════
# Cached resources - built once and stored for the application's lifetime
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_workflow_app():
    """
    Build the LangGraph app once and keep it cached.
    MemorySaver stores checkpoints in RAM; there is no database.
    """
    workflow = create_workflow()
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


@st.cache_resource
def get_rag_chain():
    """
    Build a direct RAG chain (LCEL) without an agent.

        Chain:
            question -> retriever -> format_docs -> PromptTemplate -> LLM -> StrOutputParser
    """
    if not is_api_key_valid:
        return None

    try:
        retriever = get_retriever()

        def format_docs(docs):
            """Join chunk contents into one text separated by blank lines."""
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = PromptTemplate.from_template(
            "You are an AI legal assistant specializing in Egyptian labor law.\n"
            "Answer the following question using only the provided legal context.\n"
            "If the answer is not in the context, say so explicitly.\n\n"
            "Legal context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        # LCEL pipe connects each step to the next using |
        rag_chain = (
            {
                # itemgetter("question") extracts the question, then passes it
                # to the retriever and format_docs
                "context": itemgetter("question") | retriever | format_docs,
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain
    except Exception as e:
        # Log the error and return None
        print(f"Error initializing RAG chain: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Session state - preserves data between Streamlit reruns
# ═══════════════════════════════════════════════════════════════════════════
if "thread_id" not in st.session_state:
    # Each session has a unique thread_id and independent MemorySaver memory
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "contract_text" not in st.session_state:
    st.session_state.contract_text = ""

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚖️ AI Legal Assistant")
    st.caption("AI Legal Assistant — Powered by Gemini + LangGraph")

    st.markdown("---")

    # ─── Mode selection ──────────────────────────────────────────────────
    st.markdown("### ⚙️ Operating mode")
    mode = st.radio(
        "How should the assistant process your question?",
        [
            "🤖 Agentic Workflow (LangGraph)",
            "📚 RAG Direct (LCEL)",
        ],
        help=(
            "Agentic: the LLM chooses which tools to use.\n"
            "RAG Direct: searches the legal knowledge base directly."
        ),
    )

    st.markdown("---")

    # ─── Graph structure ─────────────────────────────────────────────────
    st.markdown("### 🗺️ Graph structure")
    st.code(
        "START\n"
        "  ↓\n"
        "[agent_node]\n"
        "  ↓\n"
        "should_continue()\n"
        "  ├─ tools → [ToolNode] ─┐\n"
        "  │                      │\n"
        "  │         ┌────────────┘\n"
        "  │         ↓\n"
        "  │     [agent_node]\n"
        "  └─ END",
        language="text",
    )

    st.markdown("---")

    # ─── Available tools ─────────────────────────────────────────────────
    st.markdown("### 🔧 Available tools")
    st.markdown("""
1. `pdf_parser` - contract structure and section count
2. `clause_detector` - detect key clauses
3. `risk_analyzer` - risk analysis (🔴🟡🟢)
4. `legal_knowledge_retriever` - RAG knowledge base
5. `web_search` - internet search
    """)

    st.markdown("---")

    # ─── Technology stack ─────────────────────────────────────────────────
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
- **LangChain** — Tools, Prompts, LCEL
- **LangGraph** — StateGraph, ToolNode, MemorySaver
- **RAG** — TextLoader, FAISS, Embeddings
- **Gemini 2.5 Flash** — LLM
- **Streamlit** — UI
    """)

    st.markdown("---")

    # ─── Clear conversation ───────────────────────────────────────────────
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.contract_text = ""
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Main page
# ═══════════════════════════════════════════════════════════════════════════
st.title("🤖 AI Legal Assistant")
st.caption("Contract analysis, risk detection, and legal guidance")

if not is_api_key_valid:
    st.warning(
        "⚠️ **Invalid or missing API key!** Please update the `.env` file and set the `GOOGLE_API_KEY` variable with your Gemini API key to run the legal assistant.\n\n"
        "⚠️ **Invalid or missing API key!** Please update the `.env` file and set the `GOOGLE_API_KEY` variable with your Gemini API key to run the legal assistant."
    )

# ─── Quick actions ───────────────────────────────────────────────────────
st.markdown("### ⚡ Quick actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📄 Load demo contract", use_container_width=True):
        st.session_state.contract_text = DEMO_CONTRACT
        st.rerun()

with col2:
    if st.button("🔍 Full analysis", use_container_width=True):
        st.session_state.quick_prompt = (
            "Analyze the contract step by step: extract its structure, detect clauses, and analyze risks."
        )

with col3:
    if st.button("⚠️ Risks only", use_container_width=True):
        st.session_state.quick_prompt = (
            "Analyze the contract risks and classify them as (HIGH / MEDIUM / LOW)."
        )

with col4:
    if st.button("📋 Extract clauses", use_container_width=True):
        st.session_state.quick_prompt = (
            "Extract all key clauses from the contract."
        )

# ─── Contract text ────────────────────────────────────────────────────────
st.markdown("### 📝 Contract text")
contract_text = st.text_area(
    "Enter the contract text here (or use the button above to load a demo contract):",
    value=st.session_state.contract_text,
    height=180,
    placeholder="Type or paste the contract text here...",
)
st.session_state.contract_text = contract_text

st.markdown("---")

# ─── Conversation history ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(normalize_content(msg["content"]))
        if msg.get("tools_used"):
            st.caption(f"🛠️ Tools used: `{'`, `'.join(msg['tools_used'])}`")

# ─── Chat input ───────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a legal question or request contract analysis...", disabled=not is_api_key_valid)

# Handle quick-action buttons
if "quick_prompt" in st.session_state:
    prompt = st.session_state.pop("quick_prompt")

# ═══════════════════════════════════════════════════════════════════════════
# Process the question
# ═══════════════════════════════════════════════════════════════════════════
if prompt:
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add the contract text to the question when provided
    full_query = prompt
    if contract_text.strip():
        full_query = f"{prompt}\n\nContract text:\n{contract_text.strip()}"

    with st.chat_message("assistant"):
        with st.spinner("The legal assistant is thinking..."):

            # ─── Mode 1: Agentic Workflow ─────────────────────────────────
            if "Agentic" in mode:
                app = get_workflow_app()
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                used_tools = []
                final_answer = ""

                for event in app.stream(
                    {"messages": [HumanMessage(content=full_query)]},
                    config,
                    stream_mode="values",
                ):
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        if last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                used_tools.append(tc["name"])
                        elif last_msg.content:
                            final_answer = normalize_content(last_msg.content)

                st.markdown(final_answer)
                used_tools = list(dict.fromkeys(used_tools))  # Remove duplicates while preserving order
                if used_tools:
                    st.caption(f"🛠️ Tools used: `{'`, `'.join(used_tools)}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "tools_used": used_tools,
                })

            # ─── Mode 2: RAG Direct (LCEL) ────────────────────────────────
            else:
                rag_chain = get_rag_chain()
                if rag_chain is None:
                    response = "⚠️ Direct RAG cannot run without a valid API key. Update `.env` with your Gemini API key in `GOOGLE_API_KEY`."
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tools_used": [],
                    })
                else:
                    response = normalize_content(rag_chain.invoke({"question": full_query}))
                    st.markdown(response)
                    st.caption("🛠️ Tool used: `legal_knowledge_retriever` (RAG Direct)")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "tools_used": ["legal_knowledge_retriever"],
                    })
