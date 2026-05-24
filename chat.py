"""
SOP Assistant — Interactive CLI
Loads a FAISS index and answers questions with cited sources.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

load_dotenv()

# --- Configuration ---
INDEX_PATH = "db/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
TOP_K = 3

PROMPT_TEMPLATE = """You are an expert SOP (Standard Operating Procedure) assistant.
Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain the answer, reply exactly: "I couldn't find this in the provided SOPs."
- Be concise and factual. Use numbered steps when describing procedures.
- Do NOT invent steps, numbers, names, or policies that aren't in the context.

Context:
{context}

Question: {input}

Answer:"""


def build_chain():
    if not HF_TOKEN:
        sys.exit("❌ HUGGINGFACEHUB_API_TOKEN missing. Copy .env.example → .env and fill it in.")
    if not Path(INDEX_PATH).exists():
        sys.exit("❌ FAISS index not found. Run `python index.py` first.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # Safe: we own this index
    )

    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return create_retrieval_chain(retriever, document_chain)


def format_sources(docs) -> str:
    """Deduplicate (file, page) pairs and format for display."""
    seen, lines = set(), []
    for d in docs:
        src = d.metadata.get("source_file") or d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        loc = src + (f" — page {page + 1}" if isinstance(page, int) else "")
        lines.append(f"  • {loc}")
    return "\n".join(lines) if lines else "  (none)"


def run_chat():
    chain = build_chain()
    print("\n" + "=" * 60)
    print(" 📚  SOP Assistant — ask a question (type 'exit' to quit)")
    print("=" * 60)

    while True:
        try:
            query = input("\n🔍 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            print("Goodbye!")
            return

        try:
            result = chain.invoke({"input": query})
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue

        print(f"\n🤖 Answer:\n{result['answer'].strip()}")
        print(f"\n📚 Sources:\n{format_sources(result['context'])}")


if __name__ == "__main__":
    run_chat()