# RAG-agent – kan läsa och söka i lokala dokument
#
# Denna agent använder Retrieval Augmented Generation (RAG) för att svara på frågor.
# Den laddar dokument från mappen "documents/", delar upp dem i mindre delar
# och skapar embeddings som sparas i en vektordatabas (FAISS).
#
# När användaren ställer en fråga:
# 1. Agenten söker efter relevant information i dokumenten
# 2. Hämtar de mest relevanta textdelarna
# 3. Genererar ett svar baserat på dessa
#
# Agenten svarar endast utifrån dokumenten och gissar inte.

import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from util.embeddings import get_embeddings
from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input


def load_documents(directory_path: str):
    if not os.path.exists(directory_path):
        return None

    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )

    docs = loader.load()
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(docs)

    embeddings = get_embeddings()
    return FAISS.from_documents(splits, embeddings)


# Global variable (needed by tool)
vector_store = None


@tool
def search_documents(query: str) -> str:
    """Sök i lokala dokument och returnera relevant information."""
    if vector_store is None:
        return "Inga dokument hittades."

    docs = vector_store.similarity_search(query, k=3)

    return "\n\n".join(doc.page_content for doc in docs)


def run():
    global vector_store

    # Load documents
    documents_path = os.path.join(os.getcwd(), "documents")
    vector_store = load_documents(documents_path)

    # Model
    model = get_model(temperature=0.1, top_p=0.5)

    # Only RAG tool
    tools = [search_documents]

    # Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "Du är en AI-agent som kan söka i lokala dokument. "
            "När användaren ställer en fråga ska du använda verktyget 'search_documents' "
            "för att hitta relevant information. "
            "Svara endast baserat på dokumenten. "
            "Om du inte hittar relevant information, säg det tydligt. "
            "Svara alltid på svenska och var tydlig och koncis."
        ),
    )

    messages = []

    while True:
        user_input = get_user_input("Ställ din fråga")

        if not user_input:
            continue
        if user_input in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        process_stream = agent.stream(
            {"messages": messages},
            stream_mode=STREAM_MODES,
        )

        handle_stream(process_stream)


if __name__ == "__main__":
    run()