"""Central configuration + shared Azure OpenAI factories.

Everything reads from environment variables (loaded from .env). Keeping the
LLM/embeddings construction in one place means every agent shares the same
Azure deployment settings and we can swap models in a single spot.
"""
from __future__ import annotations
import os
import functools

from dotenv import load_dotenv

load_dotenv()


# ── Paths / tuning ───────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chroma")
SCAN_WORKSPACE_DIR = os.getenv("SCAN_WORKSPACE_DIR", "./.scan_workspace")
MAX_FILE_SIZE_KB = int(os.getenv("MAX_FILE_SIZE_KB", "512"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "12"))

# ── Azure deployments ────────────────────────────────────────────────────────
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# ── Embeddings provider ──────────────────────────────────────────────────────
# "azure" (default) uses Azure OpenAI embeddings; "huggingface" runs a local
# sentence-transformers model (no API calls, free) such as all-MiniLM-L6-v2.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "azure").lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@functools.lru_cache(maxsize=4)
def get_chat_llm(max_tokens: int = 4000, json_mode: bool = True):
    """Return a cached AzureChatOpenAI client.

    json_mode forces the model to emit a JSON object, which every agent relies
    on to parse structured findings without regex scraping.
    """
    from langchain_openai import AzureChatOpenAI

    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        max_tokens=max_tokens,
        model_kwargs=kwargs,
    )


@functools.lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached embeddings client used to index the repo.

    Provider is selected via EMBEDDING_PROVIDER: "azure" (Azure OpenAI) or
    "huggingface" (local sentence-transformers, no API calls).
    """
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
