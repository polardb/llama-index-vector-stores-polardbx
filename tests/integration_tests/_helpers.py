"""Shared test helpers for LlamaIndex PolarDBXVectorStore tests."""

import hashlib
import os
import uuid
from urllib.parse import urlparse

import numpy as np
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from llama_index.vector_stores.polardbx import PolarDBXVectorStore

# ---- .env loading (search upward from this file) ----


def _load_dotenv():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        env_path = os.path.join(d, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            return
        d = os.path.dirname(d)


_load_dotenv()

_uri = urlparse(os.environ.get("POLARDBX_URI", ""))
DB_HOST = _uri.hostname or "localhost"
DB_PORT = _uri.port or 3306
DB_USER = _uri.username or "root"
DB_PASS = _uri.password or ""
DB_NAME = _uri.path.lstrip("/") or "test"

# DN node name for LOCALITY tests (instance-specific)
DN_NODE = os.environ.get("POLARDBX_DN_NODE", "")

EMBED_DIM = 128


def uri() -> str:
    """Build a mysql+pymysql URI from .env credentials."""
    from urllib.parse import quote_plus

    return f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ---- Fake embeddings ----


class FakeEmbeddings:
    """Deterministic embedding using MD5 hash — same text always yields same vector."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).digest()
        vec = np.frombuffer(h * (self.dim // 16 + 1), dtype=np.uint8)[: self.dim]
        return (vec / 255.0).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


EMB = FakeEmbeddings(dim=EMBED_DIM)

# ---- Test data ----

TEXTS = [
    "PolarDB-X is a distributed database",
    "LlamaIndex is a framework for building RAG applications",
    "Vector search uses cosine similarity for relevance",
    "HNSW index provides fast approximate nearest neighbor search",
    "Python is a popular programming language for data science",
]

METADATAS = [
    {"category": "database", "lang": "en"},
    {"category": "framework", "lang": "en"},
    {"category": "search", "lang": "en"},
    {"category": "index", "lang": "en"},
    {"category": "language", "lang": "en"},
]


# ---- Node factory ----


def make_nodes(
    texts: list[str] | None = None,
    metadatas: list[dict] | None = None,
    emb: FakeEmbeddings | None = None,
    ref_doc_ids: list[str] | None = None,
) -> list[TextNode]:
    """Create TextNode objects with pre-computed embeddings.

    Args:
        texts: List of text strings. Defaults to TEXTS.
        metadatas: List of metadata dicts. Defaults to METADATAS.
        emb: Embedding function. Defaults to EMB.
        ref_doc_ids: Optional list of ref_doc_id strings to set on nodes.
    """
    if texts is None:
        texts = TEXTS
    if metadatas is None:
        metadatas = METADATAS
    if emb is None:
        emb = EMB

    nodes = []
    for i, text in enumerate(texts):
        meta = metadatas[i] if i < len(metadatas) else {}
        node = TextNode(
            id_=str(uuid.uuid4()),
            text=text,
            metadata=meta,
            embedding=emb.embed_query(text),
        )
        ref_id = ref_doc_ids[i] if ref_doc_ids and i < len(ref_doc_ids) else node.node_id
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref_id)
        nodes.append(node)
    return nodes


# ---- Store factory ----


def make_store(
    table_name: str = "test_polardbx_llamaindex",
    distance_method: str = "COSINE",
    pre_delete: bool = True,
    **kwargs,
) -> PolarDBXVectorStore:
    """Create a PolarDBXVectorStore for testing.

    If pre_delete is True, drops any existing table before creating the store.
    """
    if pre_delete:
        _drop_table_safely(table_name)

    return PolarDBXVectorStore(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        table_name=table_name,
        embed_dim=kwargs.pop("embed_dim", EMBED_DIM),
        default_m=kwargs.pop("default_m", 6),
        distance_method=distance_method,
        perform_setup=kwargs.pop("perform_setup", True),
        debug=kwargs.pop("debug", False),
        **kwargs,
    )


def _drop_table_safely(table_name: str) -> None:
    """Drop a table if it exists, swallowing errors."""
    import sqlalchemy
    from urllib.parse import quote_plus

    try:
        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        with engine.connect() as conn:
            conn.execute(
                sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`")
            )
            conn.commit()
        engine.dispose()
    except Exception:
        pass


def is_v3(store: PolarDBXVectorStore) -> bool:
    """Check if the connected instance supports v3 vector features.

    Uses vec_dim (VECTOR_DIM) as the v3 indicator — present iff the
    instance has v3 vector features (EF_CONSTRUCTION DDL, INNER_PRODUCT,
    dbms_vidx procedures, etc.).
    """
    return store._capabilities.get("vec_dim", False)
