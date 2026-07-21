# 🦙 LlamaIndex Vector Stores PolarDB-X

A powerful integration between LlamaIndex and PolarDB-X, enabling native vector search capabilities for AI applications.

## Overview

LlamaIndex PolarDB-X provides seamless integration between LlamaIndex, a framework for building context-augmented LLM applications, and PolarDB-X with native vector search support. This integration enables efficient vector storage and retrieval for AI applications like semantic search, recommendation systems, and RAG (Retrieval Augmented Generation).

PolarDB-X is a cloud-native distributed database system developed by Alibaba Cloud, featuring native HNSW-based vector index support that delivers high-performance approximate nearest neighbor (ANN) search directly within the database engine.

## Requirements

- Python 3.10+
- PolarDB-X with vector index support
- SQLAlchemy: `sqlalchemy>=1.4.0` (included in package dependencies)
- Async support: `aiomysql>=0.2.0` (included in package dependencies)
- MySQL driver: `pymysql>=1.0.0` (included in package dependencies)

### Enable Vector Index

PolarDB-X disables the vector index feature by default (`vidx_disabled = ON`). You need to enable it before using this package:

```sql
-- Enable vector index (run as admin/root on DN node)
SET GLOBAL vidx_disabled = OFF;
```

This setting takes effect immediately for new connections. No restart required.

All transaction isolation levels (READ-COMMITTED, REPEATABLE-READ, SERIALIZABLE) are supported — choose according to your business needs.

## Features

- **Native Vector Storage**: Store embeddings using PolarDB-X's native `VECTOR(N)` data type
- **HNSW Index**: Efficient approximate nearest neighbor search with configurable `M` and `EF_CONSTRUCTION` parameters
- **Multiple Distance Metrics**: Support for Cosine, Euclidean, and Inner Product distance (v3)
- **Similarity Search**: Perform efficient similarity searches with score thresholds
- **Metadata Filtering**: Filter search results by metadata with rich operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$like`)
- **Dynamic Index Management**: Create, drop, and rebuild vector indexes at runtime without recreating tables
- **Search Mode Control**: Switch between ANN (index-accelerated) and KNN (full-scan) modes per query
- **Per-Query Tuning**: Adjust `ef_search` on a per-query basis for accuracy/latency trade-offs
- **Index Health Monitoring**: Runtime statistics, index health diagnostics, and preload checks (v3)
- **Batch Operations**: Efficient batch insert and bulk upsert with configurable batch size
- **Full Async Support**: All public methods have async equivalents (`async_add`, `aquery`, etc.)
- **Dual-Version Compatibility**: Automatically detects database capabilities and adapts SQL accordingly
- **Connection Pooling**: Built-in SQLAlchemy Engine with connection pooling

## Installation

<!-- PyPI publish pending, use local install for now -->

```bash
# From source (until PyPI release)
pip install -e .
```

<!-- Once published to PyPI:
```bash
pip install -U llama-index-vector-stores-polardbx
```
-->

### Optional Dependencies

For using OpenAI embeddings:

```bash
pip install llama-index-embeddings-openai
```

For using DashScope embeddings (Alibaba Cloud):

```bash
pip install llama-index-embeddings-dashscope
```

## Quick Start

### Basic Usage

```python
from llama_index.vector_stores.polardbx import PolarDBXVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure embedding model
Settings.embed_model = OpenAIEmbedding()

# Create vector store
vector_store = PolarDBXVectorStore(
    host="your-polardbx-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    table_name="my_vectors",
    embed_dim=1536,
    distance_method="COSINE",  # or "EUCLIDEAN", "INNER_PRODUCT" (v3 only)
    default_m=16,  # HNSW index M parameter (3-200)
)

# Create index from documents
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is PolarDB-X?")
print(response)
```

### Using DashScope Embeddings

```python
from llama_index.vector_stores.polardbx import PolarDBXVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding

embed_model = DashScopeEmbedding(
    model_name="text-embedding-v4",
    api_key="your-dashscope-api-key",
)

vector_store = PolarDBXVectorStore(
    host="your-polardbx-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    table_name="langchain_vectors",
    embed_dim=1024,
)
```

## Usage Examples

### Direct Vector Store Operations

```python
from llama_index.vector_stores.polardbx import PolarDBXVectorStore
from llama_index.core.schema import TextNode

vector_store = PolarDBXVectorStore(
    host="your-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    table_name="my_vectors",
    embed_dim=1536,
)

# Add nodes
nodes = [
    TextNode(text="Hello world", embedding=[0.1, 0.2, ...]),
    TextNode(text="PolarDB-X is great", embedding=[0.3, 0.4, ...]),
]
ids = vector_store.add(nodes)

# Query
from llama_index.core.vector_stores.types import VectorStoreQuery
query = VectorStoreQuery(
    query_embedding=[0.1, 0.2, ...],
    similarity_top_k=5,
)
result = vector_store.query(query)
for node, score in zip(result.nodes, result.similarities):
    print(f"[Score: {score:.4f}] {node.text}")
```

### Search with Metadata Filter

```python
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

# Add nodes with metadata
nodes = [
    TextNode(text="Apple is a fruit", embedding=[...], metadata={"category": "fruit", "price": 5}),
    TextNode(text="Banana is yellow", embedding=[...], metadata={"category": "fruit", "price": 3}),
    TextNode(text="Car is a vehicle", embedding=[...], metadata={"category": "vehicle", "price": 20000}),
]
vector_store.add(nodes)

# Filter: category = "fruit" AND price > 2
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="fruit", operator=FilterOperator.EQ),
        MetadataFilter(key="price", value=2, operator=FilterOperator.GT),
    ]
)
query = VectorStoreQuery(query_embedding=[...], similarity_top_k=5, filters=filters)
result = vector_store.query(query)
```

### Search Mode Control

```python
# Force ANN (use vector index for HNSW acceleration)
result = vector_store.query(query, search_type="ann")

# Force KNN (full table scan, bypass vector index)
result = vector_store.query(query, search_type="knn")

# Let the optimizer decide (default)
result = vector_store.query(query, search_type="auto")

# Tune ef_search per query (higher = more accurate, slower)
result = vector_store.query(query, ef_search=100)
```

### Dynamic Vector Index Management

```python
# Create a vector index at runtime
vector_store.apply_vector_index(
    index_name="my_vi",
    m=16,
    distance="COSINE",
    ef_construction=200,  # v3 only, ignored on old versions
)

# Drop the vector index
vector_store.drop_vector_index()

# Rebuild the index to reclaim space and improve recall
vector_store.optimize()
```

### Index Monitoring (v3 only)

```python
# Get runtime statistics
stats = vector_store.get_stats()
print(stats)  # e.g. {"Vidx_query_count": 100, "Vidx_load_node_hits": 950, ...}

# Preload HNSW index into memory cache to eliminate cold-start latency
vector_store.preload_index()

# Check if preloading would fit in cache
check_result = vector_store.preload_check()
print(check_result)

# Diagnose index health
health = vector_store.explain_index_health()
print(health)
```

### Delete and Manage Nodes

```python
# Delete by ref_doc_id
vector_store.delete(ref_doc_id="doc-001")

# Delete by node_ids
vector_store.delete_nodes(node_ids=["node-1", "node-2"])

# Get nodes by node_ids
nodes = vector_store.get_nodes(node_ids=["node-1", "node-2"])

# Count vectors
count = vector_store.count()

# Clear all data (TRUNCATE TABLE)
vector_store.clear()

# Drop the entire table
vector_store.drop()
```

### Async API

All public methods have async equivalents:

```python
import asyncio

async def main():
    # Add nodes
    ids = await vector_store.async_add(nodes)

    # Query
    result = await vector_store.aquery(query)

    # Delete
    await vector_store.adelete(ref_doc_id="doc-001")

    # Delete nodes
    await vector_store.adelete_nodes(node_ids=["node-1"])

    # Count
    count = await vector_store.acount()

    # Clear
    await vector_store.aclear()

    # Dynamic index management
    await vector_store.aapply_vector_index(index_name="vi", m=16)
    await vector_store.adrop_vector_index()
    await vector_store.aoptimize()

    # v3 monitoring
    stats = await vector_store.aget_stats()
    await vector_store.apreload_index()
    health = await vector_store.aexplain_index_health()

asyncio.run(main())
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | - | PolarDB-X host address |
| `port` | int | - | PolarDB-X port number |
| `user` | str | - | Username |
| `password` | str | - | Password |
| `database` | str | - | Database name |
| `table_name` | str | `"llama_index_table"` | Table name for vector storage |
| `embed_dim` | int | 1536 | Embedding dimension |
| `distance_method` | str | `"COSINE"` | Distance function: `"COSINE"`, `"EUCLIDEAN"`, or `"INNER_PRODUCT"` (v3) |
| `default_m` | int | 6 | HNSW index M parameter (3-200) |
| `perform_setup` | bool | True | Whether to auto-create table on init |
| `debug` | bool | False | Enable SQLAlchemy echo mode |
| `ef_construction` | int | None | HNSW build-time candidate list size (5-1000, v3 only) |
| `vector_index_name` | str | None | Vector index name for FORCE INDEX hints (auto-detected if None) |

## PolarDB-X Vector Functions Used

This integration uses PolarDB-X's native vector functions:

- `VECTOR(N)` — Vector column data type with N dimensions
- `VEC_FROMTEXT('[1,2,3]')` — Convert JSON array string to vector
- `VEC_TOTEXT(vector)` — Convert vector to JSON array string
- `VEC_DISTANCE(v1, v2)` — Auto-inferred distance function (v3)
- `VEC_DISTANCE_COSINE(v1, v2)` — Cosine distance (old versions)
- `VEC_DISTANCE_EUCLIDEAN(v1, v2)` — Euclidean distance (old versions)
- `VEC_DISTANCE_INNER_PRODUCT(v1, v2)` — Inner product distance (old versions)
- `VECTOR_DIM(v)` — Get vector dimension (v3)
- `VECTOR INDEX (col) M=N DISTANCE=COSINE` — HNSW vector index DDL
- `EF_CONSTRUCTION=N` — HNSW build-time parameter in DDL (v3)
- `SET SESSION vidx_hnsw_ef_search = N` — Per-session search width tuning
- `SHOW GLOBAL STATUS LIKE 'Vidx%'` — Runtime index statistics
- `CALL dbms_vidx.preload(db, table, col)` — Preload index into cache (v3)
- `CALL dbms_vidx.preload_check(db, table, col)` — Check preload feasibility (v3)
- `information_schema.VECTOR_INDEXES` — Vector index metadata view (v3)

## Development

This package uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync --group dev

# Run tests
pytest -v

# Lint
ruff check .
```

## License

MIT
