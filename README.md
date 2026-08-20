# 🦙 LlamaIndex Vector Stores PolarDB-X

A powerful integration between LlamaIndex and PolarDB-X, enabling native vector search and SQL query capabilities for AI applications.

## Overview

LlamaIndex PolarDB-X provides seamless integration between LlamaIndex, a framework for building context-augmented LLM applications, and PolarDB-X with native vector search support. This integration enables efficient vector storage and retrieval for AI applications like semantic search, recommendation systems, and RAG (Retrieval Augmented Generation).

PolarDB-X is a cloud-native distributed database system developed by Alibaba Cloud, featuring native HNSW-based vector index support that delivers high-performance approximate nearest neighbor (ANN) search directly within the database engine.

In addition to vector search, this package provides a `PolarDBXSQLDatabase` wrapper that enables LlamaIndex SQL query engines (e.g. `NLSQLTableQueryEngine`) to work seamlessly with PolarDB-X, and a `create_partitioned_table` helper for creating non-vector partitioned tables for distributed data scenarios.

## Requirements

- Python 3.10+
- PolarDB-X with vector index support
- LlamaIndex core: `llama-index-core>=0.13.0,<0.15` (included in package dependencies)
- SQLAlchemy: `sqlalchemy>=2.0.0` (included in package dependencies)
- Async support: `aiomysql>=0.2.0` (included in package dependencies)
- MySQL driver: `pymysql>=1.0.0` (included in package dependencies)
- NumPy: `numpy>=1.24.0` (included in package dependencies, required for MMR search)

### Enable Vector Index

PolarDB-X disables the vector index feature by default (`vidx_disabled = ON`). You need to enable it before using this package:

```sql
-- Enable vector index (run as admin/root on DN node)
SET GLOBAL vidx_disabled = OFF;
```

This setting takes effect immediately for new connections. No restart required.

All transaction isolation levels (READ-COMMITTED, REPEATABLE-READ, SERIALIZABLE) are supported — choose according to your business needs.

> **Note**: Some advanced features (e.g., inner product distance, index monitoring, `EF_CONSTRUCTION` parameter) require newer PolarDB-X versions. The package automatically detects available capabilities and adapts accordingly.

## Features

- **Native Vector Storage**: Store embeddings using PolarDB-X's native `VECTOR(N)` data type
- **HNSW Index**: Efficient approximate nearest neighbor search with configurable `M` and `EF_CONSTRUCTION` parameters
- **Multiple Distance Metrics**: Support for Cosine, Euclidean, and Inner Product distance (v3)
- **Similarity Search**: Perform efficient similarity searches with configurable top-k
- **MMR Search**: Maximal Marginal Relevance search for diverse results
- **Metadata Filtering**: Filter search results by metadata with rich operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`)
- **Metadata-Based Operations**: Search and delete nodes by metadata conditions without vector similarity
- **Dynamic Index Management**: Create, drop, and rebuild vector indexes at runtime without recreating tables
- **Search Mode Control**: Switch between ANN (index-accelerated) and KNN (full-scan) modes per query
- **Per-Query Tuning**: Adjust `ef_search` on a per-query basis for accuracy/latency trade-offs
- **Index Health Monitoring**: Runtime statistics, index health diagnostics, and preload checks (v3)
- **Batch Operations**: Batch insert with UPSERT support within a single transaction
- **Full Async Support**: All public methods have async equivalents (`async_add`, `aquery`, etc.)
- **Dual-Version Compatibility**: Automatically detects database capabilities and adapts SQL accordingly
- **Partitioned Table Support**: Create partitioned vector tables with HASH/KEY/RANGE/LIST strategies, broadcast tables, and LOCALITY node assignment
- **Custom Column Schema**: Customize column names for all core fields (`id`, `node_id`, `text`, `embedding`, `metadata`) and promote specific metadata keys to dedicated typed columns for faster filtering and indexing
- **Connection Pooling**: Built-in SQLAlchemy Engine with connection pooling and automatic retry logic
- **SQL Database Integration**: Use PolarDB-X as a SQL database for LlamaIndex SQL agents with automatic DDL reflection compatibility (tab indentation, ENUM spacing, VECTOR type support)

## Installation

```bash
pip install -U llama-index-vector-stores-polardbx
```

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

### Using from_params Factory Method

```python
from llama_index.vector_stores.polardbx import PolarDBXVectorStore

vector_store = PolarDBXVectorStore.from_params(
    host="your-polardbx-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    table_name="my_vectors",
    embed_dim=1536,
    distance_method="COSINE",
    ssl=True,           # Enable TLS
    ssl_ca="/path/to/ca.pem",  # CA certificate
)
```

`from_params` accepts all the same parameters as the constructor, including custom column parameters (`id_column`, `metadata_columns`, etc.). See [Configuration Options](#configuration-options) for the full list.

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
    table_name="my_vectors",
    embed_dim=1024,
)
```

## Custom Column Schema

By default, the vector store creates a table with five columns: `id`, `node_id`, `text`, `metadata` (JSON), and `embedding` (VECTOR). You can customize column names and promote specific metadata keys to dedicated typed columns for faster filtering and indexing.

### Customizing Core Column Names

```python
from llama_index.vector_stores.polardbx import PolarDBXVectorStore

vector_store = PolarDBXVectorStore(
    host="your-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    table_name="my_vectors",
    embed_dim=1536,
    id_column="my_id",                    # default: "id"
    node_id_column="my_node_id",           # default: "node_id"
    text_column="my_text",                 # default: "text"
    embedding_column="my_embedding",       # default: "embedding"
    metadata_json_column="my_meta",        # default: "metadata"; set None to disable
)
```

### Promoting Metadata to Dedicated Columns

By default, all metadata is stored as a single JSON column. You can promote specific metadata keys to dedicated typed columns for faster filtering (direct column reference instead of `JSON_EXTRACT`) and native indexing.

Use `Column` objects when creating a new table (the `data_type` is included in DDL):

```python
from llama_index.vector_stores.polardbx import PolarDBXVectorStore, Column

vector_store = PolarDBXVectorStore(
    host="your-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    table_name="my_vectors",
    embed_dim=1536,
    metadata_columns=[
        Column("category", "VARCHAR(64)", nullable=False),
        Column("lang", "VARCHAR(8)"),
        Column("price", "DECIMAL(10,2)"),
    ],
)
```

Use plain strings when connecting to an existing table (data type is ignored, taken from the existing schema):

```python
vector_store = PolarDBXVectorStore(
    host="your-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    table_name="existing_table",
    embed_dim=1536,
    perform_setup=False,              # do not auto-create the table
    metadata_columns=["category", "lang", "price"],
)
```

### How Mapped Columns Work

When metadata columns are configured:

- **On insert**: Metadata keys matching mapped column names are extracted to their own columns. Remaining metadata keys (including LlamaIndex internal serialization data) are stored in the JSON column.
- **On query**: Mapped column values and JSON column values are merged into the metadata dict returned to LlamaIndex.
- **On filter**: Filters on mapped column keys use direct column references (`WHERE \`category\` = ?`) for better performance. Filters on non-mapped keys use `JSON_EXTRACT` on the JSON column.

### No JSON Column Mode

Set `metadata_json_column=None` to store metadata exclusively in mapped columns, eliminating the JSON column entirely. This is useful for schemas where all metadata keys are known upfront.

```python
vector_store = PolarDBXVectorStore(
    ...,
    metadata_json_column=None,
    metadata_columns=["category", "lang"],
)
```

> **Note**: When `metadata_json_column=None`, filtering on a metadata key that is not a mapped column raises `ValueError`. Ensure all filterable keys are listed in `metadata_columns`.

> **Note**: In no-JSON-column mode, LlamaIndex internal serialization data (e.g., `_node_content`) is not stored. Node reconstruction falls back to creating a basic `TextNode` from the text and mapped metadata columns. This is sufficient for search and retrieval but does not preserve node relationships or other internal metadata.

> **Note**: In no-JSON-column mode, `delete(ref_doc_id)` and `adelete(ref_doc_id)` raise `ValueError` because `ref_doc_id` is stored in the JSON metadata column. Use `delete_nodes(filters=...)` or `delete_by_metadata(filters=...)` instead, or add `ref_doc_id` to `metadata_columns` to enable filtering on a dedicated column.

### Column Class Reference

The `Column` dataclass defines metadata column schema for new table creation:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | (required) | Column name (must match a metadata key) |
| `data_type` | str | (required) | SQL type, e.g. `"VARCHAR(255)"`, `"INT"`, `"DECIMAL(10,2)"` |
| `nullable` | bool | `True` | Whether the column allows NULL values |
| `default` | str | `None` | SQL default expression for DDL, e.g. `"0"`, `"'active'"` |

> **Note**: `data_type` and `default` are validated to prevent SQL injection (no semicolons, comments, or newlines). Only SQL type expressions are allowed.

> **Note**: `default` only affects the DDL schema definition (`DEFAULT` clause in `CREATE TABLE`). It does not auto-fill missing values during `INSERT` through this package. When a metadata key is absent, `nullable=False` columns raise `ValueError`, and `nullable=True` columns insert `NULL`. To auto-fill defaults, provide the value in the metadata dict before calling `add()`.

## SQL Database

`PolarDBXSQLDatabase` wraps LlamaIndex's `SQLDatabase` with PolarDB-X-specific DDL reflection fixes, enabling seamless use with LlamaIndex SQL query engines (e.g. `NLSQLTableQueryEngine`). It automatically:

- Normalizes tab indentation in `SHOW CREATE TABLE` output (PolarDB-X uses tabs, standard MySQL uses two spaces)
- Fixes ENUM/SET value list spacing (`enum('A', 'B')` to `enum('A','B')`)
- Registers a custom `VECTOR` type so tables with vector columns do not crash reflection
- Auto-swaps `mysql+pymysql://` URIs to use the PolarDB-X dialect

```python
from llama_index.vector_stores.polardbx import PolarDBXSQLDatabase

db = PolarDBXSQLDatabase.from_uri(
    "mysql+pymysql://user:password@host:3306/your-database"
)

# List tables
tables = db.get_usable_table_names()

# Get table schema info for SQL query engines
info = db.get_single_table_info("your_table")

# Run SQL queries
result = db.run_sql("SELECT COUNT(*) FROM your_table")
```

### Usage with NLSQLTableQueryEngine

```python
from llama_index.core import Settings
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.vector_stores.polardbx import PolarDBXSQLDatabase

db = PolarDBXSQLDatabase.from_uri(
    "mysql+pymysql://user:password@host:3306/your-database"
)

query_engine = NLSQLTableQueryEngine(
    sql_database=db,
    tables=["orders", "customers"],
)

response = query_engine.query("How many orders were placed last month?")
print(response)
```

> **Note**: `VECTOR INDEX` definitions in `SHOW CREATE TABLE` are not parsed by SQLAlchemy and will be silently skipped with a warning. This is expected — the index info is not needed for SQL query generation. Tables with `VECTOR` columns are fully supported.

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
from llama_index.core.vector_stores import VectorStoreQuery
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
from llama_index.core.vector_stores import (
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

Supported filter operators:

| Operator | SQL | Description |
|----------|-----|-------------|
| `$eq` | `=` | Equal (default for simple values) |
| `$ne` | `!=` | Not equal |
| `$gt` | `>` | Greater than |
| `$gte` | `>=` | Greater than or equal |
| `$lt` | `<` | Less than |
| `$lte` | `<=` | Less than or equal |
| `$in` | `IN` | In a list of values |
| `$nin` | `NOT IN` | Not in a list of values |

### MMR Search (Maximal Marginal Relevance)

```python
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode

# MMR search for diverse results
query = VectorStoreQuery(
    query_embedding=[...],
    similarity_top_k=4,
    mode=VectorStoreQueryMode.MMR,
)
result = vector_store.query(
    query,
    fetch_k=20,         # Candidates to fetch before re-ranking
    lambda_mult=0.5,    # 0 = max diversity, 1 = max relevance
)
```

### Search Mode Control

```python
from llama_index.core.vector_stores import VectorStoreQuery

query = VectorStoreQuery(
    query_embedding=[0.1, 0.2, ...],
    similarity_top_k=10,
)

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
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

# Delete by ref_doc_id
vector_store.delete(ref_doc_id="doc-001")

# Delete by node_ids
vector_store.delete_nodes(node_ids=["node-1", "node-2"])

# Delete by metadata filters (alternative to node_ids)
filters = MetadataFilters(filters=[
    MetadataFilter(key="status", value="deleted", operator=FilterOperator.EQ),
])
vector_store.delete_nodes(filters=filters)

# Get nodes by node_ids
nodes = vector_store.get_nodes(node_ids=["node-1", "node-2"])

# Get nodes by metadata filters (alternative to node_ids)
nodes = vector_store.get_nodes(filters=filters)

# Count vectors
count = vector_store.count()

# Clear all data (TRUNCATE TABLE)
vector_store.clear()

# Drop the entire table
vector_store.drop()

# Search nodes by metadata only (no vector similarity)
nodes = vector_store.search_by_metadata(filters=filters, limit=10)

# Delete nodes matching metadata conditions
deleted_count = vector_store.delete_by_metadata(filters=filters)
print(f"Deleted {deleted_count} nodes")

# Close connections
vector_store.close()
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

    # Get nodes
    nodes = await vector_store.aget_nodes(node_ids=["node-1", "node-2"])

    # Count
    count = await vector_store.acount()

    # Clear
    await vector_store.aclear()

    # Drop the entire table
    await vector_store.adrop()

    # Metadata-based operations
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
    filters = MetadataFilters(filters=[
        MetadataFilter(key="category", value="fruit", operator=FilterOperator.EQ),
    ])
    nodes = await vector_store.asearch_by_metadata(filters=filters, limit=10)
    deleted = await vector_store.adelete_by_metadata(filters=filters)

    # Dynamic index management
    await vector_store.aapply_vector_index(index_name="vi", m=16)
    await vector_store.adrop_vector_index()
    await vector_store.aoptimize()

    # v3 monitoring
    stats = await vector_store.aget_stats()
    await vector_store.apreload_index()
    check = await vector_store.apreload_check()
    health = await vector_store.aexplain_index_health()

    # Close connections
    await vector_store.aclose()

asyncio.run(main())
```

## Partitioned Tables

PolarDB-X is a distributed database that supports table partitioning for scalability. This package supports creating partitioned vector tables and standalone partitioned tables.

### Vector Store with Partitioning

```python
from llama_index.vector_stores.polardbx import PolarDBXVectorStore

# HASH partitioning (8 partitions on the id column)
vector_store = PolarDBXVectorStore(
    host="your-host", port=3306, user="your-user", password="your-password",
    database="your-database", table_name="partitioned_vectors",
    embed_dim=1536,
    partition_by="HASH",          # "HASH", "KEY", "RANGE", or "LIST"
    partition_column="id",        # column to partition on (default: "id")
    partitions=8,                 # number of partitions (HASH/KEY only)
)

# Broadcast table (full copy on every DN node)
vector_store = PolarDBXVectorStore(
    ..., broadcast=True,
)

# RANGE partitioning
vector_store = PolarDBXVectorStore(
    ..., partition_by="RANGE", partition_column="id",
    partition_defs=[
        {"name": "p0", "values_less_than": 1000},
        {"name": "p1", "values_less_than": "MAXVALUE"},
    ],
)

# With LOCALITY (pin table to a specific DN node)
vector_store = PolarDBXVectorStore(
    ..., locality="dn=your-dn-node-name",
)
```

> **Note**: Partitioned vector tables are not supported on certain PolarDB-X v3 instances. The package automatically detects this and raises `NotSupportedError` if you attempt to use partitioning on an incompatible version.
>
> **Note**: When partitioning is enabled, the `node_id` UNIQUE INDEX is automatically downgraded to a regular INDEX. PolarDB-X requires that unique indexes include the partition key, and since the `node_id` column (or custom `node_id_column`) is not the partition key, a UNIQUE constraint would be incompatible with partitioning. The `add()` method automatically adapts: non-partitioned tables use `ON DUPLICATE KEY UPDATE`, while partitioned tables use DELETE-then-INSERT to preserve upsert semantics.
>
> **Note**: For vector tables, `partition_column` must match the `id_column` value (default: `"id"`). This is enforced at init time. Use `create_partitioned_table()` if you need to partition on a different column.
>
> **Note**: LIST partitioning is generally not practical for VectorStore tables because the `id` column is a UUID string — LIST requires exact value enumeration, which is infeasible for UUIDs. Use HASH or KEY partitioning for VectorStore tables instead. LIST partitioning is better suited for `create_partitioned_table()` on tables with known, bounded value sets (e.g., region codes).

### Standalone Partitioned Table (Non-Vector)

For non-vector tables (e.g., for SQL agents), use `create_partitioned_table`:

```python
from llama_index.vector_stores.polardbx import create_partitioned_table

# HASH partitioning
create_partitioned_table(
    uri="mysql+pymysql://user:password@host:3306/database",
    table_name="orders",
    columns=[
        "id BIGINT NOT NULL AUTO_INCREMENT",
        "user_id BIGINT NOT NULL",
        "amount DECIMAL(10,2)",
        "created_at DATETIME",
        "PRIMARY KEY (id)",
    ],
    partition_by="HASH",
    partition_column="user_id",
    partitions=16,
)

# Broadcast table (dimension table, full copy on every DN)
create_partitioned_table(
    uri="mysql+pymysql://user:password@host:3306/database",
    table_name="dim_currency",
    columns=["code VARCHAR(10)", "name VARCHAR(100)", "PRIMARY KEY (code)"],
    broadcast=True,
)

# RANGE partitioning
create_partitioned_table(
    uri="mysql+pymysql://user:password@host:3306/database",
    table_name="logs",
    columns=["id BIGINT NOT NULL", "ts DATETIME", "PRIMARY KEY (id)"],
    partition_by="RANGE",
    partition_column="id",
    partition_defs=[
        {"name": "p0", "values_less_than": 1000000},
        {"name": "p1", "values_less_than": 2000000},
        {"name": "p2", "values_less_than": "MAXVALUE"},
    ],
)

# LIST partitioning
create_partitioned_table(
    uri="mysql+pymysql://user:password@host:3306/database",
    table_name="customers",
    columns=[
        "id BIGINT NOT NULL AUTO_INCREMENT",
        "region VARCHAR(20) NOT NULL",
        "name VARCHAR(255)",
        "PRIMARY KEY (id, region)",
    ],
    partition_by="LIST",
    partition_column="region",
    partition_defs=[
        {"name": "p_east", "values_in": ["east"]},
        {"name": "p_west", "values_in": ["west"]},
        {"name": "p_other", "values_in": ["north", "south"]},
    ],
)
```

Supported partition strategies:

| Strategy | Parameters | Description |
|----------|------------|-------------|
| `HASH` | `partition_column`, `partitions` | Hash partitioning by column value |
| `KEY` | `partition_column`, `partitions` | Key partitioning (single column) |
| `RANGE` | `partition_column`, `partition_defs` | Range partitioning with explicit boundaries |
| `LIST` | `partition_column`, `partition_defs` | List partitioning with explicit value lists |
| `BROADCAST` | (none) | Full table copy on every DN node |
| `LOCALITY` | `locality` | Pin table to a specific storage node |

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
| `default_m` | int | 6 | HNSW index M parameter (DB allows 3-200; client validates positive int only) |
| `perform_setup` | bool | True | Whether to auto-create table on init |
| `debug` | bool | False | Enable SQLAlchemy echo mode |
| `ef_construction` | int | None | HNSW build-time candidate list size (DB allows 5-1000, v3 only; client validates positive int only) |
| `ssl` | bool | False | Enable TLS/SSL encryption |
| `ssl_ca` | str | None | Path to CA certificate for SSL verification (only effective when `ssl=True`) |
| `vector_index_name` | str | None | Vector index name for FORCE INDEX hints (auto-detected if None) |
| `connection_retries` | int | 3 | Number of connection retry attempts during initialization |
| `retry_delay` | float | 1.0 | Delay between retry attempts in seconds |
| `partition_by` | str | None | Partition strategy: `"HASH"`, `"KEY"`, `"RANGE"`, or `"LIST"` |
| `partitions` | int | 0 | Number of partitions (required for HASH/KEY) |
| `partition_column` | str | None | Column to partition on (must match `id_column` value for vector tables; defaults to `id_column` value at runtime) |
| `broadcast` | bool | False | Create a broadcast table (full copy on every DN) |
| `locality` | str | None | Pin table to a specific DN node, e.g. `"dn=node-name"` |
| `partition_defs` | list | None | Partition definitions for RANGE/LIST (see examples above) |
| `id_column` | str | `"id"` | Custom name for the primary key column (UUID VARCHAR(36)) |
| `node_id_column` | str | `"node_id"` | Custom name for the LlamaIndex node_id column (VARCHAR(255)) |
| `text_column` | str | `"text"` | Custom name for the text content column (LONGTEXT) |
| `embedding_column` | str | `"embedding"` | Custom name for the vector embedding column (VECTOR(N)) |
| `metadata_json_column` | str | `"metadata"` | Custom name for the JSON metadata column. Set `None` to disable the JSON column (requires `metadata_columns` to be non-empty) |
| `metadata_columns` | list | None | List of `Column` objects or column name strings to promote metadata keys to dedicated typed columns (see [Custom Column Schema](#custom-column-schema)) |
| `**kwargs` | - | - | Additional pymysql connection arguments (e.g. `ssl_cert`, `ssl_key`, `ssl_verify_ca`, `ssl_verify_identity`, `ssl_disabled`, `connect_timeout`, `read_timeout`, `write_timeout`, `charset`, `collation`, `autocommit`, `unix_socket`) |

## PolarDB-X Vector Functions Used

This integration uses PolarDB-X's native vector functions:

- `VECTOR(N)` — Vector column data type with N dimensions
- `VEC_FROMTEXT('[1,2,3]')` — Convert JSON array string to vector
- `VEC_TOTEXT(vector)` — Convert vector to JSON array string
- `VEC_DISTANCE(v1, v2)` — Auto-inferred distance function (v3)
- `VEC_DISTANCE_COSINE(v1, v2)` — Cosine distance (old versions)
- `VEC_DISTANCE_EUCLIDEAN(v1, v2)` — Euclidean distance (old versions)
- `VEC_DISTANCE_INNER_PRODUCT(v1, v2)` — Inner product distance (used when v3 auto-inference unavailable; INNER_PRODUCT distance itself requires v3)
- `VECTOR_DIM(v)` — Get vector dimension (v3)
- `VECTOR INDEX (col) M=N DISTANCE=COSINE` — HNSW vector index DDL
- `EF_CONSTRUCTION=N` — HNSW build-time parameter in DDL (v3)
- `SET SESSION vidx_hnsw_ef_search = N` — Per-session search width tuning
- `SHOW GLOBAL STATUS LIKE 'Vidx%'` — Runtime index statistics
- `CALL dbms_vidx.preload(db, table, col)` — Preload index into cache (v3)
- `CALL dbms_vidx.preload_check(db, table, col)` — Check preload feasibility (v3)
- `information_schema.VECTOR_INDEXES` — Vector index metadata view (v3)

## Error Handling

When using features that require PolarDB-X v3, a `NotSupportedError` is raised on older versions. Features that require v3 include:

- `distance_method="INNER_PRODUCT"` (init-time validation)
- `preload_index()` / `apreload_index()`
- `preload_check()` / `apreload_check()`
- `explain_index_health()` / `aexplain_index_health()`
- Partitioned vector table creation (`partition_by`, `broadcast` on incompatible instances)
- Partitioned standalone table creation via `create_partitioned_table()` on incompatible instances

```python
from llama_index.vector_stores.polardbx import PolarDBXVectorStore, NotSupportedError

try:
    vector_store.preload_index()
except NotSupportedError as e:
    print(f"Feature not supported: {e}")
```

## Development

This package uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync --group dev

# Run unit tests
pytest tests/unit_tests/ -v

# Run integration tests (requires a running PolarDB-X instance)
pytest tests/integration_tests/ -v

# Lint
ruff check llama_index/
```

## License

MIT
