"""PolarDB-X vector store integration for LlamaIndex.

This module will be implemented with:
- BasePydanticVectorStore base class
- PolarDB-X native VECTOR(N) type + HNSW index
- Dual-version compatibility (old + v3) via _detect_capabilities()
- V3 features: INNER_PRODUCT, EF_CONSTRUCTION, preload, VECTOR_INDEXES view
- Full async support (SQLAlchemy async engine + aiomysql)
"""

# Implementation will be added here


class NotSupportedError(Exception):
    """Raised when a feature is not supported on the current PolarDB-X version."""

    pass
