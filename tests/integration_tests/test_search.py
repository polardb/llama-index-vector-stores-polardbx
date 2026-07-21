"""Tests for LlamaIndex PolarDBXVectorStore — search & filtering (sync + async)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, METADATAS, TEXTS, make_nodes, make_store
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


def _build_query(
    text: str = "database",
    k: int = 3,
    filters: MetadataFilters | None = None,
) -> VectorStoreQuery:
    """Build a VectorStoreQuery for testing."""
    return VectorStoreQuery(
        query_embedding=EMB.embed_query(text),
        similarity_top_k=k,
        filters=filters,
        mode=VectorStoreQueryMode.DEFAULT,
    )


def _setup(vs):
    """Insert standard test data."""
    nodes = make_nodes(TEXTS, METADATAS)
    vs.add(nodes)
    return nodes


# ==================== SYNC: basic query ====================


def test_sync_basic_query():
    """query() returns correct number of results."""
    vs = make_store()
    _setup(vs)

    result = vs.query(_build_query("database", k=3))
    assert len(result.nodes) == 3
    assert len(result.similarities) == 3
    assert len(result.ids) == 3
    vs.drop()
    vs.close()


def test_sync_query_returns_nodes_with_content():
    """query() result nodes have text content."""
    vs = make_store()
    _setup(vs)

    result = vs.query(_build_query("database", k=2))
    for node in result.nodes:
        assert node.get_content() != ""
    vs.drop()
    vs.close()


def test_sync_query_similarities_are_floats():
    """query() result similarities are float values."""
    vs = make_store()
    _setup(vs)

    result = vs.query(_build_query("database", k=3))
    for sim in result.similarities:
        assert isinstance(sim, (int, float))
    vs.drop()
    vs.close()


def test_sync_query_k_exceeds_data():
    """query() with k > data count returns all available."""
    vs = make_store()
    _setup(vs)  # 5 nodes

    result = vs.query(_build_query("database", k=100))
    assert len(result.nodes) == 5  # only 5 available
    vs.drop()
    vs.close()


def test_sync_query_empty_table():
    """query() on empty table returns empty result, not error."""
    vs = make_store()
    # Don't insert anything

    result = vs.query(_build_query("anything", k=3))
    assert len(result.nodes) == 0
    vs.drop()
    vs.close()


# ==================== SYNC: query with filters ====================


def test_sync_query_with_eq_filter():
    """query() with EQ filter returns only matching nodes."""
    vs = make_store()
    _setup(vs)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="language", operator=FilterOperator.EQ
            ),
        ],
    )
    result = vs.query(_build_query("language", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") == "language"
    vs.drop()
    vs.close()


def test_sync_query_with_ne_filter():
    """query() with NE filter excludes matching nodes."""
    vs = make_store()
    _setup(vs)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="database", operator=FilterOperator.NE
            ),
        ],
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") != "database"
    vs.drop()
    vs.close()


def test_sync_query_with_in_filter():
    """query() with IN filter returns nodes matching any value."""
    vs = make_store()
    _setup(vs)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value=["database", "search"],
                operator=FilterOperator.IN,
            ),
        ],
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") in ["database", "search"]
    vs.drop()
    vs.close()


def test_sync_query_with_nin_filter():
    """query() with NIN filter excludes matching values."""
    vs = make_store()
    _setup(vs)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value=["database", "search"],
                operator=FilterOperator.NIN,
            ),
        ],
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") not in ["database", "search"]
    vs.drop()
    vs.close()


def test_sync_query_with_gt_filter():
    """query() with GT filter returns nodes where value > threshold."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    # Add score metadata
    for i, node in enumerate(nodes):
        node.metadata["score"] = (i + 1) * 10
    vs.add(nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="score", value=30, operator=FilterOperator.GT
            ),
        ],
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("score", 0) > 30
    vs.drop()
    vs.close()


def test_sync_query_with_gte_filter():
    """query() with GTE filter returns nodes where value >= threshold."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    for i, node in enumerate(nodes):
        node.metadata["score"] = (i + 1) * 10
    vs.add(nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="score", value=30, operator=FilterOperator.GTE
            ),
        ],
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("score", 0) >= 30
    vs.drop()
    vs.close()


def test_sync_query_with_lt_filter():
    """query() with LT filter returns nodes where value < threshold."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    for i, node in enumerate(nodes):
        node.metadata["score"] = (i + 1) * 10
    vs.add(nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="score", value=30, operator=FilterOperator.LT
            ),
        ],
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("score", 999) < 30
    vs.drop()
    vs.close()


def test_sync_query_with_lte_filter():
    """query() with LTE filter returns nodes where value <= threshold."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    for i, node in enumerate(nodes):
        node.metadata["score"] = (i + 1) * 10
    vs.add(nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="score", value=30, operator=FilterOperator.LTE
            ),
        ],
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("score", 999) <= 30
    vs.drop()
    vs.close()


def test_sync_query_with_multi_and_filter():
    """query() with multiple AND conditions."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    for i, node in enumerate(nodes):
        node.metadata["score"] = (i + 1) * 10
    vs.add(nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="language", operator=FilterOperator.EQ
            ),
            MetadataFilter(
                key="score", value=30, operator=FilterOperator.GT
            ),
        ],
        condition=FilterCondition.AND,
    )
    result = vs.query(_build_query("language", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") == "language"
        assert node.metadata.get("score", 0) > 30
    vs.drop()
    vs.close()


def test_sync_query_with_nonexistent_key():
    """query() with filter on nonexistent metadata key returns empty."""
    vs = make_store()
    _setup(vs)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="nonexistent_key", value="value", operator=FilterOperator.EQ
            ),
        ],
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    assert len(result.nodes) == 0
    vs.drop()
    vs.close()


# ==================== SYNC: search_type & ef_search ====================


def test_sync_query_search_type_auto():
    """query() with search_type='auto' returns results."""
    vs = make_store()
    _setup(vs)

    result = vs.query(_build_query("database", k=3), search_type="auto")
    assert len(result.nodes) <= 3
    assert len(result.nodes) > 0
    vs.drop()
    vs.close()


def test_sync_query_search_type_knn():
    """query() with search_type='knn' forces full table scan."""
    vs = make_store()
    _setup(vs)

    result = vs.query(_build_query("database", k=3), search_type="knn")
    assert len(result.nodes) <= 3
    assert len(result.nodes) > 0
    vs.drop()
    vs.close()


def test_sync_query_search_type_ann():
    """query() with search_type='ann' forces vector index usage."""
    vs = make_store()
    _setup(vs)

    result = vs.query(_build_query("database", k=3), search_type="ann")
    assert len(result.nodes) <= 3
    assert len(result.nodes) > 0
    vs.drop()
    vs.close()


def test_sync_query_with_ef_search():
    """query() with ef_search parameter sets session variable."""
    vs = make_store()
    _setup(vs)

    result = vs.query(_build_query("database", k=3), ef_search=50)
    assert len(result.nodes) <= 3
    vs.drop()
    vs.close()


def test_sync_query_ef_search_boundary():
    """query() with ef_search at extreme values (1 and 10000)."""
    vs = make_store()
    _setup(vs)

    # ef_search=1 — minimum
    result = vs.query(_build_query("database", k=3), ef_search=1)
    assert len(result.nodes) <= 3

    # ef_search=10000 — very large
    result = vs.query(_build_query("database", k=3), ef_search=10000)
    assert len(result.nodes) <= 3
    vs.drop()
    vs.close()


# ==================== SYNC: build_index_hint ====================


def test_sync_build_index_hint_auto():
    """_build_index_hint('auto') returns empty string."""
    vs = make_store()
    assert vs._build_index_hint("auto") == ""
    vs.drop()
    vs.close()


def test_sync_build_index_hint_none():
    """_build_index_hint(None) returns empty string."""
    vs = make_store()
    assert vs._build_index_hint(None) == ""
    vs.drop()
    vs.close()


def test_sync_build_index_hint_knn():
    """_build_index_hint('knn') returns FORCE INDEX(PRIMARY)."""
    vs = make_store()
    hint = vs._build_index_hint("knn")
    assert "FORCE INDEX(PRIMARY)" in hint
    vs.drop()
    vs.close()


def test_sync_build_index_hint_ann():
    """_build_index_hint('ann') returns FORCE INDEX with index name."""
    vs = make_store()
    hint = vs._build_index_hint("ann")
    assert "FORCE INDEX" in hint
    vs.drop()
    vs.close()


# ==================== ASYNC ====================


async def test_async_basic_query():
    """aquery() returns correct number of results."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    await vs.async_add(nodes)

    result = await vs.aquery(_build_query("database", k=3))
    assert len(result.nodes) == 3
    assert len(result.similarities) == 3
    assert len(result.ids) == 3
    vs.drop()
    vs.close()


async def test_async_query_k_exceeds_data():
    """aquery() with k > data count returns all available."""
    vs = make_store()
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    await vs.async_add(nodes)

    result = await vs.aquery(_build_query("database", k=100))
    assert len(result.nodes) == 3
    vs.drop()
    vs.close()


async def test_async_query_empty_table():
    """aquery() on empty table returns empty result."""
    vs = make_store()

    result = await vs.aquery(_build_query("anything", k=3))
    assert len(result.nodes) == 0
    vs.drop()
    vs.close()


async def test_async_query_with_eq_filter():
    """aquery() with EQ filter returns only matching nodes."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    await vs.async_add(nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="database", operator=FilterOperator.EQ
            ),
        ],
    )
    result = await vs.aquery(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") == "database"
    vs.drop()
    vs.close()


async def test_async_query_with_in_filter():
    """aquery() with IN filter returns nodes matching any value."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    await vs.async_add(nodes)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value=["database", "language"],
                operator=FilterOperator.IN,
            ),
        ],
    )
    result = await vs.aquery(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") in ["database", "language"]
    vs.drop()
    vs.close()


async def test_async_query_search_type_knn():
    """aquery() with search_type='knn' forces full table scan."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    await vs.async_add(nodes)

    result = await vs.aquery(
        _build_query("database", k=3), search_type="knn"
    )
    assert len(result.nodes) <= 3
    assert len(result.nodes) > 0
    vs.drop()
    vs.close()


async def test_async_query_search_type_ann():
    """aquery() with search_type='ann' forces vector index usage."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    await vs.async_add(nodes)

    result = await vs.aquery(
        _build_query("database", k=3), search_type="ann"
    )
    assert len(result.nodes) <= 3
    assert len(result.nodes) > 0
    vs.drop()
    vs.close()


async def test_async_query_with_ef_search():
    """aquery() with ef_search parameter."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    await vs.async_add(nodes)

    result = await vs.aquery(
        _build_query("database", k=3), ef_search=80
    )
    assert len(result.nodes) <= 3
    vs.drop()
    vs.close()


# ==================== SYNC: get_nodes with filters ====================


def test_sync_get_nodes_with_eq_filter():
    """get_nodes() with EQ filter returns only matching nodes."""
    vs = make_store()
    _setup(vs)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="language", operator=FilterOperator.EQ
            ),
        ],
    )
    fetched = vs.get_nodes(filters=filters)
    assert len(fetched) >= 1
    for node in fetched:
        assert node.metadata.get("category") == "language"
    vs.drop()
    vs.close()


def test_sync_get_nodes_with_in_filter():
    """get_nodes() with IN filter returns nodes matching any value."""
    vs = make_store()
    _setup(vs)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value=["database", "search"],
                operator=FilterOperator.IN,
            ),
        ],
    )
    fetched = vs.get_nodes(filters=filters)
    for node in fetched:
        assert node.metadata.get("category") in ["database", "search"]
    vs.drop()
    vs.close()


# ==================== SYNC: OR condition & nested filters ====================


def test_sync_query_with_or_condition():
    """query() with OR condition returns nodes matching either filter."""
    vs = make_store()
    _setup(vs)

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="database", operator=FilterOperator.EQ
            ),
            MetadataFilter(
                key="category", value="language", operator=FilterOperator.EQ
            ),
        ],
        condition=FilterCondition.OR,
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") in ["database", "language"]
    vs.drop()
    vs.close()


def test_sync_query_with_nested_filters():
    """query() with nested MetadataFilters (AND of OR groups)."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    for i, node in enumerate(nodes):
        node.metadata["score"] = (i + 1) * 10
    vs.add(nodes)

    # (category=database OR category=language) AND (score >= 40)
    filters = MetadataFilters(
        filters=[
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", value="database", operator=FilterOperator.EQ
                    ),
                    MetadataFilter(
                        key="category", value="language", operator=FilterOperator.EQ
                    ),
                ],
                condition=FilterCondition.OR,
            ),
            MetadataFilter(
                key="score", value=40, operator=FilterOperator.GTE
            ),
        ],
        condition=FilterCondition.AND,
    )
    result = vs.query(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        cat = node.metadata.get("category")
        score = node.metadata.get("score", 0)
        assert cat in ["database", "language"]
        assert score >= 40
    vs.drop()
    vs.close()


# ==================== SYNC: delete_nodes with filters ====================


def test_sync_delete_nodes_by_filter():
    """delete_nodes() with filters removes matching nodes."""
    vs = make_store(table_name="test_li_delnodefilt")
    _setup(vs)
    assert vs.count() == 5

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="language", operator=FilterOperator.EQ
            ),
        ],
    )
    vs.delete_nodes(filters=filters)
    assert vs.count() == 4  # only 1 "language" node deleted

    # Verify no remaining node has category=language
    remaining = vs.get_nodes()
    for node in remaining:
        assert node.metadata.get("category") != "language"
    vs.drop()
    vs.close()


# ==================== ASYNC: get_nodes with filters ====================


async def test_async_get_nodes_with_eq_filter():
    """aget_nodes() with EQ filter returns only matching nodes."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS, METADATAS))

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="database", operator=FilterOperator.EQ
            ),
        ],
    )
    fetched = await vs.aget_nodes(filters=filters)
    assert len(fetched) >= 1
    for node in fetched:
        assert node.metadata.get("category") == "database"
    vs.drop()
    vs.close()


async def test_async_get_nodes_with_in_filter():
    """aget_nodes() with IN filter returns nodes matching any value."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS, METADATAS))

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category",
                value=["framework", "index"],
                operator=FilterOperator.IN,
            ),
        ],
    )
    fetched = await vs.aget_nodes(filters=filters)
    for node in fetched:
        assert node.metadata.get("category") in ["framework", "index"]
    vs.drop()
    vs.close()


# ==================== ASYNC: OR condition ====================


async def test_async_query_with_or_condition():
    """aquery() with OR condition returns nodes matching either filter."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS, METADATAS))

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="database", operator=FilterOperator.EQ
            ),
            MetadataFilter(
                key="category", value="search", operator=FilterOperator.EQ
            ),
        ],
        condition=FilterCondition.OR,
    )
    result = await vs.aquery(_build_query("database", k=10, filters=filters))
    for node in result.nodes:
        assert node.metadata.get("category") in ["database", "search"]
    vs.drop()
    vs.close()


# ==================== ASYNC: delete_nodes with filters ====================


async def test_async_delete_nodes_by_filter():
    """adelete_nodes() with filters removes matching nodes."""
    vs = make_store(table_name="test_li_adelnodefilt")
    await vs.async_add(make_nodes(TEXTS, METADATAS))
    assert await vs.acount() == 5

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="index", operator=FilterOperator.EQ
            ),
        ],
    )
    await vs.adelete_nodes(filters=filters)
    assert await vs.acount() == 4

    # Verify no remaining node has category=index
    remaining = await vs.aget_nodes()
    for node in remaining:
        assert node.metadata.get("category") != "index"
    vs.drop()
    vs.close()
