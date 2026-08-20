"""Integration tests for custom column support — real PolarDB-X end-to-end.

These tests connect to a real PolarDB-X instance and exercise the full
pipeline: DDL generation → INSERT → SELECT (similarity search) → filter
→ get_nodes → delete → search_by_metadata, all with custom column names
and mapped metadata columns.

Requires a configured .env with POLARDBX_URI.
"""

import os
import sys
import uuid

import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import (  # noqa: E402
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    EMB,
    EMBED_DIM,
    METADATAS,
    TEXTS,
    make_nodes,
    make_store,
)
from llama_index.vector_stores.polardbx import Column  # noqa: E402


def _build_query(text="database", k=3, filters=None):
    return VectorStoreQuery(
        query_embedding=EMB.embed_query(text),
        similarity_top_k=k,
        filters=filters,
        mode=VectorStoreQueryMode.DEFAULT,
    )


def _custom_store(table_suffix: str = "", **kwargs):
    """Create a store with custom columns for IT testing."""
    table_name = f"it_custom_{uuid.uuid4().hex[:8]}{table_suffix}"
    return make_store(
        table_name=table_name,
        id_column="my_id",
        node_id_column="my_node_id",
        text_column="my_text",
        embedding_column="my_embedding",
        metadata_json_column="my_meta",
        metadata_columns=[
            Column(name="category", data_type="VARCHAR(64)", nullable=False),
            Column(name="lang", data_type="VARCHAR(8)"),
            "score",
        ],
        **kwargs,
    )


# ==================================================================
# 1. DDL: custom table creation
# ==================================================================


class TestCustomDDL:
    """Verify DDL executes successfully on a real instance."""

    def test_custom_columns_table_creation(self):
        """Table with custom column names + metadata columns is created."""
        vs = _custom_store()
        assert vs._has_custom_columns
        # Table should exist and be empty
        assert vs.count() == 0
        vs.drop()
        vs.close()

    def test_custom_columns_no_json_table_creation(self):
        """Table without a JSON metadata column is created."""
        vs = make_store(
            table_name=f"it_nojson_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        assert vs._has_custom_columns
        assert vs.count() == 0
        vs.drop()
        vs.close()


# ==================================================================
# 2. INSERT: add nodes with custom columns
# ==================================================================


class TestCustomAdd:
    """Verify INSERT works with custom column layout."""

    def test_add_writes_mapped_columns(self):
        """Metadata values are written to mapped columns, rest to JSON."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        # Add score to metadata for the "score" mapped column
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10

        ids = vs.add(nodes)
        assert len(ids) == 5
        assert vs.count() == 5
        vs.drop()
        vs.close()

    def test_add_upsert_on_duplicate(self):
        """Re-adding same node_id upserts, not duplicates."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS[:3], METADATAS[:3])
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)
        assert vs.count() == 3

        # Re-add with updated text
        updated = make_nodes(["UPD " + t for t in TEXTS[:3]], METADATAS[:3])
        for i, n in enumerate(updated):
            n.id_ = nodes[i].node_id
            n.metadata["score"] = 99
        vs.add(updated)
        assert vs.count() == 3  # still 3, not 6

        fetched = vs.get_nodes(node_ids=[nodes[0].node_id])
        assert "UPD" in fetched[0].get_content()
        vs.drop()
        vs.close()

    def test_not_null_validation(self):
        """NOT NULL column without a value raises ValueError."""
        vs = _custom_store()
        # category is NOT NULL but node metadata omits it
        nodes = make_nodes(["test text"], [{"lang": "en"}])
        with pytest.raises(ValueError, match="NOT NULL"):
            vs.add(nodes)
        vs.drop()
        vs.close()

    async def test_async_add_writes_mapped_columns(self):
        """async_add writes mapped columns correctly."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10

        ids = await vs.async_add(nodes)
        assert len(ids) == 5
        assert await vs.acount() == 5
        vs.drop()
        vs.close()


# ==================================================================
# 3. SELECT: similarity search with custom columns
# ==================================================================


class TestCustomQuery:
    """Verify query() returns correct results with merged metadata."""

    def test_query_returns_correct_results(self):
        """query() returns nodes with correct content and similarity."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)

        result = vs.query(_build_query("database", k=3))
        assert len(result.nodes) == 3
        assert len(result.similarities) == 3
        assert len(result.ids) == 3
        for node in result.nodes:
            assert node.get_content() != ""
        vs.drop()
        vs.close()

    def test_query_metadata_merged_correctly(self):
        """Mapped column values and JSON values are merged in metadata."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
            node.metadata["extra_field"] = f"extra_{i}"
        vs.add(nodes)

        result = vs.query(_build_query("database", k=10))
        assert len(result.nodes) >= 1

        # Mapped columns should be in metadata
        for node in result.nodes:
            assert "category" in node.metadata
            assert "lang" in node.metadata
            assert "score" in node.metadata
            assert "extra_field" in node.metadata  # from JSON
        vs.drop()
        vs.close()

    def test_query_no_json_column(self):
        """Query works when there is no JSON metadata column."""
        vs = make_store(
            table_name=f"it_qnojson_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        result = vs.query(_build_query("database", k=3))
        assert len(result.nodes) == 3
        for node in result.nodes:
            assert node.metadata.get("category") is not None
            assert node.metadata.get("lang") is not None
        vs.drop()
        vs.close()

    async def test_async_query_returns_results(self):
        """async query works with custom columns."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        await vs.async_add(nodes)

        result = await vs.aquery(_build_query("database", k=3))
        assert len(result.nodes) == 3
        vs.drop()
        vs.close()


# ==================================================================
# 4. FILTER: mapped column vs JSON_EXTRACT branching
# ==================================================================


class TestCustomFilter:
    """Verify filter clause branching: mapped column vs JSON."""

    def test_filter_on_mapped_column(self):
        """Filter on a mapped column uses direct column reference."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)

        # category is a mapped column → direct column reference
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value="language",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        result = vs.query(_build_query("language", k=10, filters=filters))
        for node in result.nodes:
            assert node.metadata.get("category") == "language"
        vs.drop()
        vs.close()

    def test_filter_on_json_key(self):
        """Filter on a non-mapped key uses JSON_EXTRACT."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
            node.metadata["source"] = f"src_{i}"
        vs.add(nodes)

        # 'source' is not a mapped column → JSON_EXTRACT on custom JSON col
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="source",
                    value="src_0",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        result = vs.query(_build_query("database", k=10, filters=filters))
        for node in result.nodes:
            assert node.metadata.get("source") == "src_0"
        vs.drop()
        vs.close()

    def test_filter_on_mapped_column_with_gt(self):
        """Numeric comparison on a mapped column works."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="score",
                    value=30,
                    operator=FilterOperator.GTE,
                ),
            ],
        )
        result = vs.query(_build_query("database", k=10, filters=filters))
        for node in result.nodes:
            assert node.metadata.get("score", 0) >= 30
        vs.drop()
        vs.close()

    def test_filter_no_json_raises_on_unmapped_key(self):
        """Without JSON column, filtering on unmapped key raises."""
        vs = make_store(
            table_name=f"it_fnomap_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="nonexistent_key",
                    value="x",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        with pytest.raises(ValueError, match="Cannot filter"):
            vs.query(_build_query("database", k=3, filters=filters))
        vs.drop()
        vs.close()


# ==================================================================
# 5. get_nodes / delete / search_by_metadata with custom columns
# ==================================================================


class TestCustomCRUD:
    """Verify get_nodes, delete, search_by_metadata work."""

    def test_get_nodes_by_ids(self):
        """get_nodes returns nodes with correct metadata."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)

        fetched = vs.get_nodes(
            node_ids=[nodes[0].node_id, nodes[2].node_id]
        )
        assert len(fetched) == 2
        for n in fetched:
            assert n.metadata.get("category") is not None
            assert n.metadata.get("score") is not None
        vs.drop()
        vs.close()

    def test_get_nodes_all(self):
        """get_nodes without args returns all nodes."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)

        fetched = vs.get_nodes()
        assert len(fetched) == 5
        vs.drop()
        vs.close()

    def test_delete_by_ref_doc_id(self):
        """delete removes nodes by ref_doc_id."""
        vs = _custom_store()
        ref_ids = ["cdoc-1", "cdoc-2", "cdoc-3"]
        nodes = make_nodes(TEXTS[:3], METADATAS[:3], ref_doc_ids=ref_ids)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)
        assert vs.count() == 3

        vs.delete(ref_doc_id="cdoc-2")
        assert vs.count() == 2
        vs.drop()
        vs.close()

    def test_delete_nodes_by_ids(self):
        """delete_nodes removes nodes by node_ids."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS[:3], METADATAS[:3])
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)
        assert vs.count() == 3

        vs.delete_nodes(node_ids=[nodes[0].node_id])
        assert vs.count() == 2
        vs.drop()
        vs.close()

    def test_delete_nodes_by_filter(self):
        """delete_nodes with filters removes matching nodes."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)
        assert vs.count() == 5

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value="language",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        vs.delete_nodes(filters=filters)
        assert vs.count() == 4
        vs.drop()
        vs.close()

    def test_search_by_metadata(self):
        """search_by_metadata returns matching nodes."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value="database",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        results = vs.search_by_metadata(filters, limit=10)
        assert len(results) == 1
        assert "PolarDB-X" in results[0].get_content()
        vs.drop()
        vs.close()

    def test_search_by_metadata_with_score_filter(self):
        """search_by_metadata with numeric filter on mapped column."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="score",
                    value=30,
                    operator=FilterOperator.GTE,
                ),
            ],
        )
        results = vs.search_by_metadata(filters, limit=10)
        assert len(results) == 3  # scores 30, 40, 50
        vs.drop()
        vs.close()

    async def test_async_get_nodes_by_ids(self):
        """aget_nodes returns nodes with custom columns."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        await vs.async_add(nodes)

        fetched = await vs.aget_nodes(
            node_ids=[nodes[1].node_id, nodes[3].node_id]
        )
        assert len(fetched) == 2
        vs.drop()
        vs.close()

    async def test_async_delete_nodes_by_ids(self):
        """adelete_nodes removes nodes by node_ids."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS[:3], METADATAS[:3])
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        await vs.async_add(nodes)
        assert await vs.acount() == 3

        await vs.adelete_nodes(node_ids=[nodes[0].node_id, nodes[1].node_id])
        assert await vs.acount() == 1
        vs.drop()
        vs.close()

    async def test_async_search_by_metadata(self):
        """asearch_by_metadata returns matching nodes."""
        vs = _custom_store()
        nodes = make_nodes(TEXTS, METADATAS)
        for i, node in enumerate(nodes):
            node.metadata["score"] = (i + 1) * 10
        await vs.async_add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value="search",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        results = await vs.asearch_by_metadata(filters, limit=10)
        assert len(results) == 1
        vs.drop()
        vs.close()


# ==================================================================
# 6. No-JSON-column mode: end-to-end
# ==================================================================


class TestNoJsonMode:
    """Tests for metadata_json_column=None with only mapped columns."""

    def test_no_json_add_and_query(self):
        """Add and query with no JSON column."""
        vs = make_store(
            table_name=f"it_nj_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)
        assert vs.count() == 5

        result = vs.query(_build_query("database", k=3))
        assert len(result.nodes) == 3
        for node in result.nodes:
            assert node.metadata.get("category") is not None
            assert node.metadata.get("lang") is not None
        vs.drop()
        vs.close()

    def test_no_json_get_nodes(self):
        """get_nodes returns correct metadata without JSON column."""
        vs = make_store(
            table_name=f"it_njgn_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        nodes = make_nodes(TEXTS[:3], METADATAS[:3])
        vs.add(nodes)

        fetched = vs.get_nodes(node_ids=[nodes[0].node_id])
        assert len(fetched) == 1
        assert fetched[0].metadata.get("category") == "database"
        assert fetched[0].metadata.get("lang") == "en"
        vs.drop()
        vs.close()

    def test_no_json_search_by_metadata(self):
        """search_by_metadata works without JSON column."""
        vs = make_store(
            table_name=f"it_njsm_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value="framework",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        results = vs.search_by_metadata(filters, limit=10)
        assert len(results) == 1
        assert "LlamaIndex" in results[0].get_content()
        vs.drop()
        vs.close()

    def test_no_json_delete_by_filter(self):
        """delete_nodes with filter works without JSON column."""
        vs = make_store(
            table_name=f"it_njdf_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)
        assert vs.count() == 5

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value="database",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        vs.delete_nodes(filters=filters)
        assert vs.count() == 4
        vs.drop()
        vs.close()


# ==================================================================
# 7. Regression: default schema still works
# ==================================================================


class TestDefaultSchemaRegression:
    """Ensure the default five-column schema still works after all changes."""

    def test_default_add_and_query(self):
        """Default schema add + query still works."""
        vs = make_store()
        assert not vs._has_custom_columns
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)
        assert vs.count() == 5

        result = vs.query(_build_query("database", k=3))
        assert len(result.nodes) == 3
        for node in result.nodes:
            assert node.metadata.get("category") is not None
        vs.drop()
        vs.close()

    def test_default_filter_on_metadata(self):
        """Default schema filter on metadata JSON still works."""
        vs = make_store()
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value="database",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        result = vs.query(_build_query("database", k=10, filters=filters))
        for node in result.nodes:
            assert node.metadata.get("category") == "database"
        vs.drop()
        vs.close()

    def test_default_get_nodes_and_delete(self):
        """Default schema get_nodes + delete still work."""
        vs = make_store()
        ref_ids = ["rg-1", "rg-2", "rg-3"]
        nodes = make_nodes(TEXTS[:3], METADATAS[:3], ref_doc_ids=ref_ids)
        vs.add(nodes)
        assert vs.count() == 3

        fetched = vs.get_nodes(node_ids=[nodes[0].node_id])
        assert len(fetched) == 1

        vs.delete(ref_doc_id="rg-1")
        assert vs.count() == 2

        results = vs.search_by_metadata(
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category",
                        value="framework",
                        operator=FilterOperator.EQ,
                    ),
                ],
            ),
            limit=10,
        )
        assert len(results) == 1
        vs.drop()
        vs.close()
