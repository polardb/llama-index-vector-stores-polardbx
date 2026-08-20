"""Integration tests for review bug fixes — real PolarDB-X end-to-end."""

import os
import sys
import uuid

import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import (  # noqa: E402
    EMB,
    METADATAS,
    TEXTS,
    make_nodes,
    make_store,
)


# ==================================================================
# Bug-1: delete/adelete in no-JSON-column mode
# ==================================================================


class TestBug1DeleteNoJsonIT:
    """Verify delete/adelete raise clear ValueError when metadata_json_column=None."""

    def test_delete_raises_in_no_json_mode(self):
        """delete(ref_doc_id) raises ValueError, not SQL crash."""
        vs = make_store(
            table_name=f"it_bug1_del_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        nodes = make_nodes(TEXTS[:3], METADATAS[:3])
        vs.add(nodes)
        assert vs.count() == 3

        with pytest.raises(ValueError, match="requires a JSON metadata column"):
            vs.delete(ref_doc_id="doc-1")

        # Table should still have all 3 nodes (delete didn't execute)
        assert vs.count() == 3
        vs.drop()
        vs.close()

    async def test_adelete_raises_in_no_json_mode(self):
        """adelete(ref_doc_id) raises ValueError, not SQL crash."""
        vs = make_store(
            table_name=f"it_bug1_adel_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        nodes = make_nodes(TEXTS[:3], METADATAS[:3])
        await vs.async_add(nodes)
        assert await vs.acount() == 3

        with pytest.raises(ValueError, match="requires a JSON metadata column"):
            await vs.adelete(ref_doc_id="doc-1")

        assert await vs.acount() == 3
        vs.drop()
        vs.close()

    def test_delete_works_in_json_mode(self):
        """delete(ref_doc_id) works when metadata_json_column is set."""
        vs = make_store(
            table_name=f"it_bug1_json_{uuid.uuid4().hex[:8]}",
            metadata_json_column="my_meta",
            metadata_columns=["category"],
        )
        ref_ids = ["rdoc-1", "rdoc-2", "rdoc-3"]
        nodes = make_nodes(TEXTS[:3], METADATAS[:3], ref_doc_ids=ref_ids)
        vs.add(nodes)
        assert vs.count() == 3

        vs.delete(ref_doc_id="rdoc-2")
        assert vs.count() == 2
        vs.drop()
        vs.close()

    def test_delete_nodes_by_filter_works_in_no_json_mode(self):
        """delete_nodes(filters=...) works in no-JSON mode as alternative."""
        vs = make_store(
            table_name=f"it_bug1_dnf_{uuid.uuid4().hex[:8]}",
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
# Bug-2: partition_column + custom id_column
# ==================================================================


class TestBug2PartitionIdColumnIT:
    """Verify custom id_column + partition_by works end-to-end."""

    def test_custom_id_column_with_hash_partition(self):
        """id_column="my_id" + HASH partitioning creates table and works."""
        vs = make_store(
            table_name=f"it_bug2_hash_{uuid.uuid4().hex[:8]}",
            id_column="my_id",
            node_id_column="my_node_id",
            text_column="my_text",
            embedding_column="my_emb",
            metadata_json_column="my_meta",
            partition_by="HASH",
            partitions=4,
        )
        assert vs._id_column == "my_id"
        assert vs._partition_column == "my_id"

        nodes = make_nodes(TEXTS[:3], METADATAS[:3])
        vs.add(nodes)
        assert vs.count() == 3

        result = vs.query(
            __import__(
                "llama_index.core.vector_stores.types",
                fromlist=["VectorStoreQuery", "VectorStoreQueryMode"],
            ).VectorStoreQuery(
                query_embedding=EMB.embed_query("database"),
                similarity_top_k=2,
            )
        )
        assert len(result.nodes) == 2
        vs.drop()
        vs.close()

    def test_custom_id_column_partition_column_match(self):
        """Explicit partition_column matching id_column passes validation."""
        vs = make_store(
            table_name=f"it_bug2_match_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            partition_by="HASH",
            partitions=4,
            partition_column="cid",
        )
        assert vs._id_column == "cid"
        assert vs._partition_column == "cid"
        assert vs.count() == 0
        vs.drop()
        vs.close()

    def test_default_id_with_partition_still_works(self):
        """Default id_column="id" with partition_by still works (regression)."""
        vs = make_store(
            table_name=f"it_bug2_def_{uuid.uuid4().hex[:8]}",
            partition_by="HASH",
            partitions=4,
        )
        assert vs._id_column == "id"
        assert vs._partition_column == "id"

        nodes = make_nodes(TEXTS[:2], METADATAS[:2])
        vs.add(nodes)
        assert vs.count() == 2
        vs.drop()
        vs.close()

    def test_custom_id_column_with_custom_columns_and_partition(self):
        """Full custom schema: id_column + metadata_columns + partition."""
        from llama_index.vector_stores.polardbx import Column

        vs = make_store(
            table_name=f"it_bug2_full_{uuid.uuid4().hex[:8]}",
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column="cmeta",
            metadata_columns=[
                Column("category", "VARCHAR(64)", nullable=False),
                Column("lang", "VARCHAR(8)"),
            ],
            partition_by="HASH",
            partitions=4,
        )
        assert vs._has_custom_columns
        assert vs._id_column == "cid"
        assert vs._partition_column == "cid"

        nodes = make_nodes(TEXTS[:3], METADATAS[:3])
        vs.add(nodes)
        assert vs.count() == 3

        # Verify query works
        from llama_index.core.vector_stores.types import (
            VectorStoreQuery,
            VectorStoreQueryMode,
        )

        result = vs.query(
            VectorStoreQuery(
                query_embedding=EMB.embed_query("database"),
                similarity_top_k=2,
                mode=VectorStoreQueryMode.DEFAULT,
            )
        )
        assert len(result.nodes) == 2
        for node in result.nodes:
            assert node.metadata.get("category") is not None

        # Verify filter on mapped column works
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value="database",
                    operator=FilterOperator.EQ,
                ),
            ],
        )
        result = vs.query(
            VectorStoreQuery(
                query_embedding=EMB.embed_query("database"),
                similarity_top_k=10,
                filters=filters,
                mode=VectorStoreQueryMode.DEFAULT,
            )
        )
        for node in result.nodes:
            assert node.metadata.get("category") == "database"

        vs.drop()
        vs.close()
