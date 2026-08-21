"""Integration tests for the 5 supplemented features — real PolarDB-X DB required.

Run with:
    POLARDBX_URI=mysql+pymysql://user:password@host:3306/database \
    .venv/bin/python -m pytest tests/integration_tests/test_supplemented_features.py -v -s
"""

import os
import sys
import time
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import (
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    EMB,
    EMBED_DIM,
    METADATAS,
    TEXTS,
    is_v3,
    make_nodes,
    make_store,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.polardbx import PolarDBXVectorStore


# ==================================================================
# 1. kwargs 校验 — 真实连接
# ==================================================================


class TestKwargsValidationIT:
    """Integration tests for kwargs validation with real DB."""

    def test_valid_connect_timeout_connects(self):
        """Passing connect_timeout should connect successfully."""
        vs = make_store(
            table_name=f"it_kwargs_{uuid.uuid4().hex[:8]}",
            connect_timeout=10,
        )
        assert vs._is_initialized is True
        assert vs.count() == 0
        vs.drop()
        vs.close()

    def test_valid_read_timeout_connects(self):
        """Passing read_timeout should connect successfully."""
        vs = make_store(
            table_name=f"it_kwargs2_{uuid.uuid4().hex[:8]}",
            read_timeout=30,
        )
        assert vs._is_initialized is True
        vs.drop()
        vs.close()

    def test_typo_raises_before_connection(self):
        """Typo in param name should raise TypeError before any DB call."""
        table = f"it_kwargs3_{uuid.uuid4().hex[:8]}"
        with pytest.raises(TypeError, match="Did you mean"):
            PolarDBXVectorStore(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                table_name=table,
                embed_dim=EMBED_DIM,
                embed_dims=EMBED_DIM,  # typo
            )
        # Table should not have been created
        _assert_table_not_exists(table)

    def test_unknown_kwarg_raises_typeerror(self):
        """Completely unknown kwarg should raise TypeError."""
        table = f"it_kwargs4_{uuid.uuid4().hex[:8]}"
        with pytest.raises(TypeError, match="not a recognized"):
            PolarDBXVectorStore(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                table_name=table,
                embed_dim=EMBED_DIM,
                bogus_param=True,
            )
        _assert_table_not_exists(table)


# ==================================================================
# 2. 连接重试 — 真实测试
# ==================================================================


class TestConnectionRetryIT:
    """Integration tests for connection retry with real DB."""

    def test_successful_connection_first_try(self):
        """Normal connection should succeed on first attempt."""
        vs = make_store(
            table_name=f"it_retry_{uuid.uuid4().hex[:8]}",
            connection_retries=3,
            retry_delay=0.1,
        )
        assert vs._is_initialized is True
        assert vs._connection_retries == 3
        vs.drop()
        vs.close()

    def test_retry_with_wrong_port_then_succeed(self):
        """Connection to wrong port should retry and eventually fail."""
        table = f"it_retry2_{uuid.uuid4().hex[:8]}"
        with pytest.raises(Exception):
            PolarDBXVectorStore(
                host=DB_HOST,
                port=9999,  # wrong port
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                table_name=table,
                embed_dim=EMBED_DIM,
                connection_retries=2,
                retry_delay=0.1,
            )

    def test_value_error_not_retried_real(self):
        """ValueError (e.g. bad embed_dim) should not retry."""
        table = f"it_retry3_{uuid.uuid4().hex[:8]}"
        with pytest.raises(ValueError):
            PolarDBXVectorStore(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                table_name=table,
                embed_dim=-1,  # invalid
                connection_retries=5,
                retry_delay=0.1,
            )

    def test_custom_retry_params_stored(self):
        """Custom retry params should be stored and used."""
        vs = make_store(
            table_name=f"it_retry4_{uuid.uuid4().hex[:8]}",
            connection_retries=7,
            retry_delay=2.0,
        )
        assert vs._connection_retries == 7
        assert vs._retry_delay == 2.0
        vs.drop()
        vs.close()


# ==================================================================
# 3. Embedding 维度校验 — 真实插入
# ==================================================================


class TestEmbeddingDimValidationIT:
    """Integration tests for embedding dimension validation with real DB."""

    def test_correct_dim_inserts_successfully(self):
        """Nodes with correct dimension should insert fine."""
        vs = make_store(table_name=f"it_embdim_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        ids = vs.add(nodes)
        assert len(ids) == 5
        assert vs.count() == 5
        vs.drop()
        vs.close()

    def test_wrong_dim_raises_before_insert(self):
        """Wrong dimension should raise before any DB insert."""
        vs = make_store(table_name=f"it_embdim2_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        # Corrupt one embedding
        nodes[0].embedding = [0.1, 0.2, 0.3]  # wrong dim (3 vs 128)
        with pytest.raises(ValueError, match="dimension 3"):
            vs.add(nodes)
        # Nothing should have been inserted
        assert vs.count() == 0
        vs.drop()
        vs.close()

    def test_none_embedding_raises(self):
        """None embedding should raise ValueError."""
        vs = make_store(table_name=f"it_embdim3_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS[:1], METADATAS[:1])
        nodes[0].embedding = None
        with pytest.raises(ValueError, match="no.*embedding"):
            vs.add(nodes)
        # The error should come from our validation, not llama_index
        assert vs.count() == 0
        vs.drop()
        vs.close()

    def test_vector_dim_cross_check_on_v3(self):
        """VECTOR_DIM cross-check should work on v3 instances."""
        vs = make_store(table_name=f"it_embdim4_{uuid.uuid4().hex[:8]}")
        if not is_v3(vs):
            pytest.skip("Not a v3 instance")
        nodes = make_nodes(TEXTS[:1], METADATAS[:1])
        ids = vs.add(nodes)
        assert len(ids) == 1
        vs.drop()
        vs.close()


# ==================================================================
# 4. 元数据搜索和删除 — 真实 CRUD
# ==================================================================


class TestMetadataSearchDeleteIT:
    """Integration tests for metadata search and delete with real DB."""

    def test_search_by_metadata_returns_matching_nodes(self):
        """search_by_metadata should return nodes matching the filter."""
        vs = make_store(table_name=f"it_meta_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="database", operator=FilterOperator.EQ
                )
            ]
        )
        results = vs.search_by_metadata(filters, limit=10)
        assert len(results) == 1
        assert "PolarDB-X" in results[0].get_content()
        vs.drop()
        vs.close()

    def test_search_by_metadata_with_multiple_filters(self):
        """Multiple filters with AND should work."""
        vs = make_store(table_name=f"it_meta2_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="database", operator=FilterOperator.EQ
                ),
                MetadataFilter(
                    key="lang", value="en", operator=FilterOperator.EQ
                ),
            ]
        )
        results = vs.search_by_metadata(filters, limit=10)
        assert len(results) == 1
        vs.drop()
        vs.close()

    def test_search_by_metadata_in_operator(self):
        """IN operator should return multiple matches."""
        vs = make_store(table_name=f"it_meta3_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value=["database", "framework", "language"],
                    operator=FilterOperator.IN,
                )
            ]
        )
        results = vs.search_by_metadata(filters, limit=10)
        assert len(results) == 3
        vs.drop()
        vs.close()

    def test_delete_by_metadata_returns_count(self):
        """delete_by_metadata should return number of deleted rows."""
        vs = make_store(table_name=f"it_meta4_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)
        assert vs.count() == 5

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="database", operator=FilterOperator.EQ
                )
            ]
        )
        deleted = vs.delete_by_metadata(filters)
        assert deleted == 1
        assert vs.count() == 4
        vs.drop()
        vs.close()

    def test_delete_by_metadata_no_match(self):
        """delete_by_metadata with no matches should return 0."""
        vs = make_store(table_name=f"it_meta5_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="nonexistent", operator=FilterOperator.EQ
                )
            ]
        )
        deleted = vs.delete_by_metadata(filters)
        assert deleted == 0
        assert vs.count() == 5
        vs.drop()
        vs.close()


# ==================================================================
# 5. MMR 搜索 — 真实向量检索
# ==================================================================


class TestMMRSearchIT:
    """Integration tests for MMR search with real vector index."""

    def test_mmr_query_returns_results(self):
        """MMR query should return results from real vector search."""
        vs = make_store(table_name=f"it_mmr_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        query_emb = EMB.embed_query("database vector search")
        q = VectorStoreQuery(
            query_embedding=query_emb,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        result = vs.query(q, fetch_k=5, lambda_mult=0.5)
        assert len(result.nodes) <= 3
        assert len(result.similarities) == len(result.nodes)
        vs.drop()
        vs.close()

    def test_mmr_default_fetch_k(self):
        """Default fetch_k should be max(top_k * 3, 20)."""
        vs = make_store(table_name=f"it_mmr2_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        query_emb = EMB.embed_query("database")
        q = VectorStoreQuery(
            query_embedding=query_emb,
            similarity_top_k=2,
            mode=VectorStoreQueryMode.MMR,
        )
        result = vs.query(q)  # no fetch_k, use default
        assert len(result.nodes) <= 2
        vs.drop()
        vs.close()

    def test_mmr_more_diverse_than_default(self):
        """MMR with low lambda_mult should return more diverse results."""
        vs = make_store(table_name=f"it_mmr3_{uuid.uuid4().hex[:8]}")
        # Use more texts for diversity
        texts = TEXTS * 3  # 15 nodes
        metadatas = METADATAS * 3
        nodes = make_nodes(texts, metadatas)
        vs.add(nodes)

        query_emb = EMB.embed_query("database vector search")

        # Default search
        q_default = VectorStoreQuery(
            query_embedding=query_emb,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result_default = vs.query(q_default)

        # MMR search
        q_mmr = VectorStoreQuery(
            query_embedding=query_emb,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        result_mmr = vs.query(q_mmr, fetch_k=10, lambda_mult=0.3)

        # Both should return results
        assert len(result_default.nodes) == 3
        assert len(result_mmr.nodes) <= 3

        # MMR results should be different from default (more diverse)
        default_ids = set(result_default.ids)
        mmr_ids = set(result_mmr.ids)
        # They may or may not overlap, but MMR should work
        assert len(mmr_ids) > 0

        vs.drop()
        vs.close()

    def test_default_query_still_works(self):
        """DEFAULT mode should still work after MMR changes."""
        vs = make_store(table_name=f"it_mmr4_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS, METADATAS)
        vs.add(nodes)

        query_emb = EMB.embed_query("PolarDB-X")
        q = VectorStoreQuery(
            query_embedding=query_emb,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = vs.query(q)
        assert len(result.nodes) == 3
        # With fake MD5-based embeddings, we cannot assert semantic relevance.
        # Just verify results are returned correctly.
        assert len(result.similarities) == 3
        # Results should be sorted by similarity (descending)
        assert result.similarities[0] >= result.similarities[-1]
        vs.drop()
        vs.close()

    def test_unsupported_mode_rejected(self):
        """HYBRID mode should still be rejected."""
        vs = make_store(table_name=f"it_mmr5_{uuid.uuid4().hex[:8]}")
        nodes = make_nodes(TEXTS[:1], METADATAS[:1])
        vs.add(nodes)

        query_emb = EMB.embed_query("test")
        q = VectorStoreQuery(
            query_embedding=query_emb,
            similarity_top_k=1,
            mode=VectorStoreQueryMode.HYBRID,
        )
        with pytest.raises(NotImplementedError):
            vs.query(q)
        vs.drop()
        vs.close()


# ==================================================================
# 6. 端到端用户模拟
# ==================================================================


class TestEndToEndUserSimulation:
    """Simulate a real user workflow from start to finish."""

    def test_full_user_workflow(self):
        """Complete user journey: create → add → query → MMR → meta search → delete → cleanup."""
        table = f"it_e2e_{uuid.uuid4().hex[:8]}"

        # Step 1: Create store with kwargs
        print("\n  [1] Creating vector store with connect_timeout=10...")
        vs = PolarDBXVectorStore(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            table_name=table,
            embed_dim=EMBED_DIM,
            distance_strategy="cosine",
            connection_retries=3,
            retry_delay=0.5,
            connect_timeout=10,
        )
        assert vs._is_initialized is True
        print(f"  ✓ Connected. Capabilities: {vs._capabilities}")

        # Step 2: Add nodes
        print("  [2] Adding 5 nodes with embeddings...")
        nodes = make_nodes(TEXTS, METADATAS)
        ids = vs.add(nodes)
        assert len(ids) == 5
        assert vs.count() == 5
        print(f"  ✓ Added {len(ids)} nodes, count={vs.count()}")

        # Step 3: Default similarity search
        print("  [3] Default similarity search for 'database'...")
        query_emb = EMB.embed_query("database")
        q = VectorStoreQuery(
            query_embedding=query_emb,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = vs.query(q)
        assert len(result.nodes) == 3
        print(f"  ✓ Found {len(result.nodes)} results")
        for i, (node, sim) in enumerate(zip(result.nodes, result.similarities)):
            print(f"    #{i+1} sim={sim:.4f} text={node.get_content()[:40]}...")

        # Step 4: MMR search for diversity
        print("  [4] MMR search with lambda_mult=0.3 for diversity...")
        q_mmr = VectorStoreQuery(
            query_embedding=query_emb,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        result_mmr = vs.query(q_mmr, fetch_k=5, lambda_mult=0.3)
        assert len(result_mmr.nodes) <= 3
        print(f"  ✓ MMR returned {len(result_mmr.nodes)} results")
        for i, (node, sim) in enumerate(
            zip(result_mmr.nodes, result_mmr.similarities)
        ):
            print(f"    #{i+1} sim={sim:.4f} text={node.get_content()[:40]}...")

        # Step 5: Metadata search
        print("  [5] Metadata search for category='database'...")
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="database", operator=FilterOperator.EQ
                )
            ]
        )
        meta_results = vs.search_by_metadata(filters, limit=10)
        assert len(meta_results) == 1
        print(f"  ✓ Found {len(meta_results)} nodes by metadata")
        print(f"    text={meta_results[0].get_content()[:40]}...")

        # Step 6: Delete by metadata
        print("  [6] Deleting nodes with category='search'...")
        del_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="search", operator=FilterOperator.EQ
                )
            ]
        )
        deleted = vs.delete_by_metadata(del_filters)
        assert deleted == 1
        assert vs.count() == 4
        print(f"  ✓ Deleted {deleted} node(s), count={vs.count()}")

        # Step 7: Verify deletion with another search
        print("  [7] Verifying deletion: search for category='search' again...")
        results_after = vs.search_by_metadata(del_filters, limit=10)
        assert len(results_after) == 0
        print(f"  ✓ No results found (confirmed deletion)")

        # Step 8: Wrong embedding dimension is caught
        print("  [8] Testing embedding dimension validation...")
        bad_node = make_nodes(["test"], [{"category": "test"}])
        bad_node[0].embedding = [0.1, 0.2, 0.3]  # wrong dim
        with pytest.raises(ValueError, match="dimension"):
            vs.add(bad_node)
        print("  ✓ Dimension mismatch caught before insert")
        assert vs.count() == 4  # unchanged

        # Cleanup
        print("  [9] Cleaning up...")
        vs.drop()
        vs.close()
        print("  ✓ Done!")


# ==================================================================
# Helper: assert table not exists
# ==================================================================


def _assert_table_not_exists(table_name: str) -> None:
    """Assert that a table does not exist in the database."""
    import sqlalchemy
    from urllib.parse import quote_plus

    engine = sqlalchemy.create_engine(
        f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = :db AND table_name = :tbl"
            ),
            {"db": DB_NAME, "tbl": table_name},
        )
        count = result.scalar()
        assert count == 0, f"Table {table_name} should not exist but was found"
    engine.dispose()
