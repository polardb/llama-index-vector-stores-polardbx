"""Unit tests for MMR search — no DB required."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from llama_index.vector_stores.polardbx import PolarDBXVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)


class TestMMRAlgorithm:
    """Tests for the _maximal_marginal_relevance static method."""

    def test_empty_embedding_list(self):
        """Empty list returns empty selection."""
        result = PolarDBXVectorStore._maximal_marginal_relevance(
            query_embedding=[1.0, 0.0, 0.0],
            embedding_list=[],
            k=4,
        )
        assert result == []

    def test_single_embedding(self):
        """Single embedding returns index 0."""
        result = PolarDBXVectorStore._maximal_marginal_relevance(
            query_embedding=[1.0, 0.0, 0.0],
            embedding_list=[[1.0, 0.0, 0.0]],
            k=4,
        )
        assert result == [0]

    def test_selects_most_similar_first(self):
        """First selected should be the most similar to query."""
        query = [1.0, 0.0, 0.0]
        embeddings = [
            [0.5, 0.5, 0.0],  # less similar
            [0.9, 0.1, 0.0],  # most similar
            [0.1, 0.9, 0.0],  # least similar
        ]
        result = PolarDBXVectorStore._maximal_marginal_relevance(
            query_embedding=query,
            embedding_list=embeddings,
            k=1,
        )
        assert result == [1]  # index 1 is most similar

    def test_diversity_with_lambda_zero(self):
        """lambda_mult=0 means max diversity."""
        query = [1.0, 0.0, 0.0]
        # All embeddings are identical, so diversity doesn't matter
        embeddings = [
            [0.9, 0.1, 0.0],
            [0.9, 0.1, 0.0],
            [0.9, 0.1, 0.0],
        ]
        result = PolarDBXVectorStore._maximal_marginal_relevance(
            query_embedding=query,
            embedding_list=embeddings,
            k=2,
            lambda_mult=0.0,
        )
        assert len(result) == 2

    def test_returns_correct_count(self):
        """Should return exactly k items when enough candidates."""
        import numpy as np

        np.random.seed(42)
        query = [1.0, 0.0, 0.0, 0.0]
        embeddings = [
            list(np.random.randn(4)) for _ in range(20)
        ]
        result = PolarDBXVectorStore._maximal_marginal_relevance(
            query_embedding=query,
            embedding_list=embeddings,
            k=5,
            lambda_mult=0.5,
        )
        assert len(result) == 5
        # All indices should be unique
        assert len(set(result)) == 5

    def test_k_larger_than_list(self):
        """k > len(embeddings) should return all."""
        query = [1.0, 0.0]
        embeddings = [[0.9, 0.1], [0.1, 0.9]]
        result = PolarDBXVectorStore._maximal_marginal_relevance(
            query_embedding=query,
            embedding_list=embeddings,
            k=10,
        )
        assert len(result) == 2


class TestMMRQuery:
    """Tests for MMR mode in query() and aquery()."""

    def _make_store(self):
        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                embed_dim=4,
            )
        store._is_initialized = True
        store._capabilities = {"vec_totext": True}
        return store

    def _make_mock_rows(self, count=10, dim=4):
        """Create mock DB rows for query results."""
        rows = []
        for i in range(count):
            row = MagicMock()
            row.__getitem__ = lambda s, idx, i=i: {
                0: f"node-{i}",
                1: f"text-{i}",
                2: '{"key": "val"}',
                3: 0.1 * i,
            }[idx]
            rows.append(row)
        return rows

    def test_mmr_mode_not_rejected(self):
        """query() should accept MMR mode."""
        store = self._make_store()

        # Mock the DB query
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)
        store._set_ef_search_sync = MagicMock()

        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            similarity_top_k=2,
            mode=VectorStoreQueryMode.MMR,
        )
        result = store.query(q, fetch_k=5, lambda_mult=0.5)
        assert isinstance(result, VectorStoreQueryResult)

    def test_mmr_fetch_k_default(self):
        """Default fetch_k should be max(top_k * 3, 20)."""
        store = self._make_store()

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)
        store._set_ef_search_sync = MagicMock()

        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            similarity_top_k=5,
            mode=VectorStoreQueryMode.MMR,
        )
        store.query(q)
        call_args = mock_session.execute.call_args
        params = call_args[0][1]
        # max(5*3, 20) = 20
        assert params["limit"] == 20

    def test_mmr_fetch_k_custom(self):
        """Custom fetch_k should be used."""
        store = self._make_store()

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)
        store._set_ef_search_sync = MagicMock()

        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            similarity_top_k=2,
            mode=VectorStoreQueryMode.MMR,
        )
        store.query(q, fetch_k=10)
        call_args = mock_session.execute.call_args
        params = call_args[0][1]
        assert params["limit"] == 10

    def test_default_mode_still_works(self):
        """DEFAULT mode should not trigger MMR re-ranking."""
        store = self._make_store()

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)
        store._set_ef_search_sync = MagicMock()

        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        store.query(q)
        call_args = mock_session.execute.call_args
        params = call_args[0][1]
        # For default mode, limit should be similarity_top_k
        assert params["limit"] == 5

    def test_unsupported_mode_still_rejected(self):
        """Modes other than DEFAULT and MMR should be rejected."""
        store = self._make_store()
        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            similarity_top_k=2,
            mode=VectorStoreQueryMode.HYBRID,
        )
        with pytest.raises(NotImplementedError):
            store.query(q)

    @pytest.mark.asyncio
    async def test_ammr_mode_works(self):
        """aquery() should accept MMR mode."""
        store = self._make_store()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_async_cm = AsyncMock()
        mock_async_cm.__aenter__.return_value = mock_session
        mock_async_cm.__aexit__.return_value = False
        store._async_session = MagicMock(return_value=mock_async_cm)
        store._set_ef_search_async = AsyncMock()

        q = VectorStoreQuery(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            similarity_top_k=2,
            mode=VectorStoreQueryMode.MMR,
        )
        result = await store.aquery(q, fetch_k=5)
        assert isinstance(result, VectorStoreQueryResult)


class TestFetchEmbeddings:
    """Tests for _fetch_embeddings_by_node_ids."""

    def _make_store(self):
        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                embed_dim=4,
            )
        store._is_initialized = True
        store._capabilities = {"vec_totext": True}
        return store

    def test_empty_list_returns_empty(self):
        """Empty node_ids should return empty dict."""
        store = self._make_store()
        result = store._fetch_embeddings_by_node_ids([])
        assert result == {}

    def test_fetch_returns_dict(self):
        """Should return dict mapping node_id to embedding."""
        store = self._make_store()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda s, i: ["node-1", "[0.1, 0.2]"][i]
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        result = store._fetch_embeddings_by_node_ids(["node-1"])
        assert "node-1" in result
        assert result["node-1"] == [0.1, 0.2]

    def test_uses_vec_totext_when_available(self):
        """Should use VEC_TOTEXT when v3 capabilities are available."""
        store = self._make_store()
        store._capabilities = {"vec_totext": True}
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        store._fetch_embeddings_by_node_ids(["n1"])
        sql = str(mock_session.execute.call_args[0][0])
        assert "VEC_TOTEXT" in sql

    def test_uses_cast_when_vec_totext_unavailable(self):
        """Should use CAST when v3 is not available."""
        store = self._make_store()
        store._capabilities = {"vec_totext": False}
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        store._fetch_embeddings_by_node_ids(["n1"])
        sql = str(mock_session.execute.call_args[0][0])
        assert "CAST" in sql
        assert "VEC_TOTEXT" not in sql
