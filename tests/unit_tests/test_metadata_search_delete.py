"""Unit tests for metadata search and delete — no DB required."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from llama_index.vector_stores.polardbx import PolarDBXVectorStore
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)


def make_mock_row(node_id="n1", text="hello", metadata=None):
    """Create a mock DB row (node_id, text, metadata)."""
    return MagicMock(
        __iter__=lambda *a: iter([(0, node_id, text, metadata or {})]),
        __getitem__=lambda s, i: [node_id, text, json_or_str(metadata)][i],
    )


import json


def json_or_str(metadata):
    if metadata is None:
        return "{}"
    if isinstance(metadata, str):
        return metadata
    return json.dumps(metadata)


class TestMetadataSearch:
    """Tests for search_by_metadata and asearch_by_metadata."""

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
        store._capabilities = {}
        return store

    def test_search_by_metadata_builds_correct_sql(self):
        """search_by_metadata should build correct WHERE clause."""
        store = self._make_store()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="phone", operator=FilterOperator.EQ
                )
            ]
        )
        store.search_by_metadata(filters, limit=5)

        # Verify SQL was executed
        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "SELECT" in sql_text
        assert "node_id" in sql_text
        assert "text" in sql_text
        assert "metadata" in sql_text
        assert "WHERE" in sql_text
        assert "LIMIT" in sql_text

        # Verify limit param
        params = call_args[0][1]
        assert params["limit"] == 5

    def test_search_by_metadata_returns_nodes(self):
        """search_by_metadata should return BaseNode objects."""
        store = self._make_store()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda s, i: [
            "node-1",
            "hello world",
            '{"category": "phone"}',
        ][i]
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([mock_row]))
        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        # Mock metadata_dict_to_node to avoid needing _node_content
        from llama_index.vector_stores.polardbx import base as base_mod
        mock_node = MagicMock()
        mock_node.node_id = "node-1"
        with patch.object(
            base_mod, "metadata_dict_to_node", return_value=mock_node
        ):
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category",
                        value="phone",
                        operator=FilterOperator.EQ,
                    )
                ]
            )
            nodes = store.search_by_metadata(filters, limit=10)

        assert len(nodes) == 1
        assert nodes[0].node_id == "node-1"

    def test_search_by_metadata_default_limit(self):
        """Default limit should be 10."""
        store = self._make_store()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="status", value="active", operator=FilterOperator.EQ
                )
            ]
        )
        store.search_by_metadata(filters)
        call_args = mock_session.execute.call_args
        params = call_args[0][1]
        assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_asearch_by_metadata_builds_correct_sql(self):
        """asearch_by_metadata should build correct WHERE clause."""
        store = self._make_store()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_async_cm = AsyncMock()
        mock_async_cm.__aenter__.return_value = mock_session
        mock_async_cm.__aexit__.return_value = False
        store._async_session = MagicMock(return_value=mock_async_cm)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="phone", operator=FilterOperator.EQ
                )
            ]
        )
        await store.asearch_by_metadata(filters, limit=5)

        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "SELECT" in sql_text
        assert "WHERE" in sql_text
        params = call_args[0][1]
        assert params["limit"] == 5


class TestMetadataDelete:
    """Tests for delete_by_metadata and adelete_by_metadata."""

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
        store._capabilities = {}
        return store

    def test_delete_by_metadata_returns_count(self):
        """delete_by_metadata should return number of deleted rows."""
        store = self._make_store()
        mock_result = MagicMock()
        mock_result.rowcount = 42
        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="phone", operator=FilterOperator.EQ
                )
            ]
        )
        count = store.delete_by_metadata(filters)

        assert count == 42
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "DELETE" in sql_text
        assert "WHERE" in sql_text

    def test_delete_by_metadata_zero_rows(self):
        """delete_by_metadata should return 0 when no rows match."""
        store = self._make_store()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="nonexistent", operator=FilterOperator.EQ
                )
            ]
        )
        count = store.delete_by_metadata(filters)
        assert count == 0

    @pytest.mark.asyncio
    async def test_adelete_by_metadata_returns_count(self):
        """adelete_by_metadata should return number of deleted rows."""
        store = self._make_store()
        mock_result = MagicMock()
        mock_result.rowcount = 7
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_async_cm = AsyncMock()
        mock_async_cm.__aenter__.return_value = mock_session
        mock_async_cm.__aexit__.return_value = False
        store._async_session = MagicMock(return_value=mock_async_cm)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="status", value="deleted", operator=FilterOperator.EQ
                )
            ]
        )
        count = await store.adelete_by_metadata(filters)

        assert count == 7
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_delete_by_metadata_multiple_filters(self):
        """delete_by_metadata should handle multiple filters with AND."""
        store = self._make_store()
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category", value="phone", operator=FilterOperator.EQ
                ),
                MetadataFilter(
                    key="price", value=1000, operator=FilterOperator.GT
                ),
            ]
        )
        count = store.delete_by_metadata(filters)

        assert count == 3
        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "DELETE" in sql_text
        assert "WHERE" in sql_text
        assert "AND" in sql_text

    def test_search_by_metadata_with_in_operator(self):
        """search_by_metadata should support IN operator."""
        store = self._make_store()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value=["phone", "laptop"],
                    operator=FilterOperator.IN,
                )
            ]
        )
        store.search_by_metadata(filters, limit=5)

        call_args = mock_session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "IN" in sql_text
