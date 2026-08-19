"""Unit tests for embedding dimension validation — no DB required."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from llama_index.vector_stores.polardbx import PolarDBXVectorStore
from llama_index.vector_stores.polardbx.base import NotSupportedError


def make_mock_node(node_id: str, embedding=None, text: str = "test"):
    """Create a mock BaseNode with the given embedding."""
    node = MagicMock()
    node.node_id = node_id
    node.get_content.return_value = text
    node.get_embedding.return_value = embedding
    node.embedding = embedding
    return node


class TestEmbeddingDimensionValidation:
    """Tests for _validate_embedding_dimensions."""

    def _make_store(self, embed_dim=4):
        """Create a store with mocked initialization."""
        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                embed_dim=embed_dim,
            )
        store._is_initialized = True
        store._capabilities = {"vec_dim": False}
        store._session = MagicMock()
        return store

    def test_correct_dimension_passes(self):
        """Embeddings with matching dimension should pass."""
        store = self._make_store(embed_dim=4)
        nodes = [
            make_mock_node("n1", [0.1, 0.2, 0.3, 0.4]),
            make_mock_node("n2", [0.5, 0.6, 0.7, 0.8]),
        ]
        store._validate_embedding_dimensions(nodes)  # should not raise

    def test_mismatched_dimension_raises(self):
        """Embedding with wrong dimension should raise ValueError."""
        store = self._make_store(embed_dim=4)
        nodes = [make_mock_node("n1", [0.1, 0.2, 0.3])]
        with pytest.raises(ValueError, match="dimension 3.*expected 4"):
            store._validate_embedding_dimensions(nodes)

    def test_none_embedding_raises(self):
        """Node with no embedding should raise ValueError."""
        store = self._make_store(embed_dim=4)
        nodes = [make_mock_node("n1", None)]
        with pytest.raises(ValueError, match="no.*embedding"):
            store._validate_embedding_dimensions(nodes)

    def test_mixed_dimensions_raises_on_first_mismatch(self):
        """Should raise on the first mismatched embedding."""
        store = self._make_store(embed_dim=4)
        nodes = [
            make_mock_node("n1", [0.1, 0.2, 0.3, 0.4]),
            make_mock_node("n2", [0.1, 0.2, 0.3]),  # wrong dim
            make_mock_node("n3", [0.1, 0.2, 0.3, 0.4]),
        ]
        with pytest.raises(ValueError, match="index 1.*dimension 3"):
            store._validate_embedding_dimensions(nodes)

    def test_empty_list_passes(self):
        """Empty node list should pass."""
        store = self._make_store(embed_dim=4)
        store._validate_embedding_dimensions([])  # should not raise

    def test_error_message_includes_node_id(self):
        """Error message should include the node_id for debugging."""
        store = self._make_store(embed_dim=4)
        nodes = [make_mock_node("my-node-42", [0.1, 0.2, 0.3])]
        with pytest.raises(ValueError, match="my-node-42"):
            store._validate_embedding_dimensions(nodes)

    def test_single_node_correct_dim(self):
        """Single node with correct dimension should pass."""
        store = self._make_store(embed_dim=3)
        nodes = [make_mock_node("n1", [1.0, 0.0, 0.0])]
        store._validate_embedding_dimensions(nodes)  # should not raise

    def test_vector_dim_cross_check_skipped_when_unavailable(self):
        """VECTOR_DIM cross-check should be skipped on old versions."""
        store = self._make_store(embed_dim=4)
        store._capabilities = {"vec_dim": False}
        nodes = [make_mock_node("n1", [0.1, 0.2, 0.3, 0.4])]
        # Mock session to ensure it's NOT called for VECTOR_DIM
        mock_session = MagicMock()
        store._session = MagicMock(return_value=mock_session)
        store._validate_embedding_dimensions(nodes)
        # Session should not be opened for cross-check
        store._session.__enter__.assert_not_called()

    def test_vector_dim_cross_check_called_when_available(self):
        """VECTOR_DIM cross-check should run on v3 instances."""
        store = self._make_store(embed_dim=4)
        store._capabilities = {"vec_dim": True}

        # Mock session and result
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: 4  # VECTOR_DIM returns 4
        mock_result.fetchone.return_value = mock_row

        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        store._session = MagicMock(return_value=mock_session)
        nodes = [make_mock_node("n1", [0.1, 0.2, 0.3, 0.4])]
        store._validate_embedding_dimensions(nodes)  # should not raise
        # Verify VECTOR_DIM was queried
        mock_session.execute.assert_called()

    def test_vector_dim_cross_check_mismatch_raises(self):
        """VECTOR_DIM reporting different dim should raise ValueError."""
        store = self._make_store(embed_dim=4)
        store._capabilities = {"vec_dim": True}

        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: 3  # mismatch!
        mock_result.fetchone.return_value = mock_row

        mock_session = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        store._session = MagicMock(return_value=mock_session)
        nodes = [make_mock_node("n1", [0.1, 0.2, 0.3, 0.4])]
        with pytest.raises(ValueError, match="DN VECTOR_DIM reports 3"):
            store._validate_embedding_dimensions(nodes)

    def test_add_calls_validation(self):
        """add() should call _validate_embedding_dimensions."""
        store = self._make_store(embed_dim=4)
        store._validate_embedding_dimensions = MagicMock()
        store._node_to_table_row = MagicMock(
            return_value={"node_id": "n1", "text": "t", "embedding": [0.1], "metadata": {}}
        )
        mock_session = MagicMock()
        mock_session.execute.return_value = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        store._session = MagicMock(return_value=mock_session)
        nodes = [make_mock_node("n1", [0.1, 0.2, 0.3, 0.4])]
        store.add(nodes)
        store._validate_embedding_dimensions.assert_called_once_with(nodes)

    def test_add_with_wrong_dim_raises_before_insert(self):
        """add() should raise before any DB insert on dimension mismatch."""
        store = self._make_store(embed_dim=4)
        store._session = MagicMock()
        nodes = [make_mock_node("n1", [0.1, 0.2, 0.3])]  # wrong dim
        with pytest.raises(ValueError, match="dimension 3"):
            store.add(nodes)
        # No DB session should have been opened for insert
        store._session.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_add_calls_validation(self):
        """async_add() should call _validate_embedding_dimensions."""
        store = self._make_store(embed_dim=4)
        store._validate_embedding_dimensions = MagicMock()
        store._node_to_table_row = MagicMock(
            return_value={"node_id": "n1", "text": "t", "embedding": [0.1], "metadata": {}}
        )
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_async_cm = AsyncMock()
        mock_async_cm.__aenter__.return_value = mock_session
        mock_async_cm.__aexit__.return_value = False
        store._async_session = MagicMock(return_value=mock_async_cm)
        nodes = [make_mock_node("n1", [0.1, 0.2, 0.3, 0.4])]
        await store.async_add(nodes)
        store._validate_embedding_dimensions.assert_called_once_with(nodes)
