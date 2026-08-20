"""Unit tests for _reconstruct_node fallback behavior."""

from unittest.mock import patch

import pytest
from llama_index.core.schema import TextNode

from llama_index.vector_stores.polardbx import PolarDBXVectorStore


def _make_store(**overrides):
    """Create a PolarDBXVectorStore with _initialize mocked out."""
    defaults = dict(
        host="localhost",
        port=3306,
        user="root",
        password="pw",
        database="testdb",
        table_name="test_table",
        embed_dim=4,
        perform_setup=False,
    )
    defaults.update(overrides)
    with patch.object(PolarDBXVectorStore, "_initialize", lambda self: None):
        return PolarDBXVectorStore(**defaults)


class TestReconstructNode:
    """Tests for _reconstruct_node() method."""

    def test_reconstruct_with_internal_keys(self):
        """When _node_content is present, uses metadata_dict_to_node."""
        vs = _make_store()
        # Simulate metadata with internal serialization keys
        node = TextNode(id_="node-123", text="hello world", metadata={"cat": "a"})
        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        full_meta = node_to_metadata_dict(node, remove_text=True)

        result = vs._reconstruct_node(full_meta, "hello world", "node-123")
        assert result.node_id == "node-123"
        assert result.get_content() == "hello world"

    def test_reconstruct_without_internal_keys(self):
        """When _node_content is absent, falls back to TextNode."""
        vs = _make_store()
        metadata = {"category": "database", "lang": "en"}

        result = vs._reconstruct_node(metadata, "some text", "node-456")
        assert result.node_id == "node-456"
        assert result.get_content() == "some text"
        assert result.metadata.get("category") == "database"
        assert result.metadata.get("lang") == "en"

    def test_reconstruct_empty_metadata(self):
        """Empty metadata dict creates a TextNode with no metadata."""
        vs = _make_store()

        result = vs._reconstruct_node({}, "text only", "node-789")
        assert result.node_id == "node-789"
        assert result.get_content() == "text only"
        assert result.metadata == {}

    def test_reconstruct_strips_underscore_keys_in_fallback(self):
        """Fallback mode strips internal keys (starting with _)."""
        vs = _make_store()
        # Simulate metadata with both user keys and internal-like keys
        metadata = {
            "category": "db",
            "_internal_key": "should_be_stripped",
            "lang": "en",
        }

        result = vs._reconstruct_node(metadata, "text", "node-1")
        assert "category" in result.metadata
        assert "lang" in result.metadata
        assert "_internal_key" not in result.metadata

    def test_reconstruct_with_custom_columns_no_json(self):
        """No-JSON-column mode: fallback path is used."""
        vs = _make_store(
            id_column="cid",
            node_id_column="cnode",
            text_column="ctext",
            embedding_column="cemb",
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        # Simulate metadata from _record_to_metadata (no _node_content)
        metadata = {"category": "database", "lang": "en"}

        result = vs._reconstruct_node(metadata, "db text", "node-1")
        assert result.node_id == "node-1"
        assert result.get_content() == "db text"
        assert result.metadata.get("category") == "database"

    def test_reconstruct_node_id_from_metadata_when_none(self):
        """When node_id is None, falls back to metadata['node_id']."""
        vs = _make_store()
        metadata = {"node_id": "from-meta", "category": "x"}

        result = vs._reconstruct_node(metadata, "text", None)
        assert result.node_id == "from-meta"

    def test_reconstruct_default_schema_uses_dict_to_node(self):
        """Default schema still uses metadata_dict_to_node path."""
        vs = _make_store()
        assert not vs._has_custom_columns

        node = TextNode(id_="def-1", text="default text", metadata={"k": "v"})
        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        full_meta = node_to_metadata_dict(node, remove_text=True)

        result = vs._reconstruct_node(full_meta, "default text", "def-1")
        assert result.node_id == "def-1"
        assert result.get_content() == "default text"
