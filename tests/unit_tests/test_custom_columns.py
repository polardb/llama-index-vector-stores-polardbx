"""Unit tests for PolarDBXVectorStore custom column constructor validation.

These tests verify validation logic that runs before any database
connection is attempted.  They mock out _initialize to avoid needing
a real PolarDB-X instance.
"""

from unittest.mock import patch

import pytest

from llama_index.vector_stores.polardbx import Column, PolarDBXVectorStore


def _make_store(**overrides):
    """Create a PolarDBXVectorStore with _initialize mocked out.

    All database connection parameters are fake; only the validation
    logic is exercised.
    """
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


class TestConstructorDefaults:
    """Default column configuration is the classic 5-column schema."""

    def test_default_column_names(self):
        """Without custom params, column names are the defaults."""
        store = _make_store()
        assert store._id_column == "id"
        assert store._node_id_column == "node_id"
        assert store._text_column == "text"
        assert store._embedding_column == "embedding"
        assert store._metadata_json_column == "metadata"
        assert store._metadata_column_objs == []
        assert store._metadata_column_names == []
        assert store._has_custom_columns is False

    def test_has_custom_columns_true_with_renamed_id(self):
        """Renaming id_column triggers custom mode."""
        store = _make_store(id_column="uid")
        assert store._id_column == "uid"
        assert store._has_custom_columns is True

    def test_has_custom_columns_true_with_metadata_cols(self):
        """Adding metadata_columns triggers custom mode."""
        store = _make_store(
            metadata_columns=[Column("price", "DECIMAL(10,2)")]
        )
        assert store._has_custom_columns is True
        assert "price" in store._metadata_column_names

    def test_has_custom_columns_true_without_json(self):
        """Disabling metadata_json_column triggers custom mode."""
        store = _make_store(
            metadata_json_column=None,
            metadata_columns=["category"],
        )
        assert store._has_custom_columns is True
        assert store._metadata_json_column is None


class TestColumnNormalization:
    """metadata_columns normalization: Column objects vs strings."""

    def test_column_objects_extracted(self):
        """Column objects are split into objs and names."""
        cols = [
            Column("price", "DECIMAL(10,2)", nullable=False),
            Column("category", "VARCHAR(100)"),
        ]
        store = _make_store(metadata_columns=cols)
        assert len(store._metadata_column_objs) == 2
        assert store._metadata_column_names == ["price", "category"]

    def test_string_columns_extracted(self):
        """String names are added to names list only."""
        store = _make_store(metadata_columns=["price", "category"])
        assert store._metadata_column_objs == []
        assert store._metadata_column_names == ["price", "category"]

    def test_mixed_column_types(self):
        """Mix of Column and string is valid."""
        store = _make_store(
            metadata_columns=[
                Column("price", "DECIMAL(10,2)"),
                "category",
            ]
        )
        assert len(store._metadata_column_objs) == 1
        assert store._metadata_column_names == ["price", "category"]

    def test_invalid_metadata_column_type(self):
        """Non-Column, non-string items raise TypeError."""
        with pytest.raises(TypeError, match="must be Column or str"):
            _make_store(metadata_columns=[123])


class TestDuplicateDetection:
    """Duplicate column name detection."""

    def test_duplicate_core_column(self):
        """Renaming a core column to match another is rejected."""
        with pytest.raises(ValueError, match="Duplicate column name"):
            _make_store(id_column="node_id")

    def test_metadata_col_overlaps_core(self):
        """Metadata column name matching a core column is rejected."""
        with pytest.raises(ValueError, match="Duplicate column name"):
            _make_store(metadata_columns=["text"])

    def test_duplicate_metadata_cols(self):
        """Duplicate metadata column names are rejected."""
        with pytest.raises(ValueError, match="Duplicate column name"):
            _make_store(
                metadata_columns=[
                    Column("price", "DECIMAL(10,2)"),
                    Column("price", "INT"),
                ]
            )


class TestNoStorageTarget:
    """When metadata_json_column is None and no metadata_columns."""

    def test_no_json_no_columns_raises(self):
        """Cannot store metadata without a JSON column or mapped columns."""
        with pytest.raises(ValueError, match="no place to store"):
            _make_store(metadata_json_column=None)

    def test_no_json_with_columns_ok(self):
        """No JSON column is OK if metadata_columns are provided."""
        store = _make_store(
            metadata_json_column=None,
            metadata_columns=["category"],
        )
        assert store._metadata_json_column is None
        assert store._has_custom_columns is True


class TestIdentifierValidation:
    """Custom column names must pass identifier validation."""

    def test_invalid_id_column(self):
        """Invalid id_column name is rejected."""
        with pytest.raises(ValueError, match="Invalid identifier"):
            _make_store(id_column="123bad")

    def test_invalid_metadata_column_name(self):
        """Invalid metadata column name is rejected."""
        with pytest.raises(ValueError, match="Invalid identifier"):
            _make_store(metadata_columns=["bad name!"])

    def test_invalid_column_name_in_column_obj(self):
        """Column object with invalid name is rejected."""
        with pytest.raises(ValueError, match="Invalid identifier"):
            _make_store(
                metadata_columns=[Column("bad name!", "VARCHAR(10)")]
            )
