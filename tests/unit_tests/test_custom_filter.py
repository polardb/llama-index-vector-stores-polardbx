"""Unit tests for custom column filter clause branching."""

from unittest.mock import patch

import pytest
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from llama_index.vector_stores.polardbx import Column, PolarDBXVectorStore


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


class TestFilterClauseBranching:
    """Tests for _build_filter_clause with custom columns."""

    def test_default_schema_uses_json_extract(self):
        """Default schema uses JSON_EXTRACT on 'metadata' column."""
        store = _make_store()
        f = MetadataFilter(key="category", value="books", operator=FilterOperator.EQ)
        counter = [0]
        clause, params = store._build_filter_clause(f, counter)
        assert "JSON_UNQUOTE(JSON_EXTRACT(metadata" in clause
        assert "'$.category'" in clause
        assert "=" in clause
        assert "param_0" in params
        assert params["param_0"] == "books"

    def test_mapped_column_uses_direct_reference(self):
        """When key matches a metadata column, use direct column ref."""
        store = _make_store(metadata_columns=[Column("price", "DECIMAL(10,2)")])
        f = MetadataFilter(key="price", value=10, operator=FilterOperator.EQ)
        counter = [0]
        clause, params = store._build_filter_clause(f, counter)
        assert "`price`" in clause
        assert "JSON_EXTRACT" not in clause
        assert params["param_0"] == 10

    def test_unmapped_key_uses_json_extract_on_custom_col(self):
        """Unmapped key falls back to JSON_EXTRACT on custom json column."""
        store = _make_store(
            metadata_json_column="meta",
            metadata_columns=[Column("price", "DECIMAL(10,2)")],
        )
        f = MetadataFilter(key="category", value="books", operator=FilterOperator.EQ)
        counter = [0]
        clause, params = store._build_filter_clause(f, counter)
        assert "JSON_UNQUOTE(JSON_EXTRACT(`meta`" in clause
        assert "'$.category'" in clause
        assert "JSON_EXTRACT(metadata" not in clause

    def test_no_json_column_unmapped_key_raises(self):
        """Unmapped key with no JSON column raises ValueError."""
        store = _make_store(
            metadata_json_column=None,
            metadata_columns=["price"],
        )
        f = MetadataFilter(key="category", value="books", operator=FilterOperator.EQ)
        counter = [0]
        with pytest.raises(ValueError, match="Cannot filter on 'category'"):
            store._build_filter_clause(f, counter)

    def test_mapped_column_with_ne_operator(self):
        """Mapped column works with NE operator."""
        store = _make_store(metadata_columns=["status"])
        f = MetadataFilter(key="status", value="deleted", operator=FilterOperator.NE)
        counter = [0]
        clause, params = store._build_filter_clause(f, counter)
        assert "`status`" in clause
        assert "!=" in clause
        assert params["param_0"] == "deleted"

    def test_mapped_column_with_gt_operator(self):
        """Mapped column works with GT operator."""
        store = _make_store(metadata_columns=[Column("price", "DECIMAL(10,2)")])
        f = MetadataFilter(key="price", value=100, operator=FilterOperator.GT)
        counter = [0]
        clause, params = store._build_filter_clause(f, counter)
        assert "`price` >" in clause
        assert params["param_0"] == 100

    def test_mapped_column_with_in_operator(self):
        """Mapped column works with IN operator."""
        store = _make_store(metadata_columns=["category"])
        f = MetadataFilter(
            key="category",
            value=["books", "electronics"],
            operator=FilterOperator.IN,
        )
        counter = [0]
        clause, params = store._build_filter_clause(f, counter)
        assert "`category`" in clause
        assert "IN" in clause
        assert len(params) == 2
        assert params["param_0"] == "books"
        assert params["param_1"] == "electronics"

    def test_custom_json_column_name(self):
        """Custom json column name is used in JSON_EXTRACT."""
        store = _make_store(metadata_json_column="meta_json")
        f = MetadataFilter(key="tags", value="new", operator=FilterOperator.EQ)
        counter = [0]
        clause, params = store._build_filter_clause(f, counter)
        assert "JSON_EXTRACT(`meta_json`" in clause

    def test_default_schema_in_operator(self):
        """Default schema IN operator still works."""
        store = _make_store()
        f = MetadataFilter(
            key="category",
            value=["a", "b"],
            operator=FilterOperator.IN,
        )
        counter = [0]
        clause, params = store._build_filter_clause(f, counter)
        assert "JSON_UNQUOTE(JSON_EXTRACT(metadata" in clause
        assert "IN" in clause
        assert len(params) == 2

    def test_multiple_filters_mixed(self):
        """Multiple filters with mixed mapped/unmapped keys."""
        store = _make_store(
            metadata_json_column="meta",
            metadata_columns=[Column("price", "DECIMAL(10,2)")],
        )
        filters = MetadataFilters(filters=[
            MetadataFilter(key="price", value=10, operator=FilterOperator.GT),
            MetadataFilter(key="category", value="books", operator=FilterOperator.EQ),
        ])
        counter = [0]
        clause, params = store._filters_to_where_clause(filters, counter)
        assert "`price` >" in clause
        assert "JSON_EXTRACT(`meta`" in clause
        assert "AND" in clause
        assert "param_0" in params
        assert "param_1" in params
        assert params["param_0"] == 10
        assert params["param_1"] == "books"
