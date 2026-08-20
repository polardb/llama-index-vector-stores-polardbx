"""Unit tests for custom column SELECT and result mapping."""

import json
from unittest.mock import patch

import pytest

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


class TestBuildSelectColumns:
    """Tests for _build_select_columns()."""

    def test_default_columns(self):
        """Default schema returns static column list."""
        store = _make_store()
        cols = store._build_select_columns()
        assert cols == "node_id, text, metadata"

    def test_custom_column_names(self):
        """Custom column names produce aliased SELECT."""
        store = _make_store(
            node_id_column="nid",
            text_column="content",
            metadata_json_column="meta",
        )
        cols = store._build_select_columns()
        assert "`nid` AS `node_id`" in cols
        assert "`content` AS `text`" in cols
        assert "`meta` AS `metadata`" in cols

    def test_no_json_column(self):
        """When metadata_json_column is None, no metadata alias."""
        store = _make_store(
            metadata_json_column=None,
            metadata_columns=["category"],
        )
        cols = store._build_select_columns()
        assert "AS `metadata`" not in cols
        assert "`category`" in cols

    def test_metadata_columns_appended(self):
        """Metadata columns are appended after core columns."""
        store = _make_store(
            metadata_columns=["price", "category"]
        )
        cols = store._build_select_columns()
        parts = [p.strip() for p in cols.split(",")]
        # First 3 are core (node_id, text, metadata), then metadata cols
        assert len(parts) == 5
        assert "`price`" in cols
        assert "`category`" in cols


class TestBuildSearchSql:
    """Tests for _build_search_sql()."""

    def test_default_search_sql(self):
        """Default schema search SQL contains all expected parts."""
        store = _make_store()
        sql = store._build_search_sql(
            distance_func="VEC_DISTANCE",
            index_hint="",
            where_clause="",
        )
        assert "SELECT" in sql
        assert "node_id, text, metadata" in sql
        assert "VEC_DISTANCE" in sql
        assert "VEC_FROMTEXT(:query_embedding)" in sql
        assert "ORDER BY distance" in sql
        assert "LIMIT :limit" in sql

    def test_custom_search_sql(self):
        """Custom columns in search SQL."""
        store = _make_store(
            embedding_column="emb",
            metadata_columns=[Column("price", "DECIMAL(10,2)")],
        )
        sql = store._build_search_sql(
            distance_func="VEC_DISTANCE_COSINE",
            index_hint=" FORCE INDEX (`vi`)",
            where_clause="WHERE price > 10",
        )
        assert "`emb`" in sql
        assert "`price`" in sql
        assert "FORCE INDEX" in sql
        assert "WHERE price > 10" in sql

    def test_custom_embedding_column_in_search(self):
        """Custom embedding column name in distance function."""
        store = _make_store(embedding_column="emb_vec")
        sql = store._build_search_sql(
            distance_func="VEC_DISTANCE",
            index_hint="",
            where_clause="",
        )
        assert "`emb_vec`" in sql


class TestRecordToMetadata:
    """Tests for _record_to_metadata()."""

    def test_default_schema(self):
        """Default schema deserializes JSON from index 2."""
        store = _make_store()
        row = ("node-1", "hello", '{"foo": "bar"}', 0.5)
        meta = store._record_to_metadata(row)
        assert meta == {"foo": "bar"}

    def test_custom_columns_merge(self):
        """Custom columns merge JSON + mapped column values."""
        store = _make_store(
            metadata_columns=["price", "category"]
        )
        # Row: node_id, text, metadata_json, price, category, distance
        row = ("node-1", "hello", '{"tags": ["new"]}', 9.99, "books", 0.3)
        meta = store._record_to_metadata(row)
        assert meta["tags"] == ["new"]
        assert meta["price"] == 9.99
        assert meta["category"] == "books"

    def test_custom_columns_no_json(self):
        """When metadata_json_column is None, only mapped cols are used."""
        store = _make_store(
            metadata_json_column=None,
            metadata_columns=["category"],
        )
        # Row: node_id, text, (no json), category, distance
        row = ("node-1", "hello", "books", 0.3)
        meta = store._record_to_metadata(row)
        assert meta == {"category": "books"}

    def test_custom_columns_null_mapped_value(self):
        """Null mapped column values are not added to metadata."""
        store = _make_store(
            metadata_columns=["price", "category"]
        )
        # price is None, category has a value
        row = ("node-1", "hello", '{"foo": "bar"}', None, "books", 0.3)
        meta = store._record_to_metadata(row)
        assert meta["foo"] == "bar"
        assert meta["category"] == "books"
        assert "price" not in meta

    def test_custom_column_names(self):
        """Custom column names work in result mapping."""
        store = _make_store(
            node_id_column="nid",
            text_column="content",
            metadata_json_column="meta",
            metadata_columns=[Column("price", "DECIMAL(10,2)")],
        )
        # With aliases: nid AS node_id, content AS text, meta AS metadata
        # Row: node_id(aliased), text(aliased), metadata(aliased), price, distance
        row = ("node-1", "hello", '{"x": 1}', 19.99, 0.2)
        meta = store._record_to_metadata(row)
        assert meta["x"] == 1
        assert meta["price"] == 19.99
