"""Unit tests for custom column UPSERT SQL and params generation."""

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


class TestBuildUpsertSql:
    """Tests for _build_upsert_sql()."""

    def test_default_non_partitioned(self):
        """Default schema uses static SQL with ON DUPLICATE KEY UPDATE."""
        store = _make_store()
        sql = store._build_upsert_sql(partitioned=False)
        assert "INSERT INTO" in sql
        assert "ON DUPLICATE KEY UPDATE" in sql
        assert "VEC_FROMTEXT(:embedding)" in sql

    def test_default_partitioned(self):
        """Default schema partitioned uses plain INSERT."""
        store = _make_store()
        sql = store._build_upsert_sql(partitioned=True)
        assert "INSERT INTO" in sql
        assert "ON DUPLICATE KEY UPDATE" not in sql

    def test_custom_columns_non_partitioned(self):
        """Custom columns produce dynamic SQL with all columns."""
        store = _make_store(
            metadata_columns=[
                Column("price", "DECIMAL(10,2)"),
                "category",
            ]
        )
        sql = store._build_upsert_sql(partitioned=False)
        assert "`price`" in sql
        assert "`category`" in sql
        assert ":meta_price" in sql
        assert ":meta_category" in sql
        assert "ON DUPLICATE KEY UPDATE" in sql
        # The update clause should include custom columns
        assert "`price` = VALUES(`price`)" in sql
        assert "`category` = VALUES(`category`)" in sql

    def test_custom_columns_partitioned(self):
        """Custom columns with partition uses plain INSERT."""
        store = _make_store(
            metadata_columns=[Column("price", "DECIMAL(10,2)")],
            partition_by="HASH",
            partitions=4,
        )
        sql = store._build_upsert_sql(partitioned=True)
        assert "`price`" in sql
        assert ":meta_price" in sql
        assert "ON DUPLICATE KEY UPDATE" not in sql

    def test_custom_column_names_in_sql(self):
        """Custom column names appear in INSERT."""
        store = _make_store(
            id_column="uid",
            node_id_column="nid",
            text_column="content",
            embedding_column="emb",
            metadata_json_column="meta",
            metadata_columns=[Column("price", "DECIMAL(10,2)")],
        )
        sql = store._build_upsert_sql(partitioned=False)
        assert "`uid`" in sql
        assert "`nid`" in sql
        assert "`content`" in sql
        assert "`emb`" in sql
        assert "`meta`" in sql
        assert "`price`" in sql

    def test_no_json_column_in_upsert(self):
        """When metadata_json_column is None, no :metadata param."""
        store = _make_store(
            metadata_json_column=None,
            metadata_columns=["category"],
        )
        sql = store._build_upsert_sql(partitioned=False)
        # No `metadata` column in the INSERT
        assert ":metadata" not in sql
        assert "`metadata`" not in sql
        assert ":meta_category" in sql


class TestBuildUpsertParams:
    """Tests for _build_upsert_params()."""

    def _make_item(self, metadata=None):
        """Create a fake item dict as _node_to_table_row would produce."""
        return {
            "node_id": "node-123",
            "text": "hello world",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "metadata": metadata or {},
        }

    def test_default_params(self):
        """Default schema produces standard params."""
        store = _make_store()
        params = store._build_upsert_params(self._make_item({"foo": "bar"}))
        assert params["node_id"] == "node-123"
        assert params["text"] == "hello world"
        assert json.loads(params["embedding"]) == [0.1, 0.2, 0.3, 0.4]
        assert json.loads(params["metadata"]) == {"foo": "bar"}

    def test_custom_column_params_extracted(self):
        """Mapped keys go to meta_ params, rest go to JSON."""
        store = _make_store(
            metadata_columns=[Column("price", "DECIMAL(10,2)")]
        )
        params = store._build_upsert_params(
            self._make_item({"price": 9.99, "category": "books"})
        )
        assert params["meta_price"] == 9.99
        # 'category' should remain in JSON
        remaining = json.loads(params["metadata"])
        assert remaining == {"category": "books"}

    def test_custom_column_not_null_missing_raises(self):
        """Missing value for NOT NULL column raises ValueError."""
        store = _make_store(
            metadata_columns=[
                Column("price", "DECIMAL(10,2)", nullable=False)
            ]
        )
        with pytest.raises(ValueError, match="NOT NULL"):
            store._build_upsert_params(self._make_item({"category": "books"}))

    def test_custom_column_nullable_missing_ok(self):
        """Missing value for nullable column is OK (None)."""
        store = _make_store(
            metadata_columns=[Column("price", "DECIMAL(10,2)", nullable=True)]
        )
        params = store._build_upsert_params(self._make_item({"category": "books"}))
        assert params["meta_price"] is None

    def test_no_json_column_params(self):
        """When metadata_json_column is None, no 'metadata' key in params."""
        store = _make_store(
            metadata_json_column=None,
            metadata_columns=["category"],
        )
        params = store._build_upsert_params(
            self._make_item({"category": "books", "extra": "data"})
        )
        assert "metadata" not in params
        assert params["meta_category"] == "books"

    def test_multiple_custom_columns(self):
        """Multiple metadata columns are all extracted."""
        store = _make_store(
            metadata_columns=[
                Column("price", "DECIMAL(10,2)"),
                Column("category", "VARCHAR(100)"),
            ]
        )
        params = store._build_upsert_params(
            self._make_item({
                "price": 19.99,
                "category": "electronics",
                "tags": ["new", "sale"],
            })
        )
        assert params["meta_price"] == 19.99
        assert params["meta_category"] == "electronics"
        # 'tags' should be in JSON
        remaining = json.loads(params["metadata"])
        assert remaining == {"tags": ["new", "sale"]}

    def test_string_column_missing_value_ok(self):
        """String-only column with no Column object: missing value is None."""
        store = _make_store(
            metadata_columns=["category"]
        )
        params = store._build_upsert_params(self._make_item({"price": 10}))
        assert params["meta_category"] is None
