"""Unit tests for custom column DDL generation."""

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


class TestBuildCreateTableSqlCustom:
    """Tests for _build_create_table_sql_custom()."""

    def test_default_ddl_matches_original(self):
        """When using defaults but custom mode triggered, DDL is correct."""
        store = _make_store(
            id_column="id",
            metadata_columns=["category"],
        )
        store._capabilities = {"vec_dim": False}
        ddl = store._build_create_table_sql_custom()
        assert "CREATE TABLE IF NOT EXISTS" in ddl
        assert "`id` VARCHAR(36) PRIMARY KEY" in ddl
        assert "`node_id` VARCHAR(255) NOT NULL" in ddl
        assert "`text` LONGTEXT" in ddl
        assert "`metadata` JSON" in ddl
        assert "VECTOR(4) NOT NULL" in ddl
        assert "UNIQUE INDEX" in ddl
        assert "`category` TEXT" in ddl

    def test_custom_column_names(self):
        """Custom column names appear in DDL."""
        store = _make_store(
            id_column="uid",
            node_id_column="nid",
            text_column="content",
            embedding_column="emb",
            metadata_json_column="meta",
        )
        store._capabilities = {}
        ddl = store._build_create_table_sql_custom()
        assert "`uid` VARCHAR(36) PRIMARY KEY" in ddl
        assert "`nid` VARCHAR(255) NOT NULL" in ddl
        assert "`content` LONGTEXT" in ddl
        assert "`meta` JSON" in ddl
        assert "`emb` VECTOR(4) NOT NULL" in ddl
        assert "UNIQUE INDEX `node_id_index` (nid)" in ddl
        assert "VECTOR INDEX `vi` (emb)" in ddl

    def test_no_json_column(self):
        """When metadata_json_column is None, no JSON column in DDL."""
        store = _make_store(
            metadata_json_column=None,
            metadata_columns=["category"],
        )
        store._capabilities = {}
        ddl = store._build_create_table_sql_custom()
        assert "JSON" not in ddl
        assert "`category` TEXT" in ddl

    def test_column_object_with_data_type(self):
        """Column objects produce proper DDL with data_type."""
        store = _make_store(
            metadata_columns=[
                Column("price", "DECIMAL(10,2)", nullable=False),
                Column("category", "VARCHAR(100)", default="'misc'"),
            ]
        )
        store._capabilities = {}
        ddl = store._build_create_table_sql_custom()
        assert "`price` DECIMAL(10,2) NOT NULL" in ddl
        assert "`category` VARCHAR(100) DEFAULT 'misc'" in ddl

    def test_string_column_gets_text_default(self):
        """String-only column names get TEXT type."""
        store = _make_store(
            metadata_columns=["tags"]
        )
        store._capabilities = {}
        ddl = store._build_create_table_sql_custom()
        assert "`tags` TEXT" in ddl

    def test_partition_downgrades_node_index(self):
        """Partitioned tables use INDEX not UNIQUE INDEX."""
        store = _make_store(
            partition_by="HASH",
            partitions=4,
            metadata_columns=["category"],
        )
        store._capabilities = {}
        ddl = store._build_create_table_sql_custom()
        assert "INDEX `node_id_index`" in ddl
        assert "UNIQUE INDEX" not in ddl

    def test_ef_construction_clause(self):
        """EF_CONSTRUCTION is included when capabilities support it."""
        store = _make_store(
            ef_construction=200,
            metadata_columns=["category"],
        )
        store._capabilities = {"vec_dim": True}
        ddl = store._build_create_table_sql_custom()
        assert "EF_CONSTRUCTION=200" in ddl

    def test_mixed_column_and_string(self):
        """Mixed Column objects and strings are both handled."""
        store = _make_store(
            metadata_columns=[
                Column("price", "DECIMAL(10,2)"),
                "category",
            ]
        )
        store._capabilities = {}
        ddl = store._build_create_table_sql_custom()
        assert "`price` DECIMAL(10,2)" in ddl
        assert "`category` TEXT" in ddl

    def test_table_name_in_ddl(self):
        """The table name is correctly embedded in DDL."""
        store = _make_store(
            table_name="my_vectors",
            metadata_columns=["category"],
        )
        store._capabilities = {}
        ddl = store._build_create_table_sql_custom()
        assert "`my_vectors`" in ddl

    def test_engine_and_charset(self):
        """DDL includes ENGINE and CHARSET."""
        store = _make_store(metadata_columns=["category"])
        store._capabilities = {}
        ddl = store._build_create_table_sql_custom()
        assert "ENGINE=InnoDB" in ddl
        assert "DEFAULT CHARSET=utf8mb4" in ddl
        assert "COLLATE=utf8mb4_unicode_ci" in ddl
