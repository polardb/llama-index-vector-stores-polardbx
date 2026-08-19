"""Integration tests for the SQL module against a real PolarDB-X instance.

These tests verify that PolarDBXSQLDatabase correctly:
1. Connects via from_uri with auto-swapped dialect
2. Reflects DDL with tab indentation fix (get_usable_table_names, get_table_columns, get_single_table_info)
3. Executes SQL (run_sql for DDL/DML/SELECT)
4. Inserts data via ORM (insert_into_table)
5. Reflects tables with VECTOR columns without crashing
6. Utility methods (dialect, engine, metadata_obj, truncate_word)

Requires a real PolarDB-X instance. Set POLARDBX_URI env var or edit the URI below.
"""

import os
import sys

import pytest
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import uri

URI = uri()

# Use a unique test table name to avoid collisions
TEST_TABLE = "sql_it_test_tbl"
VECTOR_TABLE = "sql_it_vector_tbl"


@pytest.fixture(scope="module")
def db():
    """Create a PolarDBXSQLDatabase connected to the real instance."""
    from llama_index.vector_stores.polardbx import PolarDBXSQLDatabase

    instance = PolarDBXSQLDatabase.from_uri(URI)
    yield instance


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """Create test tables before tests, drop them after."""
    engine = create_engine(URI)
    with engine.connect() as conn:
        # Drop existing tables
        conn.execute(text(f"DROP TABLE IF EXISTS `{TEST_TABLE}`"))
        conn.execute(text(f"DROP TABLE IF EXISTS `{VECTOR_TABLE}`"))
        conn.commit()

        # Create a standard table (PolarDB-X will output tab-indented DDL)
        conn.execute(text(f"""
            CREATE TABLE `{TEST_TABLE}` (
                `id` INT NOT NULL AUTO_INCREMENT,
                `name` VARCHAR(100) NOT NULL,
                `status` ENUM('active', 'inactive', 'pending') DEFAULT 'pending',
                `price` DECIMAL(10,2) DEFAULT 0.00,
                `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))
        conn.commit()

        # Create a table with VECTOR column (to test VECTOR type reflection)
        conn.execute(text(f"""
            CREATE TABLE `{VECTOR_TABLE}` (
                `id` VARCHAR(36) PRIMARY KEY,
                `text` LONGTEXT,
                `metadata` JSON,
                `embedding` VECTOR(4) NOT NULL,
                VECTOR INDEX `vi` (`embedding`) M=6 DISTANCE=COSINE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))
        conn.commit()

    yield

    # Cleanup
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{TEST_TABLE}`"))
        conn.execute(text(f"DROP TABLE IF EXISTS `{VECTOR_TABLE}`"))
        conn.commit()
    engine.dispose()


# ---------------------------------------------------------------------------
# 1. Connection and basic properties
# ---------------------------------------------------------------------------

class TestConnection:
    """Verify from_uri connection and basic properties."""

    def test_from_uri_creates_instance(self, db):
        """from_uri returns a PolarDBXSQLDatabase instance."""
        from llama_index.vector_stores.polardbx import PolarDBXSQLDatabase

        assert isinstance(db, PolarDBXSQLDatabase)

    def test_dialect_property(self, db):
        """dialect returns 'mysql' (PolarDBXDialect.name == 'mysql')."""
        assert db.dialect == "mysql"

    def test_engine_property(self, db):
        """engine returns a SQLAlchemy Engine."""
        from sqlalchemy.engine import Engine

        assert isinstance(db.engine, Engine)

    def test_engine_uses_polardbx_dialect(self, db):
        """The underlying engine uses PolarDBXDialect, not plain MySQLDialect."""
        from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

        assert isinstance(db.engine.dialect, PolarDBXDialect)

    def test_metadata_obj_property(self, db):
        """metadata_obj returns a MetaData with reflected tables."""
        from sqlalchemy import MetaData

        assert isinstance(db.metadata_obj, MetaData)


# ---------------------------------------------------------------------------
# 2. DDL reflection (the core fix)
# ---------------------------------------------------------------------------

class TestDDLReflection:
    """Verify PolarDB-X DDL reflection fixes work."""

    def test_get_usable_table_names(self, db):
        """get_usable_table_names includes our test tables."""
        tables = list(db.get_usable_table_names())
        assert TEST_TABLE in tables
        assert VECTOR_TABLE in tables

    def test_get_table_columns(self, db):
        """get_table_columns returns all columns for the test table."""
        cols = db.get_table_columns(TEST_TABLE)
        col_names = [c["name"] for c in cols]
        assert "id" in col_names
        assert "name" in col_names
        assert "status" in col_names
        assert "price" in col_names
        assert "created_at" in col_names
        assert len(col_names) == 5

    def test_get_table_columns_has_correct_types(self, db):
        """get_table_columns returns correct column types."""
        cols = {c["name"]: c for c in db.get_table_columns(TEST_TABLE)}
        # id should be integer
        assert "int" in str(cols["id"]["type"]).lower()
        # name should be varchar
        assert "varchar" in str(cols["name"]["type"]).lower()

    def test_get_single_table_info(self, db):
        """get_single_table_info returns DDL with column info."""
        info = db.get_single_table_info(TEST_TABLE)
        assert TEST_TABLE in info
        assert "id" in info
        assert "name" in info
        assert "status" in info

    def test_reflect_vector_table_columns(self, db):
        """get_table_columns works on tables with VECTOR columns."""
        cols = db.get_table_columns(VECTOR_TABLE)
        col_names = [c["name"] for c in cols]
        assert "id" in col_names
        assert "text" in col_names
        assert "embedding" in col_names
        assert len(col_names) == 4

    def test_get_single_table_info_vector_table(self, db):
        """get_single_table_info works on tables with VECTOR columns."""
        info = db.get_single_table_info(VECTOR_TABLE)
        assert VECTOR_TABLE in info
        assert "embedding" in info

    def test_metadata_tables_populated(self, db):
        """metadata_obj.tables contains our test tables."""
        assert TEST_TABLE in db.metadata_obj.tables
        assert VECTOR_TABLE in db.metadata_obj.tables

    def test_metadata_table_has_columns(self, db):
        """Reflected Table object has correct column count."""
        table = db.metadata_obj.tables[TEST_TABLE]
        assert len(table.columns) == 5
        assert "id" in table.columns
        assert "name" in table.columns


# ---------------------------------------------------------------------------
# 3. SQL execution
# ---------------------------------------------------------------------------

class TestSQLExecution:
    """Verify run_sql executes DDL/DML/SELECT correctly."""

    def test_run_sql_select(self, db):
        """run_sql executes SELECT and returns results."""
        result_str, meta = db.run_sql("SELECT 1 AS val")
        assert "1" in result_str
        assert meta["col_keys"] == ["val"]

    def test_run_sql_insert(self, db):
        """run_sql executes INSERT."""
        db.run_sql(
            f"INSERT INTO `{TEST_TABLE}` (name, status, price) "
            "VALUES ('test_item', 'active', 9.99)"
        )
        result_str, _ = db.run_sql(
            f"SELECT COUNT(*) FROM `{TEST_TABLE}` WHERE name = 'test_item'"
        )
        assert "1" in result_str

    def test_run_sql_update(self, db):
        """run_sql executes UPDATE."""
        db.run_sql(
            f"UPDATE `{TEST_TABLE}` SET status = 'inactive' "
            "WHERE name = 'test_item'"
        )
        result_str, _ = db.run_sql(
            f"SELECT status FROM `{TEST_TABLE}` WHERE name = 'test_item'"
        )
        assert "inactive" in result_str

    def test_run_sql_delete(self, db):
        """run_sql executes DELETE."""
        db.run_sql(f"DELETE FROM `{TEST_TABLE}` WHERE name = 'test_item'")
        result_str, _ = db.run_sql(
            f"SELECT COUNT(*) FROM `{TEST_TABLE}` WHERE name = 'test_item'"
        )
        assert "0" in result_str

    def test_run_sql_invalid_raises(self, db):
        """run_sql raises on invalid SQL."""
        with pytest.raises(NotImplementedError):
            db.run_sql("SELECT FROM nonexistent_table_xyz")


# ---------------------------------------------------------------------------
# 4. ORM insert
# ---------------------------------------------------------------------------

class TestInsertIntoTable:
    """Verify insert_into_table uses reflected metadata."""

    def test_insert_into_table(self, db):
        """insert_into_table inserts a row via SQLAlchemy ORM."""
        db.insert_into_table(
            TEST_TABLE,
            {"name": "orm_item", "status": "pending", "price": 19.99},
        )
        result_str, _ = db.run_sql(
            f"SELECT name, status, price FROM `{TEST_TABLE}` "
            "WHERE name = 'orm_item'"
        )
        assert "orm_item" in result_str
        assert "pending" in result_str
        assert "19.99" in result_str

    def test_insert_and_count(self, db):
        """Insert multiple rows and verify count."""
        for i in range(5):
            db.insert_into_table(
                TEST_TABLE,
                {"name": f"bulk_{i}", "status": "active", "price": float(i)},
            )
        result_str, _ = db.run_sql(
            f"SELECT COUNT(*) FROM `{TEST_TABLE}` WHERE name LIKE 'bulk_%'"
        )
        assert "5" in result_str


# ---------------------------------------------------------------------------
# 5. Utility methods
# ---------------------------------------------------------------------------

class TestUtilityMethods:
    """Verify utility methods."""

    def test_truncate_word_short(self, db):
        """truncate_word returns short strings unchanged."""
        assert db.truncate_word("hello", length=10) == "hello"

    def test_truncate_word_long(self, db):
        """truncate_word truncates long strings."""
        result = db.truncate_word(
            "this is a very long string that exceeds the limit",
            length=20,
        )
        assert len(result) <= 20
        assert result.endswith("...")

    def test_truncate_word_zero_length(self, db):
        """truncate_word with length=0 returns original."""
        result = db.truncate_word("hello", length=0)
        assert result == "hello"

    def test_truncate_word_non_string(self, db):
        """truncate_word with non-string returns original."""
        result = db.truncate_word(12345, length=3)
        assert result == 12345
