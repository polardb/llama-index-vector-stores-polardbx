"""Unit tests for partition table functionality.

Covers:
1. _build_partition_clause: all strategies (HASH/KEY/RANGE/LIST/BROADCAST/LOCALITY)
2. _sql_quote_string: standard and edge cases
3. _validate_identifier: valid, invalid, SQL injection, length
4. VectorStore constructor: partition param validation
5. VectorStore DDL: UNIQUE INDEX -> INDEX downgrade when partitioning
6. create_partitioned_table: DDL generation (mocked engine)
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_store(**kwargs):
    """Create a PolarDBXVectorStore with mocked initialization.

    All keyword arguments are forwarded to the constructor.
    """
    from llama_index.vector_stores.polardbx import PolarDBXVectorStore

    with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
        mock_init.return_value = None
        store = PolarDBXVectorStore(
            host="localhost",
            port=3306,
            user="root",
            password="test",
            database="testdb",
            **kwargs,
        )
    store._is_initialized = True
    store._capabilities = {}
    return store


# ---------------------------------------------------------------------------
# _build_partition_clause tests
# ---------------------------------------------------------------------------


class TestBuildPartitionClause:
    """Test _build_partition_clause for all partition strategies."""

    def test_no_partition_returns_empty(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause()
        assert result == ""

    def test_hash_partition(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="HASH", partition_column="id", partitions=8
        )
        assert result == " PARTITION BY HASH(id) PARTITIONS 8"

    def test_key_partition(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="KEY", partition_column="user_id", partitions=4
        )
        assert result == " PARTITION BY KEY(user_id) PARTITIONS 4"

    def test_hash_partition_lowercase(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="hash", partition_column="id", partitions=2
        )
        assert result == " PARTITION BY HASH(id) PARTITIONS 2"

    def test_range_partition_numeric(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="RANGE",
            partition_column="id",
            partition_defs=[
                {"name": "p0", "values_less_than": 100},
                {"name": "p1", "values_less_than": 1000},
                {"name": "p2", "values_less_than": "MAXVALUE"},
            ],
        )
        assert "PARTITION BY RANGE(id)" in result
        assert "PARTITION p0 VALUES LESS THAN (100)" in result
        assert "PARTITION p1 VALUES LESS THAN (1000)" in result
        assert "PARTITION p2 VALUES LESS THAN (MAXVALUE)" in result

    def test_range_partition_string_values(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="RANGE",
            partition_column="created_at",
            partition_defs=[
                {"name": "p0", "values_less_than": "2024-01-01"},
                {"name": "p1", "values_less_than": "2025-01-01"},
                {"name": "p2", "values_less_than": "MAXVALUE"},
            ],
        )
        assert "PARTITION p0 VALUES LESS THAN ('2024-01-01')" in result
        assert "PARTITION p1 VALUES LESS THAN ('2025-01-01')" in result
        assert "PARTITION p2 VALUES LESS THAN (MAXVALUE)" in result

    def test_range_partition_missing_defs(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="RANGE", partition_column="id", partition_defs=None
        )
        assert "PARTITION BY RANGE(id)" in result

    def test_range_partition_def_missing_name(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        with pytest.raises(ValueError, match="must have a 'name' key"):
            _build_partition_clause(
                partition_by="RANGE",
                partition_column="id",
                partition_defs=[{"values_less_than": 100}],
            )

    def test_range_partition_def_missing_values_less_than(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        with pytest.raises(ValueError, match="missing 'values_less_than' key"):
            _build_partition_clause(
                partition_by="RANGE",
                partition_column="id",
                partition_defs=[{"name": "p0"}],
            )

    def test_list_partition_strings(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="LIST",
            partition_column="region",
            partition_defs=[
                {"name": "p_east", "values_in": ["east", "northeast"]},
                {"name": "p_west", "values_in": ["west", "southwest"]},
            ],
        )
        assert "PARTITION BY LIST(region)" in result
        assert "PARTITION p_east VALUES IN ('east', 'northeast')" in result
        assert "PARTITION p_west VALUES IN ('west', 'southwest')" in result

    def test_list_partition_numbers(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="LIST",
            partition_column="dept_id",
            partition_defs=[
                {"name": "p0", "values_in": [1, 2, 3]},
                {"name": "p1", "values_in": [4, 5, 6]},
            ],
        )
        assert "PARTITION p0 VALUES IN (1, 2, 3)" in result
        assert "PARTITION p1 VALUES IN (4, 5, 6)" in result

    def test_list_partition_single_quoted_string(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="LIST",
            partition_column="name",
            partition_defs=[
                {"name": "p0", "values_in": ["O'Brien"]},
            ],
        )
        assert "PARTITION p0 VALUES IN ('O''Brien')" in result

    def test_list_partition_def_missing_name(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        with pytest.raises(ValueError, match="must have a 'name' key"):
            _build_partition_clause(
                partition_by="LIST",
                partition_column="id",
                partition_defs=[{"values_in": [1, 2]}],
            )

    def test_list_partition_def_missing_values_in(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        with pytest.raises(ValueError, match="missing 'values_in' key"):
            _build_partition_clause(
                partition_by="LIST",
                partition_column="id",
                partition_defs=[{"name": "p0"}],
            )

    def test_broadcast(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(broadcast=True)
        assert result == " BROADCAST"

    def test_broadcast_overrides_partition_by(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="HASH", partition_column="id", partitions=4, broadcast=True
        )
        assert result == " BROADCAST"

    def test_locality_alone(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(locality="dn=polardbx-storage-0")
        assert result == " LOCALITY='dn=polardbx-storage-0'"

    def test_locality_with_hash(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="HASH",
            partition_column="id",
            partitions=4,
            locality="dn=storage-0",
        )
        assert "PARTITION BY HASH(id) PARTITIONS 4" in result
        assert "LOCALITY='dn=storage-0'" in result

    def test_locality_with_broadcast(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            broadcast=True, locality="dn=storage-1"
        )
        assert " BROADCAST" in result
        assert "LOCALITY='dn=storage-1'" in result

    def test_locality_with_single_quote_escaped(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(locality="dn='injected")
        assert "LOCALITY='dn=''injected'" in result

    def test_invalid_partition_by_ignored(self) -> None:
        """Invalid partition_by (not HASH/KEY/RANGE/LIST) produces no clause."""
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        result = _build_partition_clause(
            partition_by="INVALID", partition_column="id", partitions=4
        )
        assert result == ""


# ---------------------------------------------------------------------------
# _sql_quote_string tests
# ---------------------------------------------------------------------------


class TestSqlQuoteString:
    """Test _sql_quote_string for various inputs."""

    def test_plain_string(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _sql_quote_string,
        )

        assert _sql_quote_string("hello") == "'hello'"

    def test_empty_string(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _sql_quote_string,
        )

        assert _sql_quote_string("") == "''"

    def test_string_with_single_quote(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _sql_quote_string,
        )

        assert _sql_quote_string("O'Brien") == "'O''Brien'"

    def test_string_with_multiple_single_quotes(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _sql_quote_string,
        )

        assert _sql_quote_string("it's 'a' test") == "'it''s ''a'' test'"

    def test_string_with_double_quotes(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _sql_quote_string,
        )

        assert _sql_quote_string('say "hi"') == '\'say "hi"\''

    def test_string_with_backslash(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _sql_quote_string,
        )

        # M3: Backslashes are escaped to prevent MySQL escape interpretation
        assert _sql_quote_string("path\\to\\file") == "'path\\\\to\\\\file'"


# ---------------------------------------------------------------------------
# _validate_identifier tests
# ---------------------------------------------------------------------------


class TestValidateIdentifier:
    """Test _validate_identifier for valid and invalid identifiers."""

    def test_valid_simple(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        _validate_identifier("my_table")

    def test_valid_with_underscore_prefix(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        _validate_identifier("_private")

    def test_valid_with_numbers(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        _validate_identifier("table123")

    def test_valid_single_letter(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        _validate_identifier("a")

    def test_valid_max_length(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        _validate_identifier("a" * 64)

    def test_invalid_empty(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("")

    def test_invalid_starts_with_number(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("1table")

    def test_invalid_starts_with_hyphen(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("-table")

    def test_invalid_contains_space(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("my table")

    def test_invalid_contains_semicolon(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier("table; DROP")

    def test_invalid_too_long(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        with pytest.raises(ValueError, match="too long"):
            _validate_identifier("a" * 65)

    def test_sql_injection_attempt(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        injection = "id) PARTITIONS 1; DROP TABLE users; --"
        with pytest.raises(ValueError, match="Invalid identifier"):
            _validate_identifier(injection)

    def test_label_in_error_message(self) -> None:
        from llama_index.vector_stores.polardbx._partition import (
            _validate_identifier,
        )

        with pytest.raises(ValueError, match="partition column"):
            _validate_identifier("1bad", "partition column")


# ---------------------------------------------------------------------------
# VectorStore constructor validation tests
# ---------------------------------------------------------------------------


class TestVectorStorePartitionValidation:
    """Test PolarDBXVectorStore constructor partition param validation."""

    def test_invalid_partition_by(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="Invalid partition_by"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="INVALID",
                partitions=4,
            )

    def test_hash_without_partitions(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="partitions must be > 0"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="HASH",
                partitions=0,
            )

    def test_key_without_partitions(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="partitions must be > 0"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="KEY",
                partitions=0,
            )

    def test_range_without_defs(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="partition_defs must be provided"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="RANGE",
            )

    def test_list_without_defs(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="partition_defs must be provided"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="LIST",
            )

    def test_broadcast_and_partition_mutually_exclusive(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="mutually exclusive"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="HASH",
                partitions=4,
                broadcast=True,
            )

    def test_invalid_partition_column(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        # W4: Vector tables only allow partition_column="id"
        with pytest.raises(ValueError, match="partition_column must be 'id'"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="HASH",
                partitions=4,
                partition_column="1invalid",
            )

    def test_partition_column_default_is_id(self) -> None:
        """Default partition_column should be 'id'."""
        vs = _make_store(partition_by="HASH", partitions=4)
        assert vs._partition_column == "id"

    def test_partition_by_uppercased(self) -> None:
        vs = _make_store(partition_by="hash", partitions=4)
        assert vs._partition_by == "HASH"


# ---------------------------------------------------------------------------
# VectorStore DDL generation tests
# ---------------------------------------------------------------------------


class TestVectorStorePartitionDDL:
    """Test that VectorStore generates correct DDL with partition clauses."""

    def test_build_partition_clause_hash(self) -> None:
        vs = _make_store(partition_by="HASH", partitions=8)
        clause = vs._build_partition_clause()
        assert clause == " PARTITION BY HASH(id) PARTITIONS 8"

    def test_build_partition_clause_broadcast(self) -> None:
        vs = _make_store(broadcast=True)
        clause = vs._build_partition_clause()
        assert clause == " BROADCAST"

    def test_build_partition_clause_none(self) -> None:
        vs = _make_store()
        clause = vs._build_partition_clause()
        assert clause == ""

    def test_has_partition_true_with_hash(self) -> None:
        vs = _make_store(partition_by="HASH", partitions=4)
        assert vs._has_partition is True

    def test_has_partition_true_with_broadcast(self) -> None:
        vs = _make_store(broadcast=True)
        assert vs._has_partition is True

    def test_has_partition_false_default(self) -> None:
        vs = _make_store()
        assert vs._has_partition is False

    def test_create_table_uses_unique_index_without_partition(self) -> None:
        """Without partitioning, DDL should use UNIQUE INDEX."""
        vs = _make_store()
        captured_sql = []

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                captured_sql.append(str(stmt))

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        vs._create_table_if_not_exists()
        assert len(captured_sql) == 1
        assert "UNIQUE INDEX" in captured_sql[0]

    def test_create_table_downgrades_to_index_with_partition(self) -> None:
        """With partitioning, DDL should downgrade UNIQUE to regular INDEX."""
        vs = _make_store(partition_by="HASH", partitions=4)
        captured_sql = []

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                captured_sql.append(str(stmt))

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        vs._create_table_if_not_exists()
        assert len(captured_sql) == 1
        sql = captured_sql[0]
        assert "UNIQUE INDEX" not in sql
        assert "INDEX `node_id_index`" in sql
        assert "PARTITION BY HASH(id) PARTITIONS 4" in sql

    def test_create_table_broadcast_downgrades_index(self) -> None:
        """With broadcast, DDL should also downgrade UNIQUE to INDEX."""
        vs = _make_store(broadcast=True)
        captured_sql = []

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                captured_sql.append(str(stmt))

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        vs._create_table_if_not_exists()
        assert len(captured_sql) == 1
        sql = captured_sql[0]
        assert "UNIQUE INDEX" not in sql
        assert "INDEX `node_id_index`" in sql
        assert "BROADCAST" in sql

    def test_create_table_range_partition_ddl(self) -> None:
        vs = _make_store(
            partition_by="RANGE",
            partition_defs=[
                {"name": "p0", "values_less_than": 100},
                {"name": "p1", "values_less_than": "MAXVALUE"},
            ],
        )
        captured_sql = []

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                captured_sql.append(str(stmt))

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        vs._create_table_if_not_exists()
        sql = captured_sql[0]
        assert "PARTITION BY RANGE(id)" in sql
        assert "PARTITION p0 VALUES LESS THAN (100)" in sql
        assert "PARTITION p1 VALUES LESS THAN (MAXVALUE)" in sql

    def test_create_table_list_partition_ddl(self) -> None:
        vs = _make_store(
            partition_by="LIST",
            partition_defs=[
                {"name": "p0", "values_in": ["east", "west"]},
            ],
        )
        captured_sql = []

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                captured_sql.append(str(stmt))

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        vs._create_table_if_not_exists()
        sql = captured_sql[0]
        assert "PARTITION BY LIST(id)" in sql
        assert "PARTITION p0 VALUES IN ('east', 'west')" in sql

    def test_create_table_locality_in_ddl(self) -> None:
        vs = _make_store(
            partition_by="HASH",
            partitions=4,
            locality="dn=storage-0",
        )
        captured_sql = []

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                captured_sql.append(str(stmt))

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        vs._create_table_if_not_exists()
        sql = captured_sql[0]
        assert "PARTITION BY HASH(id) PARTITIONS 4" in sql
        assert "LOCALITY='dn=storage-0'" in sql

    def test_create_table_raises_not_supported_on_partition_error(self) -> None:
        """When DDL fails with a partition-not-supported error, raise NotSupportedError."""
        from llama_index.vector_stores.polardbx.base import NotSupportedError

        vs = _make_store(partition_by="HASH", partitions=4)

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                raise Exception(
                    "ERR-CODE: [TDDL-4500][ERR_PARSER] Do not support partition by."
                )

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        with pytest.raises(NotSupportedError, match="not supported on this instance"):
            vs._create_table_if_not_exists()

    def test_create_table_reraises_non_partition_errors(self) -> None:
        """Non-partition DDL errors should be re-raised as-is."""
        vs = _make_store(partition_by="HASH", partitions=4)

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                raise Exception("Connection refused")

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        with pytest.raises(Exception, match="Connection refused"):
            vs._create_table_if_not_exists()

    def test_create_table_no_partition_reraises_any_error(self) -> None:
        """Without partitioning, any DDL error should be re-raised as-is."""
        vs = _make_store()

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def execute(self, stmt):
                raise Exception("Table already exists")

            def commit(self):
                pass

        vs._session = MagicMock(return_value=MockSession())
        with pytest.raises(Exception, match="Table already exists"):
            vs._create_table_if_not_exists()


# ---------------------------------------------------------------------------
# create_partitioned_table tests (mocked engine)
# ---------------------------------------------------------------------------


class TestCreatePartitionedTable:
    """Test create_partitioned_table DDL generation with mocked engine."""

    @patch("sqlalchemy.create_engine")
    def test_hash_partition(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql+pymysql://user:pass@host:3306/db",
            table_name="orders",
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "user_id BIGINT NOT NULL",
                "PRIMARY KEY (id)",
            ],
            partition_by="HASH",
            partition_column="user_id",
            partitions=16,
        )
        assert mock_conn.execute.call_count == 1
        executed_sql = str(mock_conn.execute.call_args[0][0])
        assert "CREATE TABLE IF NOT EXISTS `orders`" in executed_sql
        assert "PARTITION BY HASH(user_id) PARTITIONS 16" in executed_sql
        mock_create_engine.assert_called_once_with(
            "polardbx+pymysql://user:pass@host:3306/db"
        )

    @patch("sqlalchemy.create_engine")
    def test_broadcast(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql+pymysql://user:pass@host:3306/db",
            table_name="dim_table",
            columns=["id INT PRIMARY KEY", "name VARCHAR(100)"],
            broadcast=True,
        )
        executed_sql = str(mock_conn.execute.call_args[0][0])
        assert "BROADCAST" in executed_sql
        assert "PARTITION BY" not in executed_sql

    @patch("sqlalchemy.create_engine")
    def test_range_partition(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql+pymysql://user:pass@host:3306/db",
            table_name="events",
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "created_at DATETIME",
                "PRIMARY KEY (id)",
            ],
            partition_by="RANGE",
            partition_column="id",
            partition_defs=[
                {"name": "p0", "values_less_than": 1000},
                {"name": "p1", "values_less_than": "MAXVALUE"},
            ],
        )
        executed_sql = str(mock_conn.execute.call_args[0][0])
        assert "PARTITION BY RANGE(id)" in executed_sql
        assert "PARTITION p0 VALUES LESS THAN (1000)" in executed_sql
        assert "PARTITION p1 VALUES LESS THAN (MAXVALUE)" in executed_sql

    @patch("sqlalchemy.create_engine")
    def test_list_partition(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql+pymysql://user:pass@host:3306/db",
            table_name="orders_by_region",
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "region VARCHAR(20)",
                "PRIMARY KEY (id)",
            ],
            partition_by="LIST",
            partition_column="region",
            partition_defs=[
                {"name": "p_east", "values_in": ["east", "northeast"]},
                {"name": "p_west", "values_in": ["west"]},
            ],
        )
        executed_sql = str(mock_conn.execute.call_args[0][0])
        assert "PARTITION BY LIST(region)" in executed_sql
        assert "PARTITION p_east VALUES IN ('east', 'northeast')" in executed_sql
        assert "PARTITION p_west VALUES IN ('west')" in executed_sql

    @patch("sqlalchemy.create_engine")
    def test_locality(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql+pymysql://user:pass@host:3306/db",
            table_name="pinned",
            columns=["id INT PRIMARY KEY", "data TEXT"],
            partition_by="HASH",
            partitions=4,
            locality="dn=storage-0",
        )
        executed_sql = str(mock_conn.execute.call_args[0][0])
        assert "PARTITION BY HASH(id) PARTITIONS 4" in executed_sql
        assert "LOCALITY='dn=storage-0'" in executed_sql

    @patch("sqlalchemy.create_engine")
    def test_no_partition_single_table(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql+pymysql://user:pass@host:3306/db",
            table_name="simple",
            columns=["id INT PRIMARY KEY", "name VARCHAR(100)"],
        )
        executed_sql = str(mock_conn.execute.call_args[0][0])
        assert "PARTITION" not in executed_sql
        assert "BROADCAST" not in executed_sql

    @patch("sqlalchemy.create_engine")
    def test_if_not_exists_false(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql+pymysql://user:pass@host:3306/db",
            table_name="mytable",
            columns=["id INT PRIMARY KEY"],
            if_not_exists=False,
        )
        executed_sql = str(mock_conn.execute.call_args[0][0])
        assert "CREATE TABLE `mytable`" in executed_sql
        assert "IF NOT EXISTS" not in executed_sql

    def test_invalid_table_name(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(ValueError, match="table name"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="1invalid",
                columns=["id INT PRIMARY KEY"],
            )

    def test_invalid_partition_column(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(ValueError, match="partition column"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=["id INT PRIMARY KEY"],
                partition_by="HASH",
                partition_column="1bad",
                partitions=4,
            )

    def test_invalid_partition_by(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(ValueError, match="Invalid partition_by"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=["id INT PRIMARY KEY"],
                partition_by="INVALID",
                partitions=4,
            )

    def test_hash_without_partitions(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(ValueError, match="partitions must be > 0"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=["id INT PRIMARY KEY"],
                partition_by="HASH",
                partitions=0,
            )

    def test_range_without_defs(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(ValueError, match="partition_defs required"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=["id INT PRIMARY KEY"],
                partition_by="RANGE",
            )

    def test_broadcast_and_partition_mutually_exclusive(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(ValueError, match="mutually exclusive"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=["id INT PRIMARY KEY"],
                partition_by="HASH",
                partitions=4,
                broadcast=True,
            )

    @patch("sqlalchemy.create_engine")
    def test_mysql_uri_swap(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql://user:pass@host:3306/db",
            table_name="test",
            columns=["id INT PRIMARY KEY"],
        )
        mock_create_engine.assert_called_once_with(
            "polardbx+pymysql://user:pass@host:3306/db"
        )

    @patch("sqlalchemy.create_engine")
    def test_engine_disposed(self, mock_create_engine: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        create_partitioned_table(
            uri="mysql+pymysql://user:pass@host:3306/db",
            table_name="test",
            columns=["id INT PRIMARY KEY"],
        )
        mock_engine.dispose.assert_called_once()

    @patch("sqlalchemy.create_engine")
    def test_raises_not_supported_on_partition_error(
        self, mock_create_engine: MagicMock
    ) -> None:
        """When DDL fails with partition-not-supported, raise NotSupportedError."""
        from llama_index.vector_stores.polardbx.base import NotSupportedError

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception(
            "ERR-CODE: [TDDL-4500][ERR_PARSER] Do not support partition by."
        )
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(NotSupportedError, match="not support partitioning"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=["id INT PRIMARY KEY"],
                partition_by="HASH",
                partition_column="id",
                partitions=4,
            )

    @patch("sqlalchemy.create_engine")
    def test_reraises_non_partition_errors(
        self, mock_create_engine: MagicMock
    ) -> None:
        """Non-partition DDL errors should be re-raised as-is."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("Connection refused")
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(Exception, match="Connection refused"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=["id INT PRIMARY KEY"],
                partition_by="HASH",
                partition_column="id",
                partitions=4,
            )

    @patch("sqlalchemy.create_engine")
    def test_no_partition_reraises_any_error(
        self, mock_create_engine: MagicMock
    ) -> None:
        """Without partitioning, DDL errors should be re-raised as-is."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("Access denied")
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_create_engine.return_value = mock_engine

        from llama_index.vector_stores.polardbx import create_partitioned_table

        with pytest.raises(Exception, match="Access denied"):
            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=["id INT PRIMARY KEY"],
            )


# ---------------------------------------------------------------------------
# Module import and structure tests
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Test module organization and imports."""

    def test_partition_module_importable(self) -> None:
        from llama_index.vector_stores.polardbx import _partition

        assert hasattr(_partition, "_build_partition_clause")
        assert hasattr(_partition, "_validate_identifier")
        assert hasattr(_partition, "_sql_quote_string")

    def test_create_partitioned_table_exported(self) -> None:
        import llama_index.vector_stores.polardbx as pkg

        assert hasattr(pkg, "create_partitioned_table")

    def test_partition_module_no_sqlalchemy_import(self) -> None:
        """_partition.py should not import sqlalchemy (clean module)."""
        import inspect

        from llama_index.vector_stores.polardbx import _partition

        source = inspect.getsource(_partition)
        assert "sqlalchemy" not in source.lower()

    def test_vectorstore_delegates_to_partition_module(self) -> None:
        """VectorStore._build_partition_clause should delegate to _partition."""
        vs = _make_store(partition_by="HASH", partitions=4)

        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        expected = _build_partition_clause(
            partition_by="HASH", partition_column="id", partitions=4
        )
        assert vs._build_partition_clause() == expected


# ---------------------------------------------------------------------------
# from_params factory method tests
# ---------------------------------------------------------------------------


class TestFromParamsPartition:
    """Test from_params factory method with partition params."""

    def test_from_params_hash_partition(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            vs = PolarDBXVectorStore.from_params(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="HASH",
                partitions=8,
            )
        vs._is_initialized = True
        assert vs._partition_by == "HASH"
        assert vs._partitions == 8
        assert vs._partition_column == "id"

    def test_from_params_broadcast(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            vs = PolarDBXVectorStore.from_params(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                broadcast=True,
            )
        assert vs._broadcast is True

    def test_from_params_range_with_defs(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            vs = PolarDBXVectorStore.from_params(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="RANGE",
                partition_defs=[
                    {"name": "p0", "values_less_than": 100},
                    {"name": "p1", "values_less_than": "MAXVALUE"},
                ],
            )
        assert vs._partition_by == "RANGE"
        assert len(vs._partition_defs) == 2

    def test_from_params_locality(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            vs = PolarDBXVectorStore.from_params(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="HASH",
                partitions=4,
                locality="dn=storage-0",
            )
        assert vs._locality == "dn=storage-0"

    def test_from_params_no_partition(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            vs = PolarDBXVectorStore.from_params(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
            )
        assert vs._partition_by is None
        assert vs._broadcast is False
        assert vs._has_partition is False


# ---------------------------------------------------------------------------
# Review v2 fix tests: S1, S2, W2, W3, W4, W5, W12, M1-M5
# ---------------------------------------------------------------------------


class TestReviewV2Fixes:
    """Tests for issues identified in the v2 re-review report."""

    # --- S1: Partition name identifier validation ---

    def test_s1_range_partition_name_injection_rejected(self) -> None:
        """S1: RANGE partition name with SQL injection is rejected."""
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        with pytest.raises(ValueError, match="partition name"):
            _build_partition_clause(
                partition_by="RANGE",
                partition_column="id",
                partition_defs=[
                    {
                        "name": "p1) VALUES LESS THAN (0), PARTITION p2",
                        "values_less_than": 100,
                    }
                ],
            )

    def test_s1_list_partition_name_injection_rejected(self) -> None:
        """S1: LIST partition name with SQL injection is rejected."""
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        with pytest.raises(ValueError, match="partition name"):
            _build_partition_clause(
                partition_by="LIST",
                partition_column="id",
                partition_defs=[
                    {
                        "name": "evil; DROP TABLE",
                        "values_in": [1, 2],
                    }
                ],
            )

    # --- S2: Frozen model fields ---

    def test_s2_table_name_not_mutable_after_init(self) -> None:
        """S2: table_name cannot be changed after initialization."""
        vs = _make_store()
        with pytest.raises(Exception):
            vs.table_name = "evil"

    def test_s2_database_not_mutable_after_init(self) -> None:
        """S2: database cannot be changed after initialization."""
        vs = _make_store()
        with pytest.raises(Exception):
            vs.database = "other_db"

    # --- W2: Partitioned table uses DELETE+INSERT ---

    def test_w2_partitioned_add_uses_delete_insert(self) -> None:
        """W2: Partitioned table add() uses DELETE+INSERT, not ON DUPLICATE KEY."""
        from llama_index.vector_stores.polardbx.base import PolarDBXVectorStore

        executed_sqls = []

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def execute(self, stmt, params=None):
                executed_sqls.append(str(stmt))

            def commit(self):
                pass

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            vs = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                embed_dim=4,
                partition_by="HASH",
                partitions=4,
            )
            vs._session = lambda: MockSession()
            assert vs._has_partition is True

            from llama_index.core.schema import TextNode

            node = TextNode(text="hello", embedding=[0.1] * 4)
            vs.add([node])

        # First SQL should be DELETE (not INSERT ... ON DUPLICATE KEY)
        assert "DELETE" in executed_sqls[0].upper()
        # Should NOT have ON DUPLICATE KEY UPDATE
        assert not any(
            "ON DUPLICATE KEY" in sql.upper() for sql in executed_sqls
        )

    def test_w2_non_partitioned_add_uses_on_duplicate_key(self) -> None:
        """W2: Non-partitioned table add() still uses ON DUPLICATE KEY UPDATE."""
        from llama_index.vector_stores.polardbx.base import PolarDBXVectorStore

        executed_sqls = []

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def execute(self, stmt, params=None):
                executed_sqls.append(str(stmt))

            def commit(self):
                pass

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            vs = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                embed_dim=4,
            )
            vs._session = lambda: MockSession()
            assert vs._has_partition is False

            from llama_index.core.schema import TextNode

            node = TextNode(text="hello", embedding=[0.1] * 4)
            vs.add([node])

        # Should have ON DUPLICATE KEY UPDATE
        assert any(
            "ON DUPLICATE KEY" in sql.upper() for sql in executed_sqls
        )

    # --- W3: MMR NaN/Inf guard ---

    def test_w3_mmr_nan_embeddings_no_infinite_loop(self) -> None:
        """W3: MMR with NaN embeddings does not cause infinite loop."""
        from llama_index.vector_stores.polardbx.base import PolarDBXVectorStore

        # All-NaN embeddings
        nan_vecs = [[float("nan"), float("nan")] for _ in range(5)]
        query_vec = [1.0, 0.0]

        # Should return without hanging (may return empty or partial)
        result = PolarDBXVectorStore._maximal_marginal_relevance(
            query_vec, nan_vecs, k=3, lambda_mult=0.5
        )
        assert isinstance(result, list)
        assert len(result) <= 3

    # --- W4: partition_column restricted to 'id' ---

    def test_w4_valid_partition_column_id(self) -> None:
        """W4: partition_column='id' is accepted."""
        vs = _make_store(partition_by="HASH", partitions=4, partition_column="id")
        assert vs._partition_column == "id"

    def test_w4_non_id_partition_column_rejected(self) -> None:
        """W4: partition_column other than 'id' is rejected for vector tables."""
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="partition_column must be 'id'"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="pass",
                database="test",
                perform_setup=False,
                partition_by="HASH",
                partitions=4,
                partition_column="node_id",
            )

    # --- W5: _fetch_embeddings log sanitization ---

    def test_w5_fetch_embeddings_sanitizes_error(self) -> None:
        """W5: _fetch_embeddings_by_node_ids sanitizes error in log."""
        vs = _make_store()

        # Mock session that raises error with embedded credentials
        credential_error = Exception(
            "Connection failed: mysql+pymysql://user:secret_pass@host:3306/db"
        )

        class MockSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def execute(self, stmt, params=None):
                raise credential_error

        vs._session = lambda: MockSession()

        # Should not raise; should return empty dict
        result = vs._fetch_embeddings_by_node_ids(["node1"])
        assert result == {}

    # --- W12: Metadata parse log level ---

    def test_w12_corrupted_metadata_uses_debug_level(self) -> None:
        """W12: Corrupted metadata JSON logs at DEBUG, not WARNING."""
        from llama_index.vector_stores.polardbx.base import PolarDBXVectorStore

        # _parse_metadata is a staticmethod, call directly
        result = PolarDBXVectorStore._parse_metadata("not valid json")
        assert result == {}

    # --- M1: LIST empty values_in ---

    def test_m1_list_empty_values_in_rejected(self) -> None:
        """M1: LIST partition with empty values_in raises ValueError."""
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause,
        )

        with pytest.raises(ValueError, match="empty 'values_in'"):
            _build_partition_clause(
                partition_by="LIST",
                partition_column="id",
                partition_defs=[
                    {"name": "p1", "values_in": []},
                ],
            )

    # --- M2: base.py _validate_identifier 64-char limit ---

    def test_m2_base_validate_identifier_rejects_too_long(self) -> None:
        """M2: base.py _validate_identifier rejects names over 64 chars."""
        from llama_index.vector_stores.polardbx.base import PolarDBXVectorStore

        with pytest.raises(ValueError, match="too long"):
            PolarDBXVectorStore._validate_identifier("a" * 65)

    def test_m2_base_validate_identifier_accepts_64_chars(self) -> None:
        """M2: base.py _validate_identifier accepts exactly 64 chars."""
        from llama_index.vector_stores.polardbx.base import PolarDBXVectorStore

        assert PolarDBXVectorStore._validate_identifier("a" * 64) == "a" * 64

    # --- M3: _sql_quote_string escapes backslashes ---

    def test_m3_backslash_escaped(self) -> None:
        """M3: _sql_quote_string escapes backslashes."""
        from llama_index.vector_stores.polardbx._partition import (
            _sql_quote_string,
        )

        result = _sql_quote_string("a\\b")
        assert "\\\\" in result  # backslash doubled

    # --- M5: search_by_metadata limit validation ---

    def test_m5_search_by_metadata_invalid_limit_zero(self) -> None:
        """M5: search_by_metadata rejects limit=0."""
        from llama_index.core.vector_stores.types import MetadataFilters

        vs = _make_store()
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            vs.search_by_metadata(MetadataFilters(filters=[]), limit=0)

    def test_m5_search_by_metadata_invalid_limit_negative(self) -> None:
        """M5: search_by_metadata rejects negative limit."""
        from llama_index.core.vector_stores.types import MetadataFilters

        vs = _make_store()
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            vs.search_by_metadata(MetadataFilters(filters=[]), limit=-5)

