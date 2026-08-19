"""Unit tests for the SQL module (PolarDBXDialect + PolarDBXSQLDatabase).

These tests don't require a database connection. They verify:
1. DDL reflection fix logic (tab indentation, ENUM spacing)
2. URI auto-swap in from_uri
3. Dialect registration and name compatibility
"""

from unittest.mock import patch

from sqlalchemy.dialects.mysql.reflection import MySQLTableDefinitionParser


def test_import_polardbx_sql_database() -> None:
    """Test that PolarDBXSQLDatabase can be imported."""
    from llama_index.vector_stores.polardbx import PolarDBXSQLDatabase

    assert PolarDBXSQLDatabase is not None


def test_import_polardbx_dialect() -> None:
    """Test that PolarDBXDialect can be imported."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

    assert PolarDBXDialect is not None


def test_dialect_registered_in_registry() -> None:
    """Test that polardbx.pymysql is registered in SQLAlchemy's registry."""
    from sqlalchemy.dialects import registry

    dialect_cls = registry.load("polardbx.pymysql")
    assert dialect_cls is not None

    from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

    assert dialect_cls is PolarDBXDialect


def test_dialect_name_is_mysql() -> None:
    """Test that PolarDBXDialect.name returns 'mysql' for SQLDatabase compatibility."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

    assert PolarDBXDialect.name == "mysql"


def test_dialect_driver_is_pymysql() -> None:
    """Test that PolarDBXDialect.driver returns 'pymysql'."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

    assert PolarDBXDialect.driver == "pymysql"


def test_ddl_fix_tab_indentation() -> None:
    """Test that tab indentation is normalized to two-space indentation."""

    # Simulate PolarDB-X DDL with tab indentation
    ddl_polardbx = (
        "CREATE TABLE `test` (\n"
        "\t`id` int NOT NULL,\n"
        "\t`name` varchar(255) DEFAULT NULL\n"
        ") ENGINE=InnoDB"
    )

    # Mock super().parse() to capture the transformed input
    with patch.object(
        MySQLTableDefinitionParser, "parse", return_value=None
    ) as mock_parse:
        parser = _create_parser()
        parser.parse(ddl_polardbx, "utf8mb4")

        transformed = mock_parse.call_args[0][0]
        assert "\n\t" not in transformed
        assert "\n  `id`" in transformed
        assert "\n  `name`" in transformed


def test_ddl_fix_enum_spacing() -> None:
    """Test that ENUM value spacing is normalized."""

    # PolarDB-X puts spaces after commas in ENUM: enum('A', 'B')
    # Standard MySQL has no spaces: enum('A','B')
    ddl_polardbx = (
        "CREATE TABLE `test` (\n"
        "  `status` enum('A', 'B', 'C') DEFAULT NULL\n"
        ") ENGINE=InnoDB"
    )

    with patch.object(
        MySQLTableDefinitionParser, "parse", return_value=None
    ) as mock_parse:
        parser = _create_parser()
        parser.parse(ddl_polardbx, "utf8mb4")

        transformed = mock_parse.call_args[0][0]
        assert "enum('A','B','C')" in transformed
        assert "enum('A', 'B', 'C')" not in transformed


def test_ddl_fix_both_tab_and_enum() -> None:
    """Test that both tab indentation and ENUM spacing are fixed together."""

    ddl_polardbx = (
        "CREATE TABLE `users` (\n"
        "\t`id` int NOT NULL AUTO_INCREMENT,\n"
        "\t`role` enum('admin', 'user', 'guest') NOT NULL DEFAULT 'user',\n"
        "\t`name` varchar(100) NOT NULL\n"
        ") ENGINE=InnoDB AUTO_INCREMENT=1"
    )

    with patch.object(
        MySQLTableDefinitionParser, "parse", return_value=None
    ) as mock_parse:
        parser = _create_parser()
        parser.parse(ddl_polardbx, "utf8mb4")

        transformed = mock_parse.call_args[0][0]
        assert "\n\t" not in transformed
        assert "\n  `id`" in transformed
        assert "\n  `role`" in transformed
        assert "enum('admin','user','guest')" in transformed


def test_ddl_passthrough_standard_mysql() -> None:
    """Test that standard MySQL DDL (no tab, no ENUM spacing) is passed through."""

    ddl_standard = (
        "CREATE TABLE `test` (\n"
        "  `id` int NOT NULL,\n"
        "  `status` enum('A','B') DEFAULT NULL\n"
        ") ENGINE=InnoDB"
    )

    with patch.object(
        MySQLTableDefinitionParser, "parse", return_value=None
    ) as mock_parse:
        parser = _create_parser()
        parser.parse(ddl_standard, "utf8mb4")

        transformed = mock_parse.call_args[0][0]
        assert transformed == ddl_standard


def test_from_uri_swaps_mysql_pymysql() -> None:
    """Test that from_uri auto-swaps mysql+pymysql:// to polardbx+pymysql://."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    with patch.object(
        PolarDBXSQLDatabase.__mro__[1], "from_uri", return_value="mock_db"
    ) as mock_from_uri:
        PolarDBXSQLDatabase.from_uri("mysql+pymysql://user:pass@host:3306/db")

        actual_uri = mock_from_uri.call_args[0][0]
        assert actual_uri.startswith("polardbx+pymysql://")
        assert "user:pass@host:3306/db" in actual_uri


def test_from_uri_swaps_mysql_plain() -> None:
    """Test that from_uri auto-swaps mysql:// to polardbx+pymysql://."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    with patch.object(
        PolarDBXSQLDatabase.__mro__[1], "from_uri", return_value="mock_db"
    ) as mock_from_uri:
        PolarDBXSQLDatabase.from_uri("mysql://user:pass@host:3306/db")

        actual_uri = mock_from_uri.call_args[0][0]
        assert actual_uri.startswith("polardbx+pymysql://")
        assert "user:pass@host:3306/db" in actual_uri


def test_from_uri_keeps_polardbx_uri() -> None:
    """Test that explicit polardbx+pymysql:// URI is kept as-is."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    with patch.object(
        PolarDBXSQLDatabase.__mro__[1], "from_uri", return_value="mock_db"
    ) as mock_from_uri:
        PolarDBXSQLDatabase.from_uri("polardbx+pymysql://user:pass@host:3306/db")

        actual_uri = mock_from_uri.call_args[0][0]
        assert actual_uri == "polardbx+pymysql://user:pass@host:3306/db"


def test_dialect_registers_vector_type() -> None:
    """Test that PolarDBXDialect registers VECTOR in ischema_names."""
    from llama_index.vector_stores.polardbx.sql import (
        PolarDBXDialect,
        PolarDBXVector,
    )

    assert "vector" in PolarDBXDialect.ischema_names
    assert PolarDBXDialect.ischema_names["vector"] is PolarDBXVector


def test_vector_type_accepts_dimension() -> None:
    """Test that PolarDBXVector accepts a dimension argument."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXVector

    v = PolarDBXVector(128)
    assert v.dimension == 128
    assert v.get_col_spec() == "VECTOR(128)"


def test_vector_type_no_dimension() -> None:
    """Test that PolarDBXVector works without a dimension argument."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXVector

    v = PolarDBXVector()
    assert v.dimension is None
    assert v.get_col_spec() == "VECTOR"


def test_vector_type_cache_ok() -> None:
    """Test that PolarDBXVector has cache_ok=True for statement caching."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXVector

    assert PolarDBXVector.cache_ok is True


def test_dialect_ischema_names_includes_all_mysql_types() -> None:
    """Test that PolarDBXDialect.ischema_names includes all standard MySQL types."""
    from sqlalchemy.dialects.mysql.pymysql import MySQLDialect_pymysql

    from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

    for key, val in MySQLDialect_pymysql.ischema_names.items():
        assert key in PolarDBXDialect.ischema_names
        assert PolarDBXDialect.ischema_names[key] is val


def test_dialect_supports_statement_cache() -> None:
    """Test that PolarDBXDialect enables SQL compilation caching."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

    assert PolarDBXDialect.supports_statement_cache is True


def test_create_engine_uses_polardbx_dialect() -> None:
    """Test that create_engine with polardbx+pymysql:// uses PolarDBXDialect."""
    from sqlalchemy import create_engine

    from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

    engine = create_engine("polardbx+pymysql://user:pass@localhost/test")
    assert isinstance(engine.dialect, PolarDBXDialect)


def test_create_engine_standard_mysql_unaffected() -> None:
    """Test that mysql+pymysql:// still uses the original MySQLDialect_pymysql."""
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.mysql.pymysql import MySQLDialect_pymysql

    from llama_index.vector_stores.polardbx.sql import PolarDBXDialect

    engine = create_engine("mysql+pymysql://user:pass@localhost/test")
    assert isinstance(engine.dialect, MySQLDialect_pymysql)
    assert not isinstance(engine.dialect, PolarDBXDialect)


# --- Helpers ---


def _create_parser():
    """Create a PolarDBXTableDefinitionParser instance for testing."""

    from llama_index.vector_stores.polardbx.sql import (
        PolarDBXDialect,
        PolarDBXTableDefinitionParser,
    )

    dialect = PolarDBXDialect()
    preparer = dialect.identifier_preparer
    return PolarDBXTableDefinitionParser(dialect, preparer)
