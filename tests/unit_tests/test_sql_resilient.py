"""Unit tests for PolarDBXSQLDatabase resilient initialization.

These tests verify that PolarDBXSQLDatabase gracefully handles
databases containing tables with corrupted partition metadata
(TDDL-4700) by falling back to information_schema enumeration
and per-table reflection that skips corrupted tables.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import MetaData


def _make_mock_engine(table_rows, reflect_failures=None):
    """Create a mock SQLAlchemy engine.

    Args:
        table_rows: List of (TABLE_NAME, TABLE_TYPE) tuples returned
            by the information_schema query.
        reflect_failures: Set of table names that should fail during
            per-table reflection.
    """
    reflect_failures = reflect_failures or set()
    engine = MagicMock()

    # Mock connection context manager
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = table_rows
    mock_conn.execute.return_value = mock_result
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_conn)
    mock_cm.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = mock_cm

    # Mock metadata.reflect to raise for specified tables
    def mock_reflect(bind=None, only=None, schema=None, views=False):
        table_name = only[0] if only else None
        if table_name in reflect_failures:
            raise Exception(
                f"(TDDL-4700) Failed to load partitionInfo[{table_name}]"
            )

    # Patch MetaData.reflect at the instance level
    engine._mock_reflect = mock_reflect
    return engine


def test_resilient_init_fallback_on_corrupted_metadata():
    """When super().__init__ raises, resilient init skips corrupted tables."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    table_rows = [
        ("healthy_table", "BASE TABLE"),
        ("corrupted_table", "BASE TABLE"),
        ("my_view", "VIEW"),
    ]
    engine = _make_mock_engine(
        table_rows, reflect_failures={"corrupted_table"}
    )

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            side_effect=Exception("TDDL-4700"),
        ),
        patch("sqlalchemy.MetaData.reflect", autospec=True) as mock_reflect,
        patch(
            "llama_index.vector_stores.polardbx.sql.sa_inspect"
        ) as mock_inspect,
        patch.object(
            PolarDBXSQLDatabase,
            "get_usable_table_names",
            return_value=["healthy_table", "corrupted_table"],
        ),
    ):
        # Make MetaData.reflect fail for corrupted_table, succeed for others
        def reflect_side_effect(self, bind=None, only=None, **kw):
            tname = only[0] if only else None
            if tname == "corrupted_table":
                raise Exception("TDDL-4700 corrupted")

        mock_reflect.side_effect = reflect_side_effect
        mock_inspect.return_value = MagicMock()

        db = PolarDBXSQLDatabase(engine)

    # Should have enumerated tables via information_schema
    assert "healthy_table" in db._all_tables
    assert "corrupted_table" in db._all_tables
    # View should only appear if view_support=True (it is not)
    assert "my_view" not in db._all_tables

    # MetaData.reflect should have been called for each usable table
    called_tables = [
        call.args[2][0] if len(call.args) > 2 and isinstance(call.args[2], list)
        else call.kwargs.get("only", [None])[0]
        for call in mock_reflect.call_args_list
    ]
    assert "healthy_table" in called_tables
    assert "corrupted_table" in called_tables


def test_resilient_init_includes_views_when_enabled():
    """Resilient init includes views when view_support=True."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    table_rows = [
        ("my_table", "BASE TABLE"),
        ("my_view", "VIEW"),
    ]
    engine = _make_mock_engine(table_rows)

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            side_effect=Exception("TDDL-4700"),
        ),
        patch("sqlalchemy.MetaData.reflect"),
        patch(
            "llama_index.vector_stores.polardbx.sql.sa_inspect"
        ) as mock_inspect,
        patch.object(
            PolarDBXSQLDatabase,
            "get_usable_table_names",
            return_value=["my_table", "my_view"],
        ),
    ):
        mock_inspect.return_value = MagicMock()
        db = PolarDBXSQLDatabase(
            engine,
            view_support=True,
        )

    assert "my_table" in db._all_tables
    assert "my_view" in db._all_tables


def test_resilient_init_validates_include_tables():
    """Resilient init validates include_tables against enumerated tables."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    table_rows = [("table_a", "BASE TABLE")]
    engine = _make_mock_engine(table_rows)

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            side_effect=Exception("TDDL-4700"),
        ),
        patch("sqlalchemy.MetaData.reflect"),
        patch(
            "llama_index.vector_stores.polardbx.sql.sa_inspect"
        ) as mock_inspect,
        patch.object(
            PolarDBXSQLDatabase,
            "get_usable_table_names",
            return_value=["table_a"],
        ),
    ):
        mock_inspect.return_value = MagicMock()

        with pytest.raises(ValueError, match="include_tables"):
            PolarDBXSQLDatabase(
                engine,
                include_tables=["table_a", "nonexistent"],
            )


def test_resilient_init_validates_ignore_tables():
    """Resilient init validates ignore_tables against enumerated tables."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    table_rows = [("table_a", "BASE TABLE")]
    engine = _make_mock_engine(table_rows)

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            side_effect=Exception("TDDL-4700"),
        ),
        patch("sqlalchemy.MetaData.reflect"),
        patch(
            "llama_index.vector_stores.polardbx.sql.sa_inspect"
        ) as mock_inspect,
        patch.object(
            PolarDBXSQLDatabase,
            "get_usable_table_names",
            return_value=["table_a"],
        ),
    ):
        mock_inspect.return_value = MagicMock()

        with pytest.raises(ValueError, match="ignore_tables"):
            PolarDBXSQLDatabase(
                engine,
                ignore_tables=["nonexistent"],
            )


def test_resilient_init_rejects_both_include_and_ignore():
    """Resilient init rejects specifying both include_tables and ignore_tables."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    engine = _make_mock_engine([("t", "BASE TABLE")])

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            side_effect=Exception("TDDL-4700"),
        ),
        patch("sqlalchemy.MetaData.reflect"),
        patch(
            "llama_index.vector_stores.polardbx.sql.sa_inspect"
        ) as mock_inspect,
    ):
        mock_inspect.return_value = MagicMock()

        with pytest.raises(ValueError, match="Cannot specify both"):
            PolarDBXSQLDatabase(
                engine,
                include_tables=["t"],
                ignore_tables=["t"],
            )


def test_resilient_init_validates_sample_rows_type():
    """Resilient init validates sample_rows_in_table_info is an integer."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    engine = _make_mock_engine([("t", "BASE TABLE")])

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            side_effect=Exception("TDDL-4700"),
        ),
        patch("sqlalchemy.MetaData.reflect"),
        patch(
            "llama_index.vector_stores.polardbx.sql.sa_inspect"
        ) as mock_inspect,
        patch.object(
            PolarDBXSQLDatabase,
            "get_usable_table_names",
            return_value=[],
        ),
    ):
        mock_inspect.return_value = MagicMock()

        with pytest.raises(TypeError, match="sample_rows_in_table_info"):
            PolarDBXSQLDatabase(
                engine,
                sample_rows_in_table_info="three",
            )


def test_resilient_init_validates_custom_table_info_type():
    """Resilient init validates custom_table_info is a dict."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    engine = _make_mock_engine([("t", "BASE TABLE")])

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            side_effect=Exception("TDDL-4700"),
        ),
        patch("sqlalchemy.MetaData.reflect"),
        patch(
            "llama_index.vector_stores.polardbx.sql.sa_inspect"
        ) as mock_inspect,
        patch.object(
            PolarDBXSQLDatabase,
            "get_usable_table_names",
            return_value=[],
        ),
    ):
        mock_inspect.return_value = MagicMock()

        with pytest.raises(TypeError, match="table_info must be"):
            PolarDBXSQLDatabase(
                engine,
                custom_table_info=["not", "a", "dict"],
            )


def test_resilient_init_preserves_metadata_arg():
    """Resilient init uses provided metadata object if given."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    engine = _make_mock_engine([("t", "BASE TABLE")])
    custom_meta = MetaData()

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            side_effect=Exception("TDDL-4700"),
        ),
        patch("sqlalchemy.MetaData.reflect"),
        patch(
            "llama_index.vector_stores.polardbx.sql.sa_inspect"
        ) as mock_inspect,
        patch.object(
            PolarDBXSQLDatabase,
            "get_usable_table_names",
            return_value=[],
        ),
    ):
        mock_inspect.return_value = MagicMock()
        db = PolarDBXSQLDatabase(engine, metadata=custom_meta)

    assert db._metadata is custom_meta


def test_information_schema_query_with_schema():
    """_get_tables_from_information_schema uses schema when provided."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    table_rows = [("t1", "BASE TABLE")]
    engine = _make_mock_engine(table_rows)

    db = PolarDBXSQLDatabase.__new__(PolarDBXSQLDatabase)
    db._engine = engine

    result = db._get_tables_from_information_schema("mydb", False)

    assert result == {"t1"}
    # Verify the query passed schema as a positional parameter
    mock_conn = engine.connect.return_value.__enter__.return_value
    call_args = mock_conn.execute.call_args_list[0]
    # SQLAlchemy execute(statement, params) passes params as positional arg
    assert call_args.args[1] == {"schema": "mydb"}


def test_information_schema_query_without_schema():
    """_get_tables_from_information_schema uses DATABASE() when no schema."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    table_rows = [("t1", "BASE TABLE"), ("t2", "BASE TABLE")]
    engine = _make_mock_engine(table_rows)

    db = PolarDBXSQLDatabase.__new__(PolarDBXSQLDatabase)
    db._engine = engine

    result = db._get_tables_from_information_schema(None, False)

    assert result == {"t1", "t2"}


def test_super_init_succeeds_no_fallback():
    """When super().__init__ succeeds, resilient init is not called."""
    from llama_index.vector_stores.polardbx.sql import PolarDBXSQLDatabase

    engine = MagicMock()

    with (
        patch.object(
            PolarDBXSQLDatabase.__mro__[1],
            "__init__",
            return_value=None,
        ) as mock_super_init,
        patch.object(
            PolarDBXSQLDatabase, "_resilient_init"
        ) as mock_resilient,
    ):
        PolarDBXSQLDatabase(engine)

    mock_super_init.assert_called_once()
    mock_resilient.assert_not_called()
