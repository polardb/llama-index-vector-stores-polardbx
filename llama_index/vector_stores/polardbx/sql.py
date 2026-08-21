"""PolarDB-X SQL Database integration with DDL reflection compatibility.

This module provides:
- PolarDBXDialect: A custom SQLAlchemy dialect that fixes PolarDB-X DDL
  reflection issues (tab indentation, ENUM value spacing) via subclassing,
  with zero global side effects.
- PolarDBXSQLDatabase: A thin wrapper around LlamaIndex's
  SQLDatabase that auto-swaps the connection URI to use our dialect.

Since SQLAlchemy and pymysql are already core dependencies of this
package, no optional ``[sql]`` extra is needed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import MetaData, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.dialects.mysql.pymysql import MySQLDialect_pymysql
from sqlalchemy.dialects.mysql.reflection import MySQLTableDefinitionParser
from sqlalchemy.engine import Engine
from sqlalchemy.types import UserDefinedType
from sqlalchemy.util import memoized_property

logger = logging.getLogger(__name__)


class PolarDBXVector(UserDefinedType):
    """PolarDB-X VECTOR type for SQLAlchemy schema reflection.

    This type exists so that PolarDBXDialect can correctly reflect tables
    containing VECTOR columns without crashing. It does NOT provide vector
    operations — use raw SQL for vector similarity search.
    """

    cache_ok = True

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension

    def get_col_spec(self, **kw: Any) -> str:
        if self.dimension is not None:
            return f"VECTOR({self.dimension})"
        return "VECTOR"

    @property
    def python_type(self) -> type:  # type: ignore[override]
        return list


class PolarDBXTableDefinitionParser(MySQLTableDefinitionParser):
    """MySQLTableDefinitionParser with PolarDB-X DDL format fixes.

    PolarDB-X SHOW CREATE TABLE output has two format differences
    from standard MySQL:
    1. Tab indentation instead of two-space indentation
    2. ENUM/SET value lists have spaces after commas

    This parser normalizes both before delegating to the parent parser.

    Note: PolarDB-X ``VECTOR INDEX`` lines (e.g.
    ``VECTOR INDEX `vi`(`embedding`) M=6 DISTANCE=COSINE``) are not
    recognized by the upstream MySQL parser and will be skipped with a
    warning. This is expected — the index info is not needed for SQL
    query generation via SQLDatabase.
    """

    def parse(self, show_create: str, charset: Optional[str]) -> Any:
        # Fix 1: Tab indentation -> two spaces (standard MySQL format)
        show_create = show_create.replace("\n\t", "\n  ")
        # Fix 2: ENUM/SET value list spacing normalization
        #  enum('A', 'B') -> enum('A','B')
        #
        # This regex matches the pattern '<quote>,<space><quote>' which
        # only occurs between ENUM/SET values in valid DDL. String literals
        # like DEFAULT 'hello, world' are unaffected because the comma is
        # inside the quotes, not between them.
        show_create = re.sub(r"',\s+'", "','", show_create)
        return super().parse(show_create, charset)


class PolarDBXDialect(MySQLDialect_pymysql):
    """MySQL pymysql dialect with PolarDB-X DDL reflection fixes.

    This dialect overrides _tabledef_parser to return a
    PolarDBXTableDefinitionParser instead of the default parser.
    Only connections using this dialect (polardbx+pymysql://)
    are affected; other MySQL connections are completely unaffected.
    """

    # Enable SQLAlchemy SQL compilation caching. This dialect only
    # overrides DDL reflection logic, not SQL compilation, so caching
    # is safe and avoids repeated deprecation warnings.
    supports_statement_cache = True

    ischema_names = {
        **MySQLDialect_pymysql.ischema_names,
        "vector": PolarDBXVector,
    }

    @memoized_property
    def _tabledef_parser(self) -> PolarDBXTableDefinitionParser:
        preparer = self.identifier_preparer
        return PolarDBXTableDefinitionParser(self, preparer)


def _register_dialect() -> None:
    """Register the polardbx dialect with SQLAlchemy's registry.

    This is a fallback for the entry-point in pyproject.toml which also
    registers ``polardbx``. The registry call specifically registers
    ``polardbx.pymysql`` (the dialect+driver form), ensuring URIs work
    even when the package is imported without being pip-installed.
    """
    from sqlalchemy.dialects import registry

    registry.register(
        "polardbx.pymysql",
        "llama_index.vector_stores.polardbx.sql",
        "PolarDBXDialect",
    )


# Register at import time so polardbx+pymysql:// URIs work
_register_dialect()


# LlamaIndex's SQLDatabase is in llama_index.core.utilities.sql_wrapper.
# It is based on LangChain's SQLDatabase but adds methods like
# get_table_columns(), get_single_table_info(), run_sql(), and
# insert_into_table().
from llama_index.core.utilities.sql_wrapper import SQLDatabase


class PolarDBXSQLDatabase(SQLDatabase):
    """SQLDatabase with PolarDB-X DDL reflection compatibility.

    Automatically applies DDL format fixes when reflecting table
    schemas from PolarDB-X. Usage is identical to LlamaIndex's
    SQLDatabase:

        from llama_index.vector_stores.polardbx import PolarDBXSQLDatabase
        db = PolarDBXSQLDatabase.from_uri(
            "mysql+pymysql://user:pass@host:3306/db"
        )
        db.run_sql("SELECT * FROM my_table")
        db.get_single_table_info("my_table")

    If the database contains a table with corrupted partition metadata,
    the standard ``SHOW FULL TABLES`` query used during initialization
    will raise ``TDDL-4700``. This class detects that failure and falls
    back to a resilient initialization path that queries
    ``information_schema`` directly and skips tables that cannot be
    reflected.
    """

    @classmethod
    def from_uri(
        cls,
        database_uri: str,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> "PolarDBXSQLDatabase":
        # Ensure dialect is registered
        _register_dialect()

        # Auto-swap mysql+pymysql:// -> polardbx+pymysql://
        if database_uri.startswith("mysql+pymysql://"):
            database_uri = (
                "polardbx+pymysql://" + database_uri[len("mysql+pymysql://") :]
            )
        elif database_uri.startswith("mysql://"):
            database_uri = "polardbx+pymysql://" + database_uri[len("mysql://") :]

        return super().from_uri(database_uri, engine_args, **kwargs)  # type: ignore[return-value]

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[dict] = None,
        view_support: bool = False,
        max_string_length: int = 300,
    ) -> None:
        """Initialize with resilient fallback for corrupted table metadata.

        If the standard initialization fails (e.g. due to a table with
        corrupted partition metadata causing ``TDDL-4700``), falls back
        to a resilient path that enumerates tables via
        ``information_schema`` and reflects them individually, skipping
        any that cannot be reflected.
        """
        try:
            super().__init__(
                engine,
                schema=schema,
                metadata=metadata,
                ignore_tables=ignore_tables,
                include_tables=include_tables,
                sample_rows_in_table_info=sample_rows_in_table_info,
                indexes_in_table_info=indexes_in_table_info,
                custom_table_info=custom_table_info,
                view_support=view_support,
                max_string_length=max_string_length,
            )
        except Exception:
            logger.warning(
                "Standard SQLDatabase initialization failed, possibly "
                "due to corrupted table metadata (TDDL-4700). Falling "
                "back to resilient initialization."
            )
            self._resilient_init(
                engine=engine,
                schema=schema,
                metadata=metadata,
                ignore_tables=ignore_tables,
                include_tables=include_tables,
                sample_rows_in_table_info=sample_rows_in_table_info,
                indexes_in_table_info=indexes_in_table_info,
                custom_table_info=custom_table_info,
                view_support=view_support,
                max_string_length=max_string_length,
            )

    def _resilient_init(
        self,
        engine: Engine,
        schema: Optional[str],
        metadata: Optional[MetaData],
        ignore_tables: Optional[List[str]],
        include_tables: Optional[List[str]],
        sample_rows_in_table_info: int,
        indexes_in_table_info: bool,
        custom_table_info: Optional[dict],
        view_support: bool,
        max_string_length: int,
    ) -> None:
        """Initialize SQLDatabase, skipping tables with corrupted metadata."""
        if include_tables and ignore_tables:
            raise ValueError(
                "Cannot specify both include_tables and ignore_tables"
            )

        self._engine = engine
        self._schema = schema
        self._inspector = sa_inspect(self._engine)

        # Enumerate tables via information_schema, which does not
        # trigger partition metadata loading (the cause of TDDL-4700).
        self._all_tables = self._get_tables_from_information_schema(
            schema, view_support
        )

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing = self._include_tables - self._all_tables
            if missing:
                raise ValueError(
                    f"include_tables {missing} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing = self._ignore_tables - self._all_tables
            if missing:
                raise ValueError(
                    f"ignore_tables {missing} not found in database"
                )
        usable = self.get_usable_table_names()
        self._usable_tables = (
            set(usable) if usable else self._all_tables
        )

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")
        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names "
                    "as keys and the desired table info as values"
                )
            intersection = set(self._custom_table_info).intersection(
                self._all_tables
            )
            self._custom_table_info = {
                t: info
                for t, info in self._custom_table_info.items()
                if t in intersection
            }

        self._max_string_length = max_string_length

        # Reflect tables one by one, skipping any that fail.
        self._metadata = metadata or MetaData()
        skipped: List[str] = []
        for table_name in self._usable_tables:
            try:
                self._metadata.reflect(
                    bind=self._engine,
                    only=[table_name],
                    schema=self._schema,
                    views=view_support,
                )
            except Exception as exc:
                logger.warning(
                    "Skipping table '%s' due to reflection error: %s",
                    table_name,
                    exc,
                )
                skipped.append(table_name)

        if skipped:
            logger.warning(
                "Resilient initialization: skipped %d table(s) with "
                "corrupted metadata: %s",
                len(skipped),
                ", ".join(sorted(skipped)),
            )

    def _get_tables_from_information_schema(
        self, schema: Optional[str], view_support: bool
    ) -> Set[str]:
        """Enumerate tables via information_schema.

        Unlike ``SHOW FULL TABLES``, this query does not trigger
        partition metadata loading, making it resilient to tables
        with corrupted partition info (``TDDL-4700``).
        """
        with self._engine.connect() as conn:
            if schema:
                rows = conn.execute(
                    text(
                        "SELECT TABLE_NAME, TABLE_TYPE "
                        "FROM information_schema.tables "
                        "WHERE TABLE_SCHEMA = :schema"
                    ),
                    {"schema": schema},
                ).fetchall()
            else:
                rows = conn.execute(
                    text(
                        "SELECT TABLE_NAME, TABLE_TYPE "
                        "FROM information_schema.tables "
                        "WHERE TABLE_SCHEMA = DATABASE()"
                    )
                ).fetchall()
            tables = {r[0] for r in rows if r[1] == "BASE TABLE"}
            if view_support:
                tables |= {r[0] for r in rows if r[1] == "VIEW"}
            return tables


# ---------------------------------------------------------------------------
# Standalone DDL helper for non-vector partitioned tables
# ---------------------------------------------------------------------------


def create_partitioned_table(
    uri: str,
    table_name: str,
    columns: List[str],
    partition_by: Optional[str] = None,
    partition_column: str = "id",
    partitions: int = 0,
    broadcast: bool = False,
    locality: Optional[str] = None,
    partition_defs: Optional[List[Dict[str, Any]]] = None,
    if_not_exists: bool = True,
) -> None:
    """Create a table on PolarDB-X with optional partition clauses.

    This function executes raw DDL via SQLAlchemy. Since SQLAlchemy
    and pymysql are core dependencies of this package, no extra
    installation is needed.

    Args:
        uri: Connection URI, e.g.
            ``"mysql+pymysql://user:pass@host:3306/db"``.
            Will be auto-swapped to ``polardbx+pymysql://``.
        table_name: The table name.
        columns: Column definitions as SQL strings, e.g.
            ``["id BIGINT NOT NULL AUTO_INCREMENT",
            "name VARCHAR(255)", "PRIMARY KEY (id)"]``.
            Must be constructed by the developer — never accept
            user-supplied input to prevent SQL injection.
        partition_by: Partition strategy: "HASH", "KEY", "RANGE", or
            "LIST". None for single table.
        partition_column: Column to partition on. Defaults to "id".
        partitions: Number of partitions (HASH/KEY only).
        broadcast: If True, create a broadcast table.
        locality: Storage node, e.g. "dn=xxx".
        partition_defs: Partition definitions (RANGE/LIST only).
        if_not_exists: If True, add IF NOT EXISTS.

    Example:
        .. code-block:: python

            from llama_index.vector_stores.polardbx import create_partitioned_table

            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=[
                    "id BIGINT NOT NULL AUTO_INCREMENT",
                    "user_id BIGINT NOT NULL",
                    "amount DECIMAL(10,2)",
                    "created_at DATETIME",
                    "PRIMARY KEY (id)",
                ],
                partition_by="HASH",
                partition_column="user_id",
                partitions=16,
            )
    """
    from llama_index.vector_stores.polardbx._partition import (
        _build_partition_clause,
        _validate_identifier,
    )

    _register_dialect()

    # Validate identifiers to prevent SQL injection
    _validate_identifier(table_name, "table name")
    if partition_by:
        _validate_identifier(partition_column, "partition column")

    # Auto-swap mysql:// to polardbx://
    if uri.startswith("mysql+pymysql://"):
        uri = "polardbx+pymysql://" + uri[len("mysql+pymysql://") :]
    elif uri.startswith("mysql://"):
        uri = "polardbx+pymysql://" + uri[len("mysql://") :]

    # Validate params
    if partition_by:
        partition_by = partition_by.upper()
        if partition_by not in ("HASH", "KEY", "RANGE", "LIST"):
            raise ValueError(
                f"Invalid partition_by: {partition_by}. "
                "Must be 'HASH', 'KEY', 'RANGE', or 'LIST'."
            )
        if partition_by in ("HASH", "KEY") and partitions <= 0:
            raise ValueError("partitions must be > 0 for HASH/KEY partitioning.")
        if partition_by in ("RANGE", "LIST") and not partition_defs:
            raise ValueError("partition_defs required for RANGE/LIST partitioning.")
    if broadcast and partition_by:
        raise ValueError("broadcast and partition_by are mutually exclusive.")

    # Build DDL
    exists_clause = "IF NOT EXISTS " if if_not_exists else ""
    col_defs = ",\n    ".join(columns)
    partition_clause = _build_partition_clause(
        partition_by=partition_by,
        partition_column=partition_column,
        partitions=partitions,
        broadcast=broadcast,
        locality=locality,
        partition_defs=partition_defs,
    )

    ddl = (
        f"CREATE TABLE {exists_clause}`{table_name}` (\n"
        f"    {col_defs}\n"
        f"){partition_clause}"
    )

    from sqlalchemy import create_engine, text

    engine = create_engine(uri)
    try:
        with engine.connect() as conn:
            try:
                conn.execute(text(ddl))
                conn.commit()
            except Exception as e:
                if partition_clause:
                    err_msg = str(e).lower()
                    if "partition" in err_msg and (
                        "not support" in err_msg
                        or "do not support" in err_msg
                    ):
                        from llama_index.vector_stores.polardbx.base import (
                            NotSupportedError,
                        )

                        raise NotSupportedError(
                            "PolarDB-X does not support partitioning on this "
                            "instance. This may occur on certain DN "
                            "versions. Try upgrading the DN version, or "
                            "remove partition parameters to create a "
                            "non-partitioned table."
                        ) from e
                raise
    finally:
        # W1: Always dispose engine to prevent resource leak on exception
        engine.dispose()
