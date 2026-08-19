"""PolarDB-X Vector Store.

This module provides a LlamaIndex BasePydanticVectorStore implementation using
PolarDB-X with native vector search capabilities (VECTOR data type + HNSW index).

Dual-version compatibility:
    - Old versions: VEC_DISTANCE_COSINE / VEC_DISTANCE_EUCLIDEAN
    - v3: VEC_DISTANCE() auto-inference, INNER_PRODUCT, EF_CONSTRUCTION,
          information_schema.VECTOR_INDEXES, dbms_vidx.preload, etc.

The store probes capabilities at init time and caches the results, then
branches at runtime based on the detected feature set.
"""

import difflib
import json
import logging
import re
import threading
import time
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
)
from urllib.parse import quote_plus

import sqlalchemy
import sqlalchemy.ext.asyncio
from llama_index.core.bridge.pydantic import PrivateAttr, SecretStr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)


class NotSupportedError(Exception):
    """Raised when a feature is not supported on the current PolarDB-X version."""

    pass


class DBEmbeddingRow(NamedTuple):
    """Row returned from the database representing an embedding match."""

    node_id: str
    text: str
    metadata: dict
    similarity: float


# PolarDBXVectorStore __init__ parameters (for typo detection in **kwargs)
_OWN_PARAMS: set = {
    "host",
    "port",
    "user",
    "password",
    "database",
    "table_name",
    "embed_dim",
    "default_m",
    "distance_method",
    "perform_setup",
    "debug",
    "ef_construction",
    "vector_index_name",
    "ssl",
    "ssl_ca",
    "connection_retries",
    "retry_delay",
    "partition_by",
    "partitions",
    "partition_column",
    "broadcast",
    "locality",
    "partition_defs",
}

# pymysql / SQLAlchemy connect_args that users may pass via **kwargs
_KNOWN_KWARGS: set = {
    # SSL/TLS (pymysql individual params, excluding ssl_ca which
    # is a named __init__ parameter and never reaches **kwargs)
    "ssl_cert",
    "ssl_key",
    "ssl_verify_ca",
    "ssl_verify_identity",
    "ssl_disabled",
    # Connection behavior
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "charset",
    "collation",
    "autocommit",
    "client_flag",
    "compress",
    # Unix socket
    "unix_socket",
}


class PolarDBXVectorStore(BasePydanticVectorStore):
    """PolarDB-X Vector Store.

    A LlamaIndex vector store backed by PolarDB-X native vector index
    (HNSW) with dual-version compatibility.

    Requirements:
        - PolarDB-X with vector index enabled (``SET GLOBAL vidx_disabled = OFF``)
        - DN version >= 20260605

    Example:
        .. code-block:: python

            from llama_index.vector_stores.polardbx import PolarDBXVectorStore

            vector_store = PolarDBXVectorStore(
                host="your-polardbx-host",
                port=3306,
                user="your-user",
                password="your-password",
                database="your-database",
                table_name="my_vectors",
                embed_dim=1536,
                distance_method="COSINE",
            )
    """

    stores_text: bool = True
    flat_metadata: bool = False

    connection_string: SecretStr
    table_name: str = "llama_index_table"
    database: str
    embed_dim: int = 1536
    default_m: int = 6
    distance_method: Literal["EUCLIDEAN", "COSINE", "INNER_PRODUCT"] = "COSINE"
    perform_setup: bool = True
    debug: bool = False

    _engine: Any = PrivateAttr()
    _async_engine: Any = PrivateAttr()
    _session: Any = PrivateAttr()
    _async_session: Any = PrivateAttr()
    _is_initialized: bool = PrivateAttr(default=False)
    _capabilities: Dict[str, bool] = PrivateAttr(default_factory=dict)
    _vector_index_name: Optional[str] = PrivateAttr(default=None)
    _vector_index_checked: bool = PrivateAttr(default=False)
    _ef_construction: Optional[int] = PrivateAttr(default=None)
    _ssl: bool = PrivateAttr(default=False)
    _ssl_ca: Optional[str] = PrivateAttr(default=None)
    _conn_kwargs: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _connection_retries: int = PrivateAttr(default=3)
    _retry_delay: float = PrivateAttr(default=1.0)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    # ------------------------------------------------------------------
    # Security: prevent credential leakage in repr/dump
    # ------------------------------------------------------------------

    def __repr_args__(self):
        """Exclude connection_string from repr to prevent password leakage."""
        return [
            (k, v) for k, v in super().__repr_args__()
            if k != "connection_string"
        ]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_identifier(name: str) -> str:
        """Validate a SQL identifier (table name, index name, etc.)."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid identifier: {name}")
        return name

    @staticmethod
    def _validate_positive_int(value: int, param_name: str) -> int:
        """Validate that a value is a positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"Expected positive int for {param_name}, got {value}"
            )
        return value

    @staticmethod
    def _validate_metadata_key(key: str) -> str:
        """Validate metadata key for safe JSON path usage.

        Only alphanumeric characters, underscores, and dots are allowed
        to prevent JSON path injection in JSON_EXTRACT expressions.
        """
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", key):
            raise ValueError(
                f"Invalid metadata key: {key!r}. "
                "Only alphanumeric characters, underscores, "
                "and dots are allowed, starting with a letter "
                "or underscore."
            )
        return key

    # ------------------------------------------------------------------
    # Constructor & factory
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_kwargs(kwargs: Dict[str, Any]) -> None:
        """Validate **kwargs to catch common typos and invalid arguments.

        Checks each key against known PolarDBXVectorStore parameters and
        common pymysql/SQLAlchemy connection options.  If a key looks like
        a typo of a known parameter, raises ``TypeError`` with a suggestion.
        """
        for key in kwargs:
            if key.startswith("ssl_") or key in _KNOWN_KWARGS:
                continue
            matches = difflib.get_close_matches(key, _OWN_PARAMS, n=3, cutoff=0.6)
            if matches:
                raise TypeError(
                    f"PolarDBXVectorStore got an unexpected keyword "
                    f"argument '{key}'. Did you mean '{matches[0]}'?"
                )
            raise TypeError(
                f"PolarDBXVectorStore got an unexpected keyword "
                f"argument '{key}'. This is not a recognized "
                f"PolarDBXVectorStore parameter or pymysql connection "
                f"option. Valid parameters: see PolarDBXVectorStore "
                f"__init__ docstring."
            )

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        table_name: str = "llama_index_table",
        embed_dim: int = 1536,
        default_m: int = 6,
        distance_method: Literal["EUCLIDEAN", "COSINE", "INNER_PRODUCT"] = "COSINE",
        perform_setup: bool = True,
        debug: bool = False,
        ef_construction: Optional[int] = None,
        vector_index_name: Optional[str] = None,
        ssl: bool = False,
        ssl_ca: Optional[str] = None,
        connection_retries: int = 3,
        retry_delay: float = 1.0,
        partition_by: Optional[str] = None,
        partitions: int = 0,
        partition_column: Optional[str] = None,
        broadcast: bool = False,
        locality: Optional[str] = None,
        partition_defs: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PolarDB-X vector store.

        Args:
            host: PolarDB-X host address.
            port: PolarDB-X port number.
            user: Username.
            password: Password.
            database: Database name.
            table_name: Table name for vector storage.
                Defaults to ``"llama_index_table"``.
            embed_dim: Embedding dimension. Defaults to 1536.
            default_m: HNSW index M parameter (3-200). Defaults to 6.
            distance_method: Distance function — ``"COSINE"``,
                ``"EUCLIDEAN"``, or ``"INNER_PRODUCT"`` (v3 only).
                Defaults to ``"COSINE"``.
            perform_setup: If True, auto-create table on init.
                Defaults to True.
            debug: Enable SQLAlchemy echo mode. Defaults to False.
            ef_construction: HNSW build-time candidate list size
                (5-1000, v3 only). Ignored on old versions.
                Defaults to None (use DN default of 10).
            vector_index_name: Name of the vector index for FORCE INDEX
                hints. If None, auto-detected on first use.
            ssl: Enable TLS/SSL encryption. Defaults to False.
            ssl_ca: Path to CA certificate for SSL verification.
                Only effective when ``ssl=True``. Defaults to None.
            connection_retries: Number of connection retry attempts
                during initialization. Defaults to 3.
            retry_delay: Delay between retry attempts in seconds.
                Defaults to 1.0.
            partition_by: Partition strategy for the table. One of
                ``"HASH"``, ``"KEY"``, ``"RANGE"``, ``"LIST"``, or None.
                If None and broadcast is False, creates a single
                (non-partitioned) table. Defaults to None.
            partitions: Number of partitions. Required when
                partition_by is ``"HASH"`` or ``"KEY"``. Defaults to 0.
            partition_column: Column to partition on. Defaults to ``"id"``.
            broadcast: If True, creates a broadcast table (full copy on
                every DN node). Mutually exclusive with partition_by.
                Defaults to False.
            locality: Storage node specification, e.g. ``"dn=xxx"``.
                Appended to DDL as LOCALITY clause. Defaults to None.
            partition_defs: Partition definitions for RANGE/LIST
                strategies. Each dict has a ``"name"`` key and either
                ``"values_less_than"`` (RANGE) or ``"values_in"`` (LIST).
                Defaults to None.
        """
        self._validate_kwargs(kwargs)

        self._validate_identifier(table_name)
        self._validate_identifier(database)
        self._validate_positive_int(embed_dim, "embed_dim")
        self._validate_positive_int(default_m, "default_m")

        # S1: Validate ef_construction range to prevent SQL injection via DDL
        if ef_construction is not None:
            self._validate_positive_int(ef_construction, "ef_construction")
            if not (5 <= ef_construction <= 1000):
                raise ValueError(
                    f"ef_construction must be 5-1000, got {ef_construction}"
                )

        # S4: Validate connection_retries to prevent TypeError on raise None
        if not isinstance(connection_retries, int) or connection_retries < 1:
            raise ValueError(
                f"connection_retries must be a positive integer, "
                f"got {connection_retries}"
            )
        if not isinstance(retry_delay, (int, float)) or retry_delay < 0:
            raise ValueError(
                f"retry_delay must be a non-negative number, got {retry_delay}"
            )

        if distance_method not in ("EUCLIDEAN", "COSINE", "INNER_PRODUCT"):
            raise ValueError(
                f"Invalid distance_method: {distance_method}. "
                "Must be 'COSINE', 'EUCLIDEAN', or 'INNER_PRODUCT'."
            )

        # Validate partition params
        _pby = partition_by.upper() if partition_by else None
        if _pby and _pby not in ("HASH", "KEY", "RANGE", "LIST"):
            raise ValueError(
                f"Invalid partition_by: {partition_by}. "
                "Must be 'HASH', 'KEY', 'RANGE', or 'LIST'."
            )
        if _pby in ("HASH", "KEY") and partitions <= 0:
            raise ValueError(
                "partitions must be > 0 when partition_by is 'HASH' or 'KEY'."
            )
        if _pby in ("RANGE", "LIST") and not partition_defs:
            raise ValueError(
                "partition_defs must be provided when partition_by is "
                "'RANGE' or 'LIST'."
            )
        if broadcast and _pby:
            raise ValueError(
                "broadcast and partition_by are mutually exclusive. "
                "Use one or the other."
            )
        _pcolumn = partition_column or "id"
        if _pby and _pcolumn != "id":
            from llama_index.vector_stores.polardbx._partition import (
                _validate_identifier as _validate_id,
            )
            _validate_id(_pcolumn, "partition column")

        password_safe = quote_plus(password)
        user_safe = quote_plus(user)
        connection_string = (
            f"mysql+pymysql://{user_safe}:{password_safe}"
            f"@{host}:{port}/{database}"
        )

        super().__init__(
            connection_string=connection_string,
            table_name=table_name,
            database=database,
            embed_dim=embed_dim,
            default_m=default_m,
            distance_method=distance_method,
            perform_setup=perform_setup,
            debug=debug,
        )

        # Private attrs
        self._engine = None
        self._async_engine = None
        self._session = None
        self._async_session = None
        self._is_initialized = False
        self._capabilities = {}
        self._vector_index_name = vector_index_name
        self._vector_index_checked = False
        self._ef_construction = ef_construction
        self._ssl = ssl
        self._ssl_ca = ssl_ca
        self._conn_kwargs = kwargs
        self._connection_retries = connection_retries
        self._retry_delay = retry_delay
        self._partition_by = _pby
        self._partitions = partitions
        self._partition_column = _pcolumn
        self._broadcast = broadcast
        self._locality = locality
        self._partition_defs = partition_defs
        self._lock = threading.Lock()

        self._initialize()

    @classmethod
    def class_name(cls) -> str:
        return "PolarDBXVectorStore"

    @classmethod
    def from_params(
        cls,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        table_name: str = "llama_index_table",
        embed_dim: int = 1536,
        default_m: int = 6,
        distance_method: Literal["EUCLIDEAN", "COSINE", "INNER_PRODUCT"] = "COSINE",
        perform_setup: bool = True,
        debug: bool = False,
        ef_construction: Optional[int] = None,
        vector_index_name: Optional[str] = None,
        ssl: bool = False,
        ssl_ca: Optional[str] = None,
        connection_retries: int = 3,
        retry_delay: float = 1.0,
        partition_by: Optional[str] = None,
        partitions: int = 0,
        partition_column: Optional[str] = None,
        broadcast: bool = False,
        locality: Optional[str] = None,
        partition_defs: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "PolarDBXVectorStore":
        """Construct from parameters (factory method)."""
        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            table_name=table_name,
            embed_dim=embed_dim,
            default_m=default_m,
            distance_method=distance_method,
            perform_setup=perform_setup,
            debug=debug,
            ef_construction=ef_construction,
            vector_index_name=vector_index_name,
            ssl=ssl,
            ssl_ca=ssl_ca,
            connection_retries=connection_retries,
            retry_delay=retry_delay,
            partition_by=partition_by,
            partitions=partitions,
            partition_column=partition_column,
            broadcast=broadcast,
            locality=locality,
            partition_defs=partition_defs,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Connection & initialization
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Return the SQLAlchemy engine."""
        if not self._is_initialized:
            return None
        return self._engine

    def _connect(self) -> None:
        """Create SQLAlchemy engines and sessions."""
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            create_async_engine,
        )
        from sqlalchemy.orm import sessionmaker

        if self.debug:
            _logger.warning(
                "Debug mode enabled — SQL statements will be logged. "
                "Do NOT use in production."
            )

        # I6: Warn when SSL is disabled for production awareness
        if not self._ssl:
            _logger.warning(
                "SSL is disabled — database credentials will be "
                "transmitted in plaintext. Set ssl=True for production "
                "deployments."
            )

        connect_args: Dict[str, Any] = {}
        if self._ssl:
            connect_args["ssl"] = (
                {"ca": self._ssl_ca} if self._ssl_ca else True
            )
        # Merge extra connection kwargs (e.g. ssl_cert, ssl_key,
        # connect_timeout, read_timeout, etc.)
        connect_args.update(self._conn_kwargs)

        conn_str = self.connection_string.get_secret_value()
        self._engine = create_engine(
            conn_str,
            echo=self.debug,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            connect_args=connect_args,
        )

        async_conn_str = conn_str.replace(
            "mysql+pymysql://", "mysql+aiomysql://"
        )
        self._async_engine = create_async_engine(
            async_conn_str,
            echo=self.debug,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            connect_args=connect_args,
        )

        self._session = sessionmaker(self._engine)
        self._async_session = sessionmaker(
            self._async_engine, class_=AsyncSession
        )

    def _initialize(self) -> None:
        """Connect, detect capabilities, and optionally create table.

        Uses double-checked locking to ensure thread safety during
        initialization.  Capability detection always runs (independent
        of ``perform_setup``); only table creation is gated.

        If the database is temporarily unreachable, retries up to
        ``connection_retries`` times with ``retry_delay`` seconds
        between attempts.  Non-transient errors (``ValueError``,
        ``NotSupportedError``) are raised immediately without retry.
        """
        if not self._is_initialized:
            with self._lock:
                if not self._is_initialized:
                    last_exc: Optional[Exception] = None
                    for attempt in range(self._connection_retries):
                        try:
                            self._connect()
                            # Capability detection is independent of
                            # table creation — always probe so v3
                            # features work even when
                            # perform_setup=False.
                            self._detect_capabilities()
                            self._validate_distance_method()
                            if self.perform_setup:
                                self._create_table_if_not_exists()
                            self._is_initialized = True
                            return
                        except (ValueError, NotSupportedError):
                            # Configuration errors — retrying won't help.
                            raise
                        except Exception as e:
                            last_exc = e
                            # Clean up partially created engines.
                            self._cleanup_engines()
                            if attempt < self._connection_retries - 1:
                                _logger.warning(
                                    "Initialization attempt %d/%d "
                                    "failed: %s. Retrying in %.1fs...",
                                    attempt + 1,
                                    self._connection_retries,
                                    self._sanitize_error(e),
                                    self._retry_delay,
                                )
                                time.sleep(self._retry_delay)
                            else:
                                _logger.error(
                                    "All %d initialization attempts "
                                    "failed. Last error: %s",
                                    self._connection_retries,
                                    self._sanitize_error(e),
                                )
                    # S4: Guard against last_exc being None
                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError(
                        "Initialization failed with no exception "
                        "captured"
                    )

    def _cleanup_engines(self) -> None:
        """Dispose engines and reset session factories.

        Called between retry attempts to avoid stale connection pools.
        """
        if self._engine is not None:
            try:
                self._engine.dispose()
            except Exception:
                pass
            self._engine = None
        if self._async_engine is not None:
            try:
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as ex:
                        future = ex.submit(
                            asyncio.run,
                            self._async_engine.dispose(),
                        )
                        future.result()
                except RuntimeError:
                    try:
                        asyncio.run(self._async_engine.dispose())
                    except RuntimeError:
                        pass
            except Exception:
                pass
            self._async_engine = None
        self._session = None
        self._async_session = None

    # ------------------------------------------------------------------
    # Capability detection (v3 dual-version support)
    # ------------------------------------------------------------------

    def _detect_capabilities(self) -> None:
        """Detect PolarDB-X vector capabilities at init time.

        Probes the database for vector feature availability and caches
        results in ``self._capabilities`` for later use.

        Detected capabilities:
            - vec_distance: VEC_DISTANCE() auto-inference function
            - vec_totext: VEC_TOTEXT() function
            - vec_dim: VECTOR_DIM() function (v3 indicator)
            - vector_indexes_view: information_schema.VECTOR_INDEXES

        Raises:
            ValueError: If vector index is disabled or vector functions
                are not available.
        """
        from sqlalchemy import text

        with self._session() as session:
            try:
                # Check if vector index is disabled via system variable
                result = session.execute(
                    text("SHOW GLOBAL VARIABLES LIKE 'vidx_disabled'")
                )
                row = result.fetchone()
                if row and str(row[1]).upper() == "ON":
                    raise ValueError(
                        "PolarDB-X vector index is disabled. "
                        "Please execute SET GLOBAL vidx_disabled = OFF "
                        "and reconnect."
                    )

                # Verify vector functions are available
                result = session.execute(
                    text(
                        "SELECT VEC_FROMTEXT('[1,2,3]') "
                        "IS NOT NULL as vector_support"
                    )
                )
                vector_result = result.fetchone()
                if not vector_result or not vector_result[0]:
                    raise ValueError(
                        "PolarDB-X vector functions are not available. "
                        "Please verify the DN version is >= 20260605 "
                        "and vector index support is enabled."
                    )

            except ValueError:
                raise
            except Exception as e:
                if "FUNCTION" in str(e) and "VEC_FROMTEXT" in str(e):
                    raise ValueError(
                        "PolarDB-X vector functions are not available. "
                        "Please verify the DN version is >= 20260605 "
                        "and vector index support is enabled."
                    ) from e
                raise

        # Probe extended capabilities (non-fatal — default to False)
        # vec_dim (VECTOR_DIM) serves as the v3 indicator: it is present
        # iff the instance has v3 vector features (EF_CONSTRUCTION DDL,
        # INNER_PRODUCT, dbms_vidx procedures, etc.).
        caps = {
            "vec_distance": self._probe_vec_distance(),
            "vec_totext": self._probe_function(
                "SELECT VEC_TOTEXT(VEC_FROMTEXT('[1,2,3]')) IS NOT NULL"
            ),
            "vec_dim": self._probe_function(
                "SELECT VECTOR_DIM(VEC_FROMTEXT('[1,2,3]')) IS NOT NULL"
            ),
        }
        caps["vector_indexes_view"] = self._probe_table_exists(
            "information_schema", "VECTOR_INDEXES"
        )
        self._capabilities = caps
        _logger.info("Detected capabilities: %s", self._capabilities)

    def _validate_distance_method(self) -> None:
        """Validate INNER_PRODUCT requires v3 support."""
        if (
            self.distance_method == "INNER_PRODUCT"
            and not self._capabilities.get("vec_dim", False)
        ):
            raise NotSupportedError(
                "distance_method='INNER_PRODUCT' requires PolarDB-X v3. "
                "Use 'COSINE' or 'EUCLIDEAN' for old versions."
            )

    def _probe_vec_distance(self) -> bool:
        """Probe whether VEC_DISTANCE() is available.

        VEC_DISTANCE requires a vector-index context to infer the distance
        metric, so a standalone SELECT fails on DN even though the function
        exists. We inspect the error message to distinguish "exists" from
        "not found".
        """
        from sqlalchemy import text

        sql = text(
            "SELECT VEC_DISTANCE(VEC_FROMTEXT('[1,2,3]'),"
            " VEC_FROMTEXT('[1,2,3]')) IS NOT NULL"
        )
        try:
            with self._session() as session:
                result = session.execute(sql)
                return result.fetchone() is not None
        except Exception as e:
            err_msg = str(e).upper()
            # v3: VEC_DISTANCE without index context reports
            # ER_VEC_DISTANCE_TYPE, or messages like "NO VECTOR INDEX",
            # "CANNOT DETERMINE". All indicate the function EXISTS.
            if (
                "NO VECTOR INDEX" in err_msg
                or "CANNOT DETERMINE" in err_msg
                or "VEC_DISTANCE_TYPE" in err_msg
                or "ER_VEC_DISTANCE" in err_msg
            ):
                _logger.debug(
                    "VEC_DISTANCE exists but needs index context: %s", e
                )
                return True
            _logger.debug("VEC_DISTANCE probe failed: %s", e)
            return False

    def _probe_function(self, sql: str) -> bool:
        """Probe whether a SQL function is available."""
        from sqlalchemy import text

        try:
            with self._session() as session:
                result = session.execute(text(sql))
                return result.fetchone() is not None
        except Exception as e:
            _logger.debug("Function probe failed [%s]: %s", sql, e)
            return False

    def _probe_table_exists(self, schema: str, table: str) -> bool:
        """Check if a table/view exists in information_schema."""
        from sqlalchemy import text

        try:
            with self._session() as session:
                result = session.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_schema = :schema AND table_name = :table"
                    ),
                    {"schema": schema, "table": table},
                )
                row = result.fetchone()
                return row[0] > 0 if row else False
        except Exception as e:
            _logger.debug(
                "Table probe failed [%s.%s]: %s", schema, table, e
            )
            return False

    def _get_distance_func(self) -> str:
        """Return the optimal distance function for the current instance.

        Prefers VEC_DISTANCE (auto-inference, v3) when available;
        falls back to explicit VEC_DISTANCE_COSINE / _EUCLIDEAN /
        _INNER_PRODUCT otherwise.
        """
        if self._capabilities.get("vec_distance", False):
            return "VEC_DISTANCE"
        if self.distance_method == "COSINE":
            return "VEC_DISTANCE_COSINE"
        if self.distance_method == "INNER_PRODUCT":
            return "VEC_DISTANCE_INNER_PRODUCT"
        return "VEC_DISTANCE_EUCLIDEAN"

    # ------------------------------------------------------------------
    # Table setup
    # ------------------------------------------------------------------

    def _build_partition_clause(self) -> str:
        """Build the PARTITION/BROADCAST/LOCALITY clause for CREATE TABLE.

        Returns an empty string for a single (non-partitioned) table.
        Delegates to _partition._build_partition_clause() to keep a
        single source of truth for partition clause generation.
        """
        from llama_index.vector_stores.polardbx._partition import (
            _build_partition_clause as _build,
        )

        return _build(
            partition_by=self._partition_by,
            partition_column=self._partition_column,
            partitions=self._partitions,
            broadcast=self._broadcast,
            locality=self._locality,
            partition_defs=self._partition_defs,
        )

    @property
    def _has_partition(self) -> bool:
        """Return True if the table uses partitioning or broadcast."""
        return self._broadcast or self._partition_by is not None

    def _create_table_if_not_exists(self) -> None:
        """Create the vector table if it does not exist."""
        from sqlalchemy import text

        # Build optional EF_CONSTRUCTION clause (v3 only)
        ef_clause = ""
        if (
            self._ef_construction is not None
            and self._capabilities.get("vec_dim", False)
        ):
            ef_clause = f" EF_CONSTRUCTION={self._ef_construction}"

        # When partitioning is enabled, PolarDB-X requires that unique
        # indexes include the partition key. The node_id unique index does
        # not include the partition key (default: id), so downgrade it
        # to a regular INDEX. The id PRIMARY KEY already guarantees row
        # uniqueness, and LlamaIndex handles dedup at the app level.
        node_index_type = (
            "INDEX" if self._has_partition else "UNIQUE INDEX"
        )

        partition_clause = self._build_partition_clause()
        # Strip trailing semicolon — partition clause goes before it
        base_ddl = f"""
        CREATE TABLE IF NOT EXISTS `{self.table_name}` (
            id VARCHAR(36) PRIMARY KEY,
            node_id VARCHAR(255) NOT NULL,
            text LONGTEXT,
            metadata JSON,
            embedding VECTOR({self.embed_dim}) NOT NULL,
            {node_index_type} `node_id_index` (node_id),
            VECTOR INDEX `vi` (embedding) M={self.default_m}{ef_clause} DISTANCE={self.distance_method}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        stmt = text(base_ddl + partition_clause + ";")
        with self._session() as session:
            try:
                session.execute(stmt)
                session.commit()
            except Exception as e:
                if self._has_partition:
                    err_msg = str(e).lower()
                    if "partition" in err_msg and (
                        "not support" in err_msg
                        or "do not support" in err_msg
                    ):
                        raise NotSupportedError(
                            "PolarDB-X vector index on partitioned tables is "
                            "not supported on this instance. This may occur "
                            "on older v3 DN versions. Try upgrading the DN "
                            "version, or remove partition parameters "
                            "(partition_by, broadcast, etc.) to create a "
                            "non-partitioned vector table."
                        ) from e
                raise

    def _node_to_table_row(self, node: BaseNode) -> Dict[str, Any]:
        """Convert a LlamaIndex node to a table row dict."""
        return {
            "node_id": node.node_id,
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "embedding": node.get_embedding(),
            "metadata": node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            ),
        }

    def _validate_embedding_dimensions(
        self, nodes: Sequence[BaseNode]
    ) -> None:
        """Validate that all node embeddings match the expected dimension.

        Checks each node's embedding length against ``self.embed_dim``.
        On v3 instances, additionally cross-checks with DN's
        ``VECTOR_DIM`` to catch client/server dimension mismatches.

        Args:
            nodes: List of nodes whose embeddings to validate.

        Raises:
            ValueError: If any embedding has a mismatched dimension.
        """
        for i, node in enumerate(nodes):
            emb = node.embedding
            if emb is None:
                raise ValueError(
                    f"Node at index {i} (id={node.node_id}) has no "
                    f"embedding. Call the embedding model first."
                )
            if len(emb) != self.embed_dim:
                raise ValueError(
                    f"Embedding at index {i} (id={node.node_id}) has "
                    f"dimension {len(emb)}, expected {self.embed_dim}."
                )

        # Cross-check with DN's VECTOR_DIM if available (v3 only)
        if self._capabilities.get("vec_dim", False) and nodes:
            from sqlalchemy import text

            sample_emb = json.dumps(nodes[0].embedding)
            try:
                with self._session() as session:
                    result = session.execute(
                        text(
                            "SELECT VECTOR_DIM(VEC_FROMTEXT(:emb)) AS dim"
                        ),
                        {"emb": sample_emb},
                    )
                    row = result.fetchone()
                    if row and row[0] != self.embed_dim:
                        raise ValueError(
                            f"DN VECTOR_DIM reports {row[0]}, "
                            f"but client expected {self.embed_dim}."
                        )
            except ValueError:
                raise
            except Exception as e:
                _logger.debug(
                    "VECTOR_DIM cross-check failed: %s", e
                )

    # ------------------------------------------------------------------
    # Metadata filter helpers
    # ------------------------------------------------------------------

    def _to_mysql_operator(self, operator: FilterOperator) -> str:
        """Map LlamaIndex FilterOperator to SQL operator."""
        mapping = {
            FilterOperator.EQ: "=",
            FilterOperator.GT: ">",
            FilterOperator.LT: "<",
            FilterOperator.NE: "!=",
            FilterOperator.GTE: ">=",
            FilterOperator.LTE: "<=",
            FilterOperator.IN: "IN",
            FilterOperator.NIN: "NOT IN",
        }
        if operator not in mapping:
            raise ValueError(
                f"Unsupported filter operator: {operator}. "
                "Supported operators: EQ, NE, GT, GTE, LT, LTE, IN, NIN."
            )
        return mapping[operator]

    def _build_filter_clause(
        self, filter_: MetadataFilter, global_param_counter: List[int]
    ) -> tuple[str, dict]:
        """Build a single filter clause.

        Uses JSON_UNQUOTE(JSON_EXTRACT(...)) instead of JSON_VALUE
        because PolarDB-X does not support the JSON_VALUE function.
        """
        self._validate_metadata_key(filter_.key)
        params: Dict[str, Any] = {}

        if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
            if not filter_.value:
                raise ValueError(
                    f"Filter '{filter_.key}' uses {filter_.operator} "
                    "with an empty list, which would generate "
                    "invalid SQL (IN ())."
                )
            placeholders = []
            for i in range(len(filter_.value)):
                param_name = f"param_{global_param_counter[0]}"
                global_param_counter[0] += 1
                placeholders.append(f":{param_name}")
                params[param_name] = filter_.value[i]
            filter_value = f"({','.join(placeholders)})"
        elif isinstance(filter_.value, (list, tuple)):
            # I3: Non-IN/NIN operators with list values generate invalid SQL
            raise ValueError(
                f"Filter '{filter_.key}' uses operator "
                f"{filter_.operator} with a list value. "
                f"List values are only supported for IN and "
                f"NIN operators."
            )
        else:
            param_name = f"param_{global_param_counter[0]}"
            global_param_counter[0] += 1
            filter_value = f":{param_name}"
            params[param_name] = filter_.value

        # PolarDB-X: JSON_UNQUOTE(JSON_EXTRACT(...)) instead of JSON_VALUE
        clause = (
            f"JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.{filter_.key}')) "
            f"{self._to_mysql_operator(filter_.operator)} {filter_value}"
        )
        return clause, params

    def _filters_to_where_clause(
        self, filters: MetadataFilters, global_param_counter: List[int]
    ) -> tuple[str, dict]:
        """Build a WHERE clause from MetadataFilters."""
        conditions = {
            FilterCondition.OR: "OR",
            FilterCondition.AND: "AND",
        }
        if filters.condition not in conditions:
            raise ValueError(
                f"Unsupported condition: {filters.condition}. "
                f"Must be one of {list(conditions.keys())}"
            )

        clauses: List[str] = []
        all_params: Dict[str, Any] = {}

        for filter_ in filters.filters:
            if isinstance(filter_, MetadataFilter):
                clause, filter_params = self._build_filter_clause(
                    filter_, global_param_counter
                )
                clauses.append(clause)
                all_params.update(filter_params)
                continue

            if isinstance(filter_, MetadataFilters):
                subclause, subparams = self._filters_to_where_clause(
                    filter_, global_param_counter
                )
                if subclause:
                    clauses.append(f"({subclause})")
                    all_params.update(subparams)
                continue

            raise ValueError(
                f"Unsupported filter type: {type(filter_)}. "
                f"Must be {MetadataFilter} or {MetadataFilters}"
            )

        return f" {conditions[filters.condition]} ".join(clauses), all_params

    def _db_rows_to_query_result(
        self, rows: List[DBEmbeddingRow]
    ) -> VectorStoreQueryResult:
        """Convert DB rows to a VectorStoreQueryResult."""
        nodes = []
        similarities = []
        ids = []
        for db_row in rows:
            node = metadata_dict_to_node(db_row.metadata)
            node.set_content(str(db_row.text))
            similarities.append(db_row.similarity)
            ids.append(db_row.node_id)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    # ------------------------------------------------------------------
    # Distance-to-similarity conversion & metadata parsing
    # ------------------------------------------------------------------

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert a raw distance value to a similarity score.

        The conversion depends on ``distance_method``:
        - COSINE: ``1 - distance``  (range [-1, 1])
        - EUCLIDEAN: ``1 / (1 + distance)``  (range (0, 1])
        - INNER_PRODUCT: ``-distance``  (distance is -dot_product)
        """
        if self.distance_method == "COSINE":
            return 1.0 - distance
        elif self.distance_method == "EUCLIDEAN":
            return 1.0 / (1.0 + max(0.0, distance))
        elif self.distance_method == "INNER_PRODUCT":
            return -distance
        return 1.0 - distance

    @staticmethod
    def _parse_metadata(raw: Any) -> dict:
        """Parse metadata from a DB row value.

        Handles JSON strings, dicts, and corrupted values gracefully.
        """
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                _logger.warning(
                    "Corrupted metadata JSON in database row, "
                    "using empty dict. Raw: %s",
                    raw[:200],
                )
                return {}
        return {}

    @staticmethod
    def _sanitize_error(e: Exception) -> str:
        """Sanitize exception message to remove embedded credentials.

        SQLAlchemy ``OperationalError`` messages may contain the full
        DSN (including password).  This method redacts the password
        portion of any ``://user:password@host`` pattern.
        """
        msg = str(e)
        msg = re.sub(
            r"://([^:/]+):([^@]+)@",
            r"://\1:***@",
            msg,
        )
        return msg

    # ------------------------------------------------------------------
    # MMR (Maximal Marginal Relevance) support
    # ------------------------------------------------------------------

    @staticmethod
    def _maximal_marginal_relevance(
        query_embedding: List[float],
        embedding_list: List[List[float]],
        k: int = 4,
        lambda_mult: float = 0.5,
    ) -> List[int]:
        """Calculate maximal marginal relevance.

        Selects ``k`` embeddings that balance similarity to the query
        with diversity among selected results.

        Args:
            query_embedding: Query embedding vector.
            embedding_list: List of candidate document embeddings.
            k: Number of embeddings to select.
            lambda_mult: Diversity factor (0 = max diversity,
                1 = min diversity). Defaults to 0.5.

        Returns:
            List of selected indices into ``embedding_list``.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "numpy is required for MMR search. "
                "Please install it with `pip install numpy`."
            ) from e

        if not embedding_list:
            return []

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(embedding_list)

        # I1: Guard against zero-norm vectors causing division by zero
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        if query_norm == 0 or np.any(doc_norms == 0):
            _logger.warning(
                "Zero-norm embedding detected in MMR; "
                "falling back to simple top-k ordering."
            )
            return list(range(min(k, len(embedding_list))))

        # Cosine similarity to query
        query_doc_sim = np.dot(doc_vecs, query_vec) / (
            doc_norms * query_norm
        )

        # Start with the most similar document
        selected = [int(np.argmax(query_doc_sim))]
        candidates = list(range(len(embedding_list)))
        candidates.remove(selected[0])

        while len(selected) < min(k, len(embedding_list)) and candidates:
            best_score = -float("inf")
            best_idx = -1

            for idx in candidates:
                max_sim_to_selected = max(
                    np.dot(doc_vecs[idx], doc_vecs[sel_idx])
                    / (
                        np.linalg.norm(doc_vecs[idx])
                        * np.linalg.norm(doc_vecs[sel_idx])
                    )
                    for sel_idx in selected
                )

                mmr_score = (
                    lambda_mult * query_doc_sim[idx]
                    - (1 - lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx != -1:
                selected.append(best_idx)
                candidates.remove(best_idx)

        return selected

    def _fetch_embeddings_by_node_ids(
        self, node_ids: List[str]
    ) -> Dict[str, List[float]]:
        """Fetch stored embedding vectors by node_id.

        Uses ``VEC_TOTEXT`` on v3 instances, ``CAST(embedding AS CHAR)``
        on old versions.

        Args:
            node_ids: List of node_id values to fetch.

        Returns:
            Dict mapping node_id to its embedding vector.
        """
        if not node_ids:
            return {}

        emb_expr = (
            "VEC_TOTEXT(embedding)"
            if self._capabilities.get("vec_totext", False)
            else "CAST(embedding AS CHAR)"
        )

        # Chunk to avoid SQL length limits
        result: Dict[str, List[float]] = {}
        chunk_size = 1000
        for start in range(0, len(node_ids), chunk_size):
            chunk = node_ids[start : start + chunk_size]
            placeholders = ",".join(
                [f":nid_{i}" for i in range(len(chunk))]
            )
            params = {f"nid_{i}": nid for i, nid in enumerate(chunk)}
            stmt = sqlalchemy.text(
                f"SELECT node_id, {emb_expr} AS emb_str "
                f"FROM `{self.table_name}` "
                f"WHERE node_id IN ({placeholders})"
            )
            try:
                with self._session() as session:
                    rows = session.execute(stmt, params).fetchall()
                    for row in rows:
                        emb_str = row[1]
                        if isinstance(emb_str, str) and emb_str:
                            try:
                                result[row[0]] = json.loads(emb_str)
                            except (json.JSONDecodeError, ValueError):
                                pass
            except Exception as e:
                # I15: Use WARNING instead of DEBUG to surface failures
                _logger.warning("Failed to fetch embeddings: %s", e)
        return result

    async def _afetch_embeddings_by_node_ids(
        self, node_ids: List[str]
    ) -> Dict[str, List[float]]:
        """Async fetch stored embedding vectors by node_id."""
        if not node_ids:
            return {}

        emb_expr = (
            "VEC_TOTEXT(embedding)"
            if self._capabilities.get("vec_totext", False)
            else "CAST(embedding AS CHAR)"
        )

        result: Dict[str, List[float]] = {}
        chunk_size = 1000
        for start in range(0, len(node_ids), chunk_size):
            chunk = node_ids[start : start + chunk_size]
            placeholders = ",".join(
                [f":nid_{i}" for i in range(len(chunk))]
            )
            params = {f"nid_{i}": nid for i, nid in enumerate(chunk)}
            stmt = sqlalchemy.text(
                f"SELECT node_id, {emb_expr} AS emb_str "
                f"FROM `{self.table_name}` "
                f"WHERE node_id IN ({placeholders})"
            )
            try:
                async with self._async_session() as session:
                    rows = (
                        await session.execute(stmt, params)
                    ).fetchall()
                    for row in rows:
                        emb_str = row[1]
                        if isinstance(emb_str, str) and emb_str:
                            try:
                                result[row[0]] = json.loads(emb_str)
                            except (json.JSONDecodeError, ValueError):
                                pass
            except Exception as e:
                # I15: Use WARNING instead of DEBUG to surface failures
                _logger.warning("Failed to fetch embeddings: %s", e)
        return result

    # ------------------------------------------------------------------
    # Index hint & ef_search helpers
    # ------------------------------------------------------------------

    def _detect_vector_index_name(self) -> Optional[str]:
        """Auto-detect the vector index name on the table.

        v3 path: query information_schema.VECTOR_INDEXES.
        Fallback: parse SHOW CREATE TABLE with regex.

        Caches the result (including "not found") via
        ``_vector_index_checked`` to avoid repeated DB probes.
        """
        if self._vector_index_name is not None:
            return self._vector_index_name
        if self._vector_index_checked:
            return None

        from sqlalchemy import text

        # v3: use information_schema.VECTOR_INDEXES view (preferred)
        if self._capabilities.get("vector_indexes_view", False):
            try:
                with self._session() as session:
                    result = session.execute(
                        text(
                            "SELECT INDEX_NAME FROM "
                            "information_schema.VECTOR_INDEXES "
                            "WHERE TABLE_SCHEMA = :schema "
                            "AND TABLE_NAME = :table LIMIT 1"
                        ),
                        {"schema": self.database, "table": self.table_name},
                    )
                    row = result.fetchone()
                    if row and row[0]:
                        self._vector_index_name = row[0]
                        return self._vector_index_name
            except Exception as e:
                _logger.debug("VECTOR_INDEXES query failed: %s", e)

        # Fallback: parse SHOW CREATE TABLE with regex
        try:
            with self._session() as session:
                result = session.execute(
                    text(f"SHOW CREATE TABLE `{self.table_name}`")
                )
                row = result.fetchone()
                if row:
                    create_sql = str(row[1]) if len(row) > 1 else ""
                    m = re.search(
                        r"VECTOR INDEX\s+`?(\w+)`?", create_sql,
                        re.IGNORECASE
                    )
                    if m:
                        self._vector_index_name = m.group(1)
                        return self._vector_index_name
        except Exception as e:
            _logger.debug("Failed to detect vector index name: %s", e)
        self._vector_index_checked = True
        return None

    def _build_index_hint(self, search_type: Optional[str]) -> str:
        """Build index hint string for search_type.

        - knn: FORCE INDEX(PRIMARY) to force full table scan
        - ann: FORCE INDEX(vector_index) to force vector index usage
        - auto/None: no hint, let optimizer decide
        """
        if search_type is None or search_type == "auto":
            return ""
        if search_type == "knn":
            return " FORCE INDEX(PRIMARY)"
        if search_type == "ann":
            idx_name = self._detect_vector_index_name()
            if idx_name:
                return f" FORCE INDEX(`{idx_name}`)"
            return ""
        return ""

    def _set_ef_search_sync(self, session: Any, ef_search: Optional[int]) -> None:
        """Set ef_search session variable (sync)."""
        if ef_search is not None:
            # I14: Validate ef_search range (1-10000)
            ef_val = int(ef_search)
            if not (1 <= ef_val <= 10000):
                raise ValueError(
                    f"ef_search must be 1-10000, got {ef_val}"
                )
            session.execute(
                sqlalchemy.text(
                    f"SET SESSION vidx_hnsw_ef_search = {ef_val}"
                )
            )

    async def _set_ef_search_async(
        self, session: Any, ef_search: Optional[int]
    ) -> None:
        """Set ef_search session variable (async)."""
        if ef_search is not None:
            # I14: Validate ef_search range (1-10000)
            ef_val = int(ef_search)
            if not (1 <= ef_val <= 10000):
                raise ValueError(
                    f"ef_search must be 1-10000, got {ef_val}"
                )
            await session.execute(
                sqlalchemy.text(
                    f"SET SESSION vidx_hnsw_ef_search = {ef_val}"
                )
            )

    # ------------------------------------------------------------------
    # CRUD: add
    # ------------------------------------------------------------------

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to the vector store.

        Uses UPSERT (ON DUPLICATE KEY UPDATE) on the ``node_id``
        unique index so re-adding a node updates rather than duplicates.

        Args:
            nodes: List of nodes to add.
            batch_size: Number of nodes per transaction commit.
                Defaults to 500.  Larger values may improve throughput
                but increase memory usage and lock duration.
        """
        self._initialize()

        if not nodes:
            return []

        self._validate_embedding_dimensions(nodes)

        batch_size = add_kwargs.get("batch_size", 500)
        # I13: Validate batch_size to prevent ValueError from range()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size}"
            )
        ids: List[str] = []
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            with self._session() as session:
                for node in batch:
                    ids.append(node.node_id)
                    item = self._node_to_table_row(node)

                    stmt = sqlalchemy.text(f"""
                    INSERT INTO `{self.table_name}` (id, node_id, text, embedding, metadata)
                    VALUES (
                        UUID(),
                        :node_id,
                        :text,
                        VEC_FROMTEXT(:embedding),
                        :metadata
                    )
                    ON DUPLICATE KEY UPDATE
                        text = VALUES(text),
                        embedding = VALUES(embedding),
                        metadata = VALUES(metadata)
                    """)
                    session.execute(
                        stmt,
                        {
                            "node_id": item["node_id"],
                            "text": item["text"],
                            "embedding": json.dumps(item["embedding"]),
                            "metadata": json.dumps(item["metadata"]),
                        },
                    )
                session.commit()
        return ids

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Async add nodes to the vector store.

        Args:
            nodes: Sequence of nodes to add.
            batch_size: Number of nodes per transaction commit.
                Defaults to 500.
        """
        self._initialize()

        if not nodes:
            return []

        self._validate_embedding_dimensions(nodes)

        batch_size = kwargs.get("batch_size", 500)
        # I13: Validate batch_size to prevent ValueError from range()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size}"
            )
        ids: List[str] = []
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            async with self._async_session() as session:
                for node in batch:
                    ids.append(node.node_id)
                    item = self._node_to_table_row(node)

                    stmt = sqlalchemy.text(f"""
                    INSERT INTO `{self.table_name}` (id, node_id, text, embedding, metadata)
                    VALUES (
                        UUID(),
                        :node_id,
                        :text,
                        VEC_FROMTEXT(:embedding),
                        :metadata
                    )
                    ON DUPLICATE KEY UPDATE
                        text = VALUES(text),
                        embedding = VALUES(embedding),
                        metadata = VALUES(metadata)
                    """)
                    await session.execute(
                        stmt,
                        {
                            "node_id": item["node_id"],
                            "text": item["text"],
                            "embedding": json.dumps(item["embedding"]),
                            "metadata": json.dumps(item["metadata"]),
                        },
                    )
                await session.commit()
        return ids

    # ------------------------------------------------------------------
    # CRUD: query
    # ------------------------------------------------------------------

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store for similar nodes.

        Keyword Args:
            ef_search: HNSW search candidate list size (1-10000).
                Higher = more accurate but slower.
            search_type: ``"ann"`` (force index), ``"knn"`` (force scan),
                or ``"auto"`` (optimizer decides). Defaults to ``"auto"``.
            fetch_k: Number of candidates to fetch before MMR re-ranking.
                Only used when ``query.mode`` is ``MMR``. Defaults to
                ``max(similarity_top_k * 3, 20)``.
            lambda_mult: MMR diversity factor (0 = max diversity,
                1 = min diversity). Only used when ``query.mode`` is ``MMR``.
                Defaults to 0.5.
        """
        if query.mode not in (
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.MMR,
        ):
            raise NotImplementedError(
                f"Query mode {query.mode} not available."
            )

        if query.query_embedding is None:
            raise ValueError(
                "query.query_embedding is None. Ensure the embedding "
                "model is properly configured (e.g. "
                "Settings.embed_model = ...)."
            )

        self._initialize()

        # MMR: fetch more candidates than needed
        is_mmr = query.mode == VectorStoreQueryMode.MMR
        if is_mmr:
            fetch_k = kwargs.get(
                "fetch_k",
                max(query.similarity_top_k * 3, 20),
            )
        else:
            fetch_k = query.similarity_top_k

        ef_search = kwargs.get("ef_search")
        search_type = kwargs.get("search_type")

        distance_func = self._get_distance_func()
        index_hint = self._build_index_hint(search_type)

        where_clause = ""
        params: Dict[str, Any] = {
            "query_embedding": json.dumps(query.query_embedding),
            "limit": fetch_k,
        }

        if query.filters:
            global_param_counter = [0]
            where_clause, filter_params = self._filters_to_where_clause(
                query.filters, global_param_counter
            )
            where_clause = f"WHERE {where_clause}"
            params.update(filter_params)

        stmt = sqlalchemy.text(f"""
        SELECT
            node_id,
            text,
            metadata,
            {distance_func}(embedding, VEC_FROMTEXT(:query_embedding)) AS distance
        FROM `{self.table_name}`{index_hint}
        {where_clause}
        ORDER BY distance
        LIMIT :limit
        """)

        with self._session() as session:
            self._set_ef_search_sync(session, ef_search)
            result = session.execute(stmt, params)
            results = result.fetchall()

        rows = []
        for item in results:
            meta = self._parse_metadata(item[2])
            rows.append(
                DBEmbeddingRow(
                    node_id=item[0],
                    text=item[1],
                    metadata=meta,
                    similarity=self._distance_to_similarity(item[3])
                    if item[3] is not None
                    else 0.0,
                )
            )

        # MMR re-ranking
        if is_mmr and len(rows) > query.similarity_top_k:
            lambda_mult = kwargs.get("lambda_mult", 0.5)
            node_ids = [r.node_id for r in rows]
            emb_map = self._fetch_embeddings_by_node_ids(node_ids)
            embedding_list = [
                emb_map.get(nid, []) for nid in node_ids
            ]
            # Only apply MMR if we got embeddings
            if all(len(e) > 0 for e in embedding_list):
                selected = self._maximal_marginal_relevance(
                    query_embedding=query.query_embedding,
                    embedding_list=embedding_list,
                    k=query.similarity_top_k,
                    lambda_mult=lambda_mult,
                )
                rows = [rows[i] for i in selected]
            else:
                # I2: Warn when MMR re-ranking is skipped
                _logger.warning(
                    "MMR re-ranking skipped: could not fetch "
                    "embeddings for %d/%d candidate rows. "
                    "Returning distance-ordered results.",
                    sum(1 for e in embedding_list if len(e) == 0),
                    len(embedding_list),
                )

        return self._db_rows_to_query_result(rows)

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Async query the vector store.

        Keyword Args:
            ef_search: HNSW search candidate list size.
            search_type: ``"ann"``, ``"knn"``, or ``"auto"``.
            fetch_k: Candidates to fetch before MMR re-ranking
                (MMR mode only). Defaults to
                ``max(similarity_top_k * 3, 20)``.
            lambda_mult: MMR diversity factor. Defaults to 0.5.
        """
        if query.mode not in (
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.MMR,
        ):
            raise NotImplementedError(
                f"Query mode {query.mode} not available."
            )

        if query.query_embedding is None:
            raise ValueError(
                "query.query_embedding is None. Ensure the embedding "
                "model is properly configured (e.g. "
                "Settings.embed_model = ...)."
            )

        self._initialize()

        is_mmr = query.mode == VectorStoreQueryMode.MMR
        if is_mmr:
            fetch_k = kwargs.get(
                "fetch_k",
                max(query.similarity_top_k * 3, 20),
            )
        else:
            fetch_k = query.similarity_top_k

        ef_search = kwargs.get("ef_search")
        search_type = kwargs.get("search_type")

        distance_func = self._get_distance_func()
        index_hint = self._build_index_hint(search_type)

        where_clause = ""
        params: Dict[str, Any] = {
            "query_embedding": json.dumps(query.query_embedding),
            "limit": fetch_k,
        }

        if query.filters:
            global_param_counter = [0]
            where_clause, filter_params = self._filters_to_where_clause(
                query.filters, global_param_counter
            )
            where_clause = f"WHERE {where_clause}"
            params.update(filter_params)

        stmt = sqlalchemy.text(f"""
        SELECT
            node_id,
            text,
            metadata,
            {distance_func}(embedding, VEC_FROMTEXT(:query_embedding)) AS distance
        FROM `{self.table_name}`{index_hint}
        {where_clause}
        ORDER BY distance
        LIMIT :limit
        """)

        async with self._async_session() as session:
            await self._set_ef_search_async(session, ef_search)
            result = await session.execute(stmt, params)
            results = result.fetchall()

        rows = []
        for item in results:
            meta = self._parse_metadata(item[2])
            rows.append(
                DBEmbeddingRow(
                    node_id=item[0],
                    text=item[1],
                    metadata=meta,
                    similarity=self._distance_to_similarity(item[3])
                    if item[3] is not None
                    else 0.0,
                )
            )

        # MMR re-ranking
        if is_mmr and len(rows) > query.similarity_top_k:
            lambda_mult = kwargs.get("lambda_mult", 0.5)
            node_ids = [r.node_id for r in rows]
            emb_map = await self._afetch_embeddings_by_node_ids(
                node_ids
            )
            embedding_list = [
                emb_map.get(nid, []) for nid in node_ids
            ]
            if all(len(e) > 0 for e in embedding_list):
                selected = self._maximal_marginal_relevance(
                    query_embedding=query.query_embedding,
                    embedding_list=embedding_list,
                    k=query.similarity_top_k,
                    lambda_mult=lambda_mult,
                )
                rows = [rows[i] for i in selected]
            else:
                # I2: Warn when MMR re-ranking is skipped
                _logger.warning(
                    "MMR re-ranking skipped: could not fetch "
                    "embeddings for %d/%d candidate rows. "
                    "Returning distance-ordered results.",
                    sum(1 for e in embedding_list if len(e) == 0),
                    len(embedding_list),
                )

        return self._db_rows_to_query_result(rows)

    # ------------------------------------------------------------------
    # CRUD: get_nodes
    # ------------------------------------------------------------------

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes from the vector store by node_ids or filters."""
        self._initialize()

        nodes: List[BaseNode] = []

        if node_ids:
            # Chunk large node_ids to avoid SQL length limits
            chunk_size = 1000
            with self._session() as session:
                for start in range(0, len(node_ids), chunk_size):
                    chunk = node_ids[start : start + chunk_size]
                    placeholders = ",".join(
                        [f":node_id_{i}" for i in range(len(chunk))]
                    )
                    params = {
                        f"node_id_{i}": nid
                        for i, nid in enumerate(chunk)
                    }
                    stmt = sqlalchemy.text(
                        f"SELECT text, metadata FROM `{self.table_name}` "
                        f"WHERE node_id IN ({placeholders})"
                    )
                    result = session.execute(stmt, params)
                    for item in result:
                        meta = self._parse_metadata(item[1])
                        node = metadata_dict_to_node(meta)
                        node.set_content(str(item[0]))
                        nodes.append(node)
        elif filters:
            global_param_counter = [0]
            where_clause, filter_params = self._filters_to_where_clause(
                filters, global_param_counter
            )
            stmt = sqlalchemy.text(
                f"SELECT text, metadata FROM `{self.table_name}` "
                f"WHERE {where_clause}"
            )
            with self._session() as session:
                result = session.execute(stmt, filter_params)
                for item in result:
                    meta = self._parse_metadata(item[1])
                    node = metadata_dict_to_node(meta)
                    node.set_content(str(item[0]))
                    nodes.append(node)
        else:
            stmt = sqlalchemy.text(
                f"SELECT text, metadata FROM `{self.table_name}` "
                f"LIMIT 10000"
            )
            _logger.warning(
                "get_nodes() called without node_ids or filters; "
                "results capped at 10000 rows."
            )
            with self._session() as session:
                result = session.execute(stmt)
                for item in result:
                    meta = self._parse_metadata(item[1])
                    node = metadata_dict_to_node(meta)
                    node.set_content(str(item[0]))
                    nodes.append(node)

        return nodes

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Async get nodes from the vector store by node_ids or filters."""
        self._initialize()

        nodes: List[BaseNode] = []

        if node_ids:
            # Chunk large node_ids to avoid SQL length limits
            chunk_size = 1000
            async with self._async_session() as session:
                for start in range(0, len(node_ids), chunk_size):
                    chunk = node_ids[start : start + chunk_size]
                    placeholders = ",".join(
                        [f":node_id_{i}" for i in range(len(chunk))]
                    )
                    params = {
                        f"node_id_{i}": nid
                        for i, nid in enumerate(chunk)
                    }
                    stmt = sqlalchemy.text(
                        f"SELECT text, metadata FROM `{self.table_name}` "
                        f"WHERE node_id IN ({placeholders})"
                    )
                    result = await session.execute(stmt, params)
                    for item in result:
                        meta = self._parse_metadata(item[1])
                        node = metadata_dict_to_node(meta)
                        node.set_content(str(item[0]))
                        nodes.append(node)
        elif filters:
            global_param_counter = [0]
            where_clause, filter_params = self._filters_to_where_clause(
                filters, global_param_counter
            )
            stmt = sqlalchemy.text(
                f"SELECT text, metadata FROM `{self.table_name}` "
                f"WHERE {where_clause}"
            )
            async with self._async_session() as session:
                result = await session.execute(stmt, filter_params)
                for item in result:
                    meta = self._parse_metadata(item[1])
                    node = metadata_dict_to_node(meta)
                    node.set_content(str(item[0]))
                    nodes.append(node)
        else:
            stmt = sqlalchemy.text(
                f"SELECT text, metadata FROM `{self.table_name}` "
                f"LIMIT 10000"
            )
            _logger.warning(
                "aget_nodes() called without node_ids or filters; "
                "results capped at 10000 rows."
            )
            async with self._async_session() as session:
                result = await session.execute(stmt)
                for item in result:
                    meta = self._parse_metadata(item[1])
                    node = metadata_dict_to_node(meta)
                    node.set_content(str(item[0]))
                    nodes.append(node)

        return nodes

    # ------------------------------------------------------------------
    # CRUD: delete
    # ------------------------------------------------------------------

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id (stored in metadata)."""
        self._initialize()

        with self._session() as session:
            stmt = sqlalchemy.text(
                f"DELETE FROM `{self.table_name}` "
                f"WHERE JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.ref_doc_id')) "
                f"= :doc_id"
            )
            session.execute(stmt, {"doc_id": ref_doc_id})
            session.commit()

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async delete nodes by ref_doc_id."""
        self._initialize()

        async with self._async_session() as session:
            stmt = sqlalchemy.text(
                f"DELETE FROM `{self.table_name}` "
                f"WHERE JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.ref_doc_id')) "
                f"= :doc_id"
            )
            await session.execute(stmt, {"doc_id": ref_doc_id})
            await session.commit()

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by node_ids or filters."""
        self._initialize()

        with self._session() as session:
            if node_ids:
                # Chunk large node_ids to avoid SQL length limits
                chunk_size = 1000
                for start in range(0, len(node_ids), chunk_size):
                    chunk = node_ids[start : start + chunk_size]
                    placeholders = ",".join(
                        [f":node_id_{i}" for i in range(len(chunk))]
                    )
                    params = {
                        f"node_id_{i}": nid for i, nid in enumerate(chunk)
                    }
                    stmt = sqlalchemy.text(
                        f"DELETE FROM `{self.table_name}` "
                        f"WHERE node_id IN ({placeholders})"
                    )
                    session.execute(stmt, params)
                session.commit()
            elif filters:
                global_param_counter = [0]
                where_clause, filter_params = self._filters_to_where_clause(
                    filters, global_param_counter
                )
                stmt = sqlalchemy.text(
                    f"DELETE FROM `{self.table_name}` "
                    f"WHERE {where_clause}"
                )
                session.execute(stmt, filter_params)
                session.commit()

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Async delete nodes by node_ids or filters."""
        self._initialize()

        async with self._async_session() as session:
            if node_ids:
                # Chunk large node_ids to avoid SQL length limits
                chunk_size = 1000
                for start in range(0, len(node_ids), chunk_size):
                    chunk = node_ids[start : start + chunk_size]
                    placeholders = ",".join(
                        [f":node_id_{i}" for i in range(len(chunk))]
                    )
                    params = {
                        f"node_id_{i}": nid for i, nid in enumerate(chunk)
                    }
                    stmt = sqlalchemy.text(
                        f"DELETE FROM `{self.table_name}` "
                        f"WHERE node_id IN ({placeholders})"
                    )
                    await session.execute(stmt, params)
                await session.commit()
            elif filters:
                global_param_counter = [0]
                where_clause, filter_params = self._filters_to_where_clause(
                    filters, global_param_counter
                )
                stmt = sqlalchemy.text(
                    f"DELETE FROM `{self.table_name}` "
                    f"WHERE {where_clause}"
                )
                await session.execute(stmt, filter_params)
                await session.commit()

    # ------------------------------------------------------------------
    # Metadata-only search and delete (no vector similarity)
    # ------------------------------------------------------------------

    def search_by_metadata(
        self,
        filters: MetadataFilters,
        limit: int = 10,
    ) -> List[BaseNode]:
        """Search nodes by metadata conditions only (no vector similarity).

        Performs a metadata-based query without vector similarity search.
        Useful for filtering, browsing, or auditing stored nodes.

        Args:
            filters: LlamaIndex ``MetadataFilters`` specifying the
                metadata conditions to match.
            limit: Maximum number of results to return. Defaults to 10.

        Returns:
            List of ``BaseNode`` objects matching the metadata filter,
            with content and metadata populated.
        """
        self._initialize()

        global_param_counter = [0]
        where_clause, filter_params = self._filters_to_where_clause(
            filters, global_param_counter
        )

        stmt = sqlalchemy.text(
            f"SELECT node_id, text, metadata "
            f"FROM `{self.table_name}` "
            f"WHERE {where_clause} LIMIT :limit"
        )
        filter_params["limit"] = limit

        with self._session() as session:
            result = session.execute(stmt, filter_params)
            nodes: List[BaseNode] = []
            for row in result:
                meta = self._parse_metadata(row[2])
                node = metadata_dict_to_node(meta or {})
                node.set_content(str(row[1]))
                node.node_id = str(row[0])
                nodes.append(node)
        return nodes

    async def asearch_by_metadata(
        self,
        filters: MetadataFilters,
        limit: int = 10,
    ) -> List[BaseNode]:
        """Async search nodes by metadata conditions only.

        Args:
            filters: LlamaIndex ``MetadataFilters`` specifying the
                metadata conditions to match.
            limit: Maximum number of results to return. Defaults to 10.

        Returns:
            List of ``BaseNode`` objects matching the metadata filter.
        """
        self._initialize()

        global_param_counter = [0]
        where_clause, filter_params = self._filters_to_where_clause(
            filters, global_param_counter
        )

        stmt = sqlalchemy.text(
            f"SELECT node_id, text, metadata "
            f"FROM `{self.table_name}` "
            f"WHERE {where_clause} LIMIT :limit"
        )
        filter_params["limit"] = limit

        async with self._async_session() as session:
            result = await session.execute(stmt, filter_params)
            nodes: List[BaseNode] = []
            for row in result:
                meta = self._parse_metadata(row[2])
                node = metadata_dict_to_node(meta or {})
                node.set_content(str(row[1]))
                node.node_id = str(row[0])
                nodes.append(node)
        return nodes

    def delete_by_metadata(self, filters: MetadataFilters) -> int:
        """Delete nodes matching metadata conditions.

        Args:
            filters: LlamaIndex ``MetadataFilters`` specifying the
                metadata conditions to match. Required.

        Returns:
            Number of deleted rows.
        """
        self._initialize()

        global_param_counter = [0]
        where_clause, filter_params = self._filters_to_where_clause(
            filters, global_param_counter
        )

        stmt = sqlalchemy.text(
            f"DELETE FROM `{self.table_name}` "
            f"WHERE {where_clause}"
        )

        with self._session() as session:
            result = session.execute(stmt, filter_params)
            session.commit()
            return result.rowcount

    async def adelete_by_metadata(
        self,
        filters: MetadataFilters,
    ) -> int:
        """Async delete nodes matching metadata conditions.

        Args:
            filters: LlamaIndex ``MetadataFilters`` specifying the
                metadata conditions to match. Required.

        Returns:
            Number of deleted rows.
        """
        self._initialize()

        global_param_counter = [0]
        where_clause, filter_params = self._filters_to_where_clause(
            filters, global_param_counter
        )

        stmt = sqlalchemy.text(
            f"DELETE FROM `{self.table_name}` "
            f"WHERE {where_clause}"
        )

        async with self._async_session() as session:
            result = await session.execute(stmt, filter_params)
            await session.commit()
            return result.rowcount

    # ------------------------------------------------------------------
    # Utility: count, clear, drop, close
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of rows in the vector table."""
        self._initialize()

        with self._session() as session:
            result = session.execute(
                sqlalchemy.text(
                    f"SELECT COUNT(*) FROM `{self.table_name}`"
                )
            )
            row = result.fetchone()
        return row[0] if row else 0

    async def acount(self) -> int:
        """Async return the number of rows."""
        self._initialize()

        async with self._async_session() as session:
            result = await session.execute(
                sqlalchemy.text(
                    f"SELECT COUNT(*) FROM `{self.table_name}`"
                )
            )
            row = result.fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        """Clear all data (TRUNCATE TABLE).

        Uses TRUNCATE instead of DELETE FROM because PolarDB-X
        restricts DELETE without a WHERE clause.
        """
        self._initialize()

        with self._session() as session:
            session.execute(
                sqlalchemy.text(f"TRUNCATE TABLE `{self.table_name}`")
            )
            session.commit()

    async def aclear(self) -> None:
        """Async clear all data (TRUNCATE TABLE)."""
        self._initialize()

        async with self._async_session() as session:
            await session.execute(
                sqlalchemy.text(f"TRUNCATE TABLE `{self.table_name}`")
            )
            await session.commit()

    def drop(self) -> None:
        """Drop the table and close connections."""
        self._initialize()

        with self._session() as session:
            session.execute(
                sqlalchemy.text(f"DROP TABLE IF EXISTS `{self.table_name}`")
            )
            session.commit()
        self.close()

    async def adrop(self) -> None:
        """Async drop the table and close connections."""
        self._initialize()

        async with self._async_session() as session:
            await session.execute(
                sqlalchemy.text(f"DROP TABLE IF EXISTS `{self.table_name}`")
            )
            await session.commit()
        await self.aclose()

    def close(self) -> None:
        """Close sync and async engines."""
        if not self._is_initialized:
            return
        if self._engine:
            self._engine.dispose()
        if self._async_engine:
            import asyncio

            try:
                asyncio.get_running_loop()
                # We're inside a running event loop — use a thread to
                # dispose the async engine without blocking the loop.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._async_engine.dispose()
                    )
                    future.result()
            except RuntimeError:
                # No running event loop — safe to use asyncio.run().
                try:
                    asyncio.run(self._async_engine.dispose())
                except RuntimeError:
                    # The event loop may have been closed (e.g. after
                    # pytest-asyncio finishes).  Create a fresh one.
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(
                            self._async_engine.dispose()
                        )
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)
        self._is_initialized = False

    async def aclose(self) -> None:
        """Async close engines."""
        if not self._is_initialized:
            return
        if self._engine:
            self._engine.dispose()
        if self._async_engine:
            await self._async_engine.dispose()
        self._is_initialized = False

    # ------------------------------------------------------------------
    # Phase 2: Dynamic vector index management
    # ------------------------------------------------------------------

    def apply_vector_index(
        self,
        index_name: str = "vi",
        m: Optional[int] = None,
        distance: Optional[str] = None,
        ef_construction: Optional[int] = None,
    ) -> None:
        """Create a vector index on the embedding column dynamically.

        Args:
            index_name: Name for the vector index. Defaults to ``"vi"``.
            m: HNSW M parameter (3-200). Defaults to the store's default_m.
            distance: Distance function (``"COSINE"``, ``"EUCLIDEAN"``,
                or ``"INNER_PRODUCT"``). Defaults to the store's method.
                ``INNER_PRODUCT`` requires v3.
            ef_construction: HNSW build-time candidate list size (5-1000).
                v3 only; silently ignored on old versions.
        """
        self._initialize()
        self._validate_identifier(index_name)
        # S1: Validate m and ef_construction to prevent SQL injection
        m_val = m or self.default_m
        self._validate_positive_int(m_val, "m")
        if not (3 <= m_val <= 200):
            raise ValueError(f"m must be 3-200, got {m_val}")
        dist_val = (distance or self.distance_method).upper()
        if dist_val not in ("COSINE", "EUCLIDEAN", "INNER_PRODUCT"):
            raise ValueError(
                f"Invalid distance function: {dist_val}. "
                "Must be 'COSINE', 'EUCLIDEAN', or 'INNER_PRODUCT'."
            )
        if (
            dist_val == "INNER_PRODUCT"
            and not self._capabilities.get("vec_dim", False)
        ):
            raise NotSupportedError(
                "distance='INNER_PRODUCT' requires PolarDB-X v3. "
                "Use 'COSINE' or 'EUCLIDEAN' for old versions."
            )
        ef_val = ef_construction or self._ef_construction
        if ef_val is not None:
            self._validate_positive_int(ef_val, "ef_construction")
            if not (5 <= ef_val <= 1000):
                raise ValueError(
                    f"ef_construction must be 5-1000, got {ef_val}"
                )
        ef_clause = ""
        if ef_val is not None and self._capabilities.get("vec_dim", False):
            ef_clause = f" EF_CONSTRUCTION={ef_val}"

        with self._session() as session:
            session.execute(
                sqlalchemy.text(
                    f"ALTER TABLE `{self.table_name}` "
                    f"ADD VECTOR INDEX `{index_name}` (embedding) "
                    f"M={m_val}{ef_clause} DISTANCE={dist_val}"
                )
            )
            session.commit()
        self._vector_index_name = index_name
        _logger.info(
            "Vector index '%s' created on table %s",
            index_name,
            self.table_name,
        )

    async def aapply_vector_index(
        self,
        index_name: str = "vi",
        m: Optional[int] = None,
        distance: Optional[str] = None,
        ef_construction: Optional[int] = None,
    ) -> None:
        """Async create a vector index."""
        self._initialize()
        self._validate_identifier(index_name)
        # S1: Validate m and ef_construction to prevent SQL injection
        m_val = m or self.default_m
        self._validate_positive_int(m_val, "m")
        if not (3 <= m_val <= 200):
            raise ValueError(f"m must be 3-200, got {m_val}")
        dist_val = (distance or self.distance_method).upper()
        if dist_val not in ("COSINE", "EUCLIDEAN", "INNER_PRODUCT"):
            raise ValueError(
                f"Invalid distance function: {dist_val}. "
                "Must be 'COSINE', 'EUCLIDEAN', or 'INNER_PRODUCT'."
            )
        if (
            dist_val == "INNER_PRODUCT"
            and not self._capabilities.get("vec_dim", False)
        ):
            raise NotSupportedError(
                "distance='INNER_PRODUCT' requires PolarDB-X v3. "
                "Use 'COSINE' or 'EUCLIDEAN' for old versions."
            )
        ef_val = ef_construction or self._ef_construction
        if ef_val is not None:
            self._validate_positive_int(ef_val, "ef_construction")
            if not (5 <= ef_val <= 1000):
                raise ValueError(
                    f"ef_construction must be 5-1000, got {ef_val}"
                )
        ef_clause = ""
        if ef_val is not None and self._capabilities.get("vec_dim", False):
            ef_clause = f" EF_CONSTRUCTION={ef_val}"

        async with self._async_session() as session:
            await session.execute(
                sqlalchemy.text(
                    f"ALTER TABLE `{self.table_name}` "
                    f"ADD VECTOR INDEX `{index_name}` (embedding) "
                    f"M={m_val}{ef_clause} DISTANCE={dist_val}"
                )
            )
            await session.commit()
        self._vector_index_name = index_name
        _logger.info(
            "Vector index '%s' created on table %s",
            index_name,
            self.table_name,
        )

    def drop_vector_index(self, index_name: Optional[str] = None) -> None:
        """Drop a vector index.

        Args:
            index_name: Name of the index to drop. If None, auto-detected.
        """
        self._initialize()
        name = index_name or self._detect_vector_index_name()
        if not name:
            raise ValueError("No vector index name specified or detectable.")
        self._validate_identifier(name)
        with self._session() as session:
            session.execute(
                sqlalchemy.text(
                    f"ALTER TABLE `{self.table_name}` DROP INDEX `{name}`"
                )
            )
            session.commit()
        self._vector_index_name = None
        self._vector_index_checked = False
        _logger.info(
            "Vector index '%s' dropped from table %s",
            name,
            self.table_name,
        )

    async def adrop_vector_index(
        self, index_name: Optional[str] = None
    ) -> None:
        """Async drop a vector index."""
        self._initialize()
        name = index_name or self._detect_vector_index_name()
        if not name:
            raise ValueError("No vector index name specified or detectable.")
        self._validate_identifier(name)
        async with self._async_session() as session:
            await session.execute(
                sqlalchemy.text(
                    f"ALTER TABLE `{self.table_name}` DROP INDEX `{name}`"
                )
            )
            await session.commit()
        self._vector_index_name = None
        self._vector_index_checked = False
        _logger.info(
            "Vector index '%s' dropped from table %s",
            name,
            self.table_name,
        )

    # ------------------------------------------------------------------
    # Phase 2: Monitoring & maintenance
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get vector index runtime statistics (Vidx* status variables)."""
        self._initialize()
        with self._session() as session:
            result = session.execute(
                sqlalchemy.text("SHOW GLOBAL STATUS LIKE 'Vidx%'")
            )
            rows = result.fetchall()
            return {row[0]: row[1] for row in rows}

    async def aget_stats(self) -> Dict[str, Any]:
        """Async get vector index runtime statistics."""
        self._initialize()
        async with self._async_session() as session:
            result = await session.execute(
                sqlalchemy.text("SHOW GLOBAL STATUS LIKE 'Vidx%'")
            )
            rows = result.fetchall()
            return {row[0]: row[1] for row in rows}

    def optimize(self) -> None:
        """Rebuild the vector index to reclaim space and improve recall.

        Note: PolarDB-X OPTIMIZE TABLE returns a result set, so we
        must fetchall() to avoid "Unread result found" errors.
        """
        self._initialize()
        with self._session() as session:
            result = session.execute(
                sqlalchemy.text(f"OPTIMIZE TABLE `{self.table_name}`")
            )
            result.fetchall()
            session.commit()
        _logger.info("OPTIMIZE TABLE executed on %s", self.table_name)

    async def aoptimize(self) -> None:
        """Async rebuild the vector index."""
        self._initialize()
        async with self._async_session() as session:
            result = await session.execute(
                sqlalchemy.text(f"OPTIMIZE TABLE `{self.table_name}`")
            )
            result.fetchall()
            await session.commit()
        _logger.info("OPTIMIZE TABLE executed on %s", self.table_name)

    # ------------------------------------------------------------------
    # v3 Enhanced features (raise NotSupportedError on old versions)
    # ------------------------------------------------------------------

    def _require_v3(self, feature: str) -> None:
        """Raise NotSupportedError if v3 capabilities are not available.

        Uses ``vec_dim`` (VECTOR_DIM) as the v3 indicator — present iff
        the instance has v3 vector features enabled.
        """
        if not self._capabilities.get("vec_dim", False):
            raise NotSupportedError(
                f"{feature} requires PolarDB-X v3 with vector index "
                "support. Current instance does not support v3 vector "
                "features."
            )

    def preload_index(self) -> None:
        """Preload the HNSW vector index into memory cache (v3 only).

        Loads the entire HNSW auxiliary table graph into the shared
        cache to eliminate cold-start latency on the first query.
        """
        self._initialize()
        self._require_v3("preload_index()")
        # S2: Re-validate mutable Pydantic fields before SQL interpolation
        self._validate_identifier(self.database)
        self._validate_identifier(self.table_name)
        with self._session() as session:
            result = session.execute(
                sqlalchemy.text(
                    f"CALL dbms_vidx.preload("
                    f"'{self.database}', '{self.table_name}', 'embedding')"
                )
            )
            result.fetchall()
            session.commit()
        _logger.info(
            "Preloaded vector index for table %s", self.table_name
        )

    async def apreload_index(self) -> None:
        """Async preload the HNSW vector index (v3 only)."""
        self._initialize()
        self._require_v3("preload_index()")
        # S2: Re-validate mutable Pydantic fields before SQL interpolation
        self._validate_identifier(self.database)
        self._validate_identifier(self.table_name)
        async with self._async_session() as session:
            result = await session.execute(
                sqlalchemy.text(
                    f"CALL dbms_vidx.preload("
                    f"'{self.database}', '{self.table_name}', 'embedding')"
                )
            )
            result.fetchall()
            await session.commit()
        _logger.info(
            "Preloaded vector index for table %s", self.table_name
        )

    def preload_check(self) -> Dict[str, Any]:
        """Check if preloading would fit in cache (v3 only).

        Returns:
            Dictionary with check results (rows, memory estimate, etc.).
        """
        self._initialize()
        self._require_v3("preload_check()")
        # S2: Re-validate mutable Pydantic fields before SQL interpolation
        self._validate_identifier(self.database)
        self._validate_identifier(self.table_name)
        with self._session() as session:
            result = session.execute(
                sqlalchemy.text(
                    f"CALL dbms_vidx.preload_check("
                    f"'{self.database}', '{self.table_name}', 'embedding')"
                )
            )
            rows = result.fetchall()
            session.commit()
        if not rows:
            return {}
        return {str(idx): dict(row._mapping) for idx, row in enumerate(rows)}

    async def apreload_check(self) -> Dict[str, Any]:
        """Async check if preloading would fit in cache (v3 only)."""
        self._initialize()
        self._require_v3("preload_check()")
        # S2: Re-validate mutable Pydantic fields before SQL interpolation
        self._validate_identifier(self.database)
        self._validate_identifier(self.table_name)
        async with self._async_session() as session:
            result = await session.execute(
                sqlalchemy.text(
                    f"CALL dbms_vidx.preload_check("
                    f"'{self.database}', '{self.table_name}', 'embedding')"
                )
            )
            rows = result.fetchall()
            await session.commit()
        if not rows:
            return {}
        return {str(idx): dict(row._mapping) for idx, row in enumerate(rows)}

    def explain_index_health(self) -> Dict[str, Any]:
        """Check vector index health and return diagnostics (v3 only).

        Combines information_schema.VECTOR_INDEXES metadata with
        EXPLAIN and EXPLAIN ANALYZE output to provide a comprehensive
        health report.

        Returns:
            Dictionary with keys:
                - index_info: VECTOR_INDEXES metadata (name, algorithm,
                  metric, dimension, M, EF_CONSTRUCTION, etc.)
                - explain: plain EXPLAIN output (index selection info)
                - explain_analyze: EXPLAIN ANALYZE output with actual
                  nodes_visited cost (v3 only)
        """
        self._initialize()
        self._require_v3("explain_index_health()")
        result: Dict[str, Any] = {}

        with self._session() as session:
            # 1. Query VECTOR_INDEXES view for index metadata
            result_set = session.execute(
                sqlalchemy.text(
                    "SELECT INDEX_NAME, HLINDEX_TABLE_NAME, COLUMN_NAME, "
                    "ALGORITHM, METRIC_TYPE, DIMENSION, M, "
                    "EF_CONSTRUCTION, QUANTIZE_TYPE "
                    "FROM information_schema.VECTOR_INDEXES "
                    "WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table"
                ),
                {"schema": self.database, "table": self.table_name},
            )
            rows = result_set.fetchall()
            if rows:
                result["index_info"] = dict(rows[0]._mapping)
                dim = rows[0]._mapping.get("DIMENSION", self.embed_dim)
                # Clamp dim to avoid generating huge SQL vectors
                dim = max(1, min(int(dim), 4096))
            else:
                result["index_info"] = None
                dim = self.embed_dim

            # 2. Run EXPLAIN to check if vector index is used
            dist_func = self._get_distance_func()
            sample_vec = json.dumps([0.0] * dim)
            result_set = session.execute(
                sqlalchemy.text(
                    f"EXPLAIN SELECT id FROM `{self.table_name}` "
                    f"ORDER BY {dist_func}(embedding, "
                    f"VEC_FROMTEXT('{sample_vec}')) LIMIT 10"
                )
            )
            result["explain"] = [
                dict(r._mapping) for r in result_set.fetchall()
            ]

            # 3. Run EXPLAIN ANALYZE for actual traversal cost
            #    (nodes_visited shows real ANN cost)
            try:
                result_set = session.execute(
                    sqlalchemy.text(
                        f"EXPLAIN ANALYZE FORMAT=TREE "
                        f"SELECT id FROM `{self.table_name}` "
                        f"ORDER BY {dist_func}(embedding, "
                        f"VEC_FROMTEXT('{sample_vec}')) LIMIT 10"
                    )
                )
                result["explain_analyze"] = [
                    dict(r._mapping) for r in result_set.fetchall()
                ]
            except Exception as e:
                _logger.debug(
                    "EXPLAIN ANALYZE failed (may not be supported): %s", e
                )
                result["explain_analyze"] = None

        return result

    async def aexplain_index_health(self) -> Dict[str, Any]:
        """Async check vector index health (v3 only)."""
        self._initialize()
        self._require_v3("explain_index_health()")
        result: Dict[str, Any] = {}

        async with self._async_session() as session:
            result_set = await session.execute(
                sqlalchemy.text(
                    "SELECT INDEX_NAME, HLINDEX_TABLE_NAME, COLUMN_NAME, "
                    "ALGORITHM, METRIC_TYPE, DIMENSION, M, "
                    "EF_CONSTRUCTION, QUANTIZE_TYPE "
                    "FROM information_schema.VECTOR_INDEXES "
                    "WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table"
                ),
                {"schema": self.database, "table": self.table_name},
            )
            rows = result_set.fetchall()
            if rows:
                result["index_info"] = dict(rows[0]._mapping)
                dim = rows[0]._mapping.get("DIMENSION", self.embed_dim)
                # Clamp dim to avoid generating huge SQL vectors
                dim = max(1, min(int(dim), 4096))
            else:
                result["index_info"] = None
                dim = self.embed_dim

            dist_func = self._get_distance_func()
            sample_vec = json.dumps([0.0] * dim)
            result_set = await session.execute(
                sqlalchemy.text(
                    f"EXPLAIN SELECT id FROM `{self.table_name}` "
                    f"ORDER BY {dist_func}(embedding, "
                    f"VEC_FROMTEXT('{sample_vec}')) LIMIT 10"
                )
            )
            result["explain"] = [
                dict(r._mapping) for r in result_set.fetchall()
            ]

            try:
                result_set = await session.execute(
                    sqlalchemy.text(
                        f"EXPLAIN ANALYZE FORMAT=TREE "
                        f"SELECT id FROM `{self.table_name}` "
                        f"ORDER BY {dist_func}(embedding, "
                        f"VEC_FROMTEXT('{sample_vec}')) LIMIT 10"
                    )
                )
                result["explain_analyze"] = [
                    dict(r._mapping) for r in result_set.fetchall()
                ]
            except Exception as e:
                _logger.debug(
                    "EXPLAIN ANALYZE failed (may not be supported): %s", e
                )
                result["explain_analyze"] = None

        return result
