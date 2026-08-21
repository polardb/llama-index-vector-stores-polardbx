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
    Union,
)
from urllib.parse import quote_plus

import sqlalchemy
import sqlalchemy.ext.asyncio
from llama_index.core.bridge.pydantic import ConfigDict, PrivateAttr, SecretStr
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
from llama_index.vector_stores.polardbx.column import Column

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
    "distance_strategy",
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
    "id_column",
    "node_id_column",
    "text_column",
    "embedding_column",
    "metadata_json_column",
    "metadata_columns",
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
                distance_strategy="cosine",
            )
    """

    # S2: Freeze model fields to prevent post-init SQL injection via
    # table_name/database mutation. PrivateAttr values (engine, session,
    # etc.) are not affected by frozen and remain mutable.
    model_config = ConfigDict(frozen=True)

    stores_text: bool = True
    flat_metadata: bool = False

    connection_string: SecretStr
    table_name: str = "llama_index_table"
    database: str
    embed_dim: int = 1536
    default_m: int = 6
    distance_strategy: Literal["euclidean", "cosine", "inner_product"] = "cosine"
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

    # Custom column configuration (Phase: custom column support)
    _id_column: str = PrivateAttr(default="id")
    _node_id_column: str = PrivateAttr(default="node_id")
    _text_column: str = PrivateAttr(default="text")
    _embedding_column: str = PrivateAttr(default="embedding")
    _metadata_json_column: Optional[str] = PrivateAttr(default="metadata")
    _metadata_column_objs: List[Column] = PrivateAttr(default_factory=list)
    _metadata_column_names: List[str] = PrivateAttr(default_factory=list)

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
        if not name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid identifier: {name!r}")
        if len(name) > 64:
            raise ValueError(
                f"Identifier too long: {name!r}. Maximum length is 64 characters."
            )
        return name

    @staticmethod
    def _validate_positive_int(value: int, param_name: str) -> int:
        """Validate that a value is a positive integer."""
        # L4: Reject booleans — isinstance(True, int) is True in Python
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
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
        distance_strategy: Literal["euclidean", "cosine", "inner_product"] = "cosine",
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
        id_column: Optional[str] = None,
        node_id_column: Optional[str] = None,
        text_column: Optional[str] = None,
        embedding_column: Optional[str] = None,
        metadata_json_column: Optional[str] = "metadata",
        metadata_columns: Optional[List[Union[Column, str]]] = None,
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
            distance_strategy: Distance function — ``"cosine"``,
                ``"euclidean"``, or ``"inner_product"`` (newer versions only).
                Defaults to ``"cosine"``.
            perform_setup: If True, auto-create table on init.
                Defaults to True.
            debug: Enable SQLAlchemy echo mode. Defaults to False.
            ef_construction: HNSW build-time candidate list size
                (5-1000, newer versions only). Ignored on older versions.
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
            id_column: Custom name for the primary key column.
                Defaults to ``"id"``.
            node_id_column: Custom name for the LlamaIndex node_id
                column. Defaults to ``"node_id"``.
            text_column: Custom name for the text content column.
                Defaults to ``"text"``.
            embedding_column: Custom name for the vector embedding
                column. Defaults to ``"embedding"``.
            metadata_json_column: Custom name for the JSON metadata
                column. Set to None to disable the JSON metadata column
                (requires metadata_columns). Defaults to ``"metadata"``.
            metadata_columns: List of Column objects or column name
                strings for dedicated metadata columns. Column objects
                require a data_type (e.g. ``Column("price", "DECIMAL(10,2)")``)
                and are used when creating a new table. String names
                (e.g. ``"category"``) are used when connecting to an
                existing table. Mapped keys are extracted from the
                metadata dict into their own columns; remaining keys
                go into the JSON metadata column. Defaults to None.
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

        if distance_strategy not in ("euclidean", "cosine", "inner_product"):
            raise ValueError(
                f"Invalid distance_strategy: {distance_strategy}. "
                "Must be 'cosine', 'euclidean', or 'inner_product'."
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

        # ------------------------------------------------------------------
        # Custom column validation and normalization
        # ------------------------------------------------------------------
        _id_col = id_column or "id"
        _node_id_col = node_id_column or "node_id"
        _text_col = text_column or "text"
        _emb_col = embedding_column or "embedding"
        self._validate_identifier(_id_col)
        self._validate_identifier(_node_id_col)
        self._validate_identifier(_text_col)
        self._validate_identifier(_emb_col)
        if metadata_json_column is not None:
            self._validate_identifier(metadata_json_column)

        # Partition column validation (must be after _id_col computation
        # so that custom id_column names are properly validated)
        _pcolumn = partition_column or _id_col
        if _pby and _pcolumn != _id_col:
            # W4: PolarDB-X requires the partition key to be part of
            # every unique index. The only unique index is PRIMARY KEY,
            # so partition_column must match the id_column.
            raise ValueError(
                f"For vector tables, partition_column must match "
                f"id_column (got partition_column={_pcolumn!r}, "
                f"id_column={_id_col!r}). The vector table schema only "
                f"supports partitioning on the primary key column."
            )

        # Normalize metadata_columns: extract Column names, keep Column
        # objects for DDL generation
        _metadata_col_objs: List[Column] = []
        _metadata_col_names: List[str] = []
        if metadata_columns:
            for mc in metadata_columns:
                if isinstance(mc, Column):
                    self._validate_identifier(mc.name)
                    _metadata_col_objs.append(mc)
                    _metadata_col_names.append(mc.name)
                elif isinstance(mc, str):
                    self._validate_identifier(mc)
                    _metadata_col_names.append(mc)
                else:
                    raise TypeError(
                        f"metadata_columns items must be Column or str, "
                        f"got {type(mc).__name__}"
                    )

        # Detect duplicate column names
        all_col_names = [_id_col, _node_id_col, _text_col, _emb_col]
        if metadata_json_column is not None:
            all_col_names.append(metadata_json_column)
        seen: set = set()
        for name in all_col_names:
            if name in seen:
                raise ValueError(
                    f"Duplicate column name '{name}'. "
                    "Column names must be unique."
                )
            seen.add(name)
        for name in _metadata_col_names:
            if name in seen:
                raise ValueError(
                    f"Duplicate column name '{name}'. "
                    "metadata_columns names must not overlap with "
                    "core column names or each other."
                )
            seen.add(name)

        # If no JSON column and no metadata_columns, all metadata is lost
        if (
            metadata_json_column is None
            and not _metadata_col_names
        ):
            raise ValueError(
                "metadata_json_column is None and "
                "metadata_columns is empty. There is no place to "
                "store unmapped metadata. Set metadata_json_column "
                "to a column name, or provide metadata_columns."
            )

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
            distance_strategy=distance_strategy,
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
        if vector_index_name is not None:
            self._validate_identifier(vector_index_name)
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

        # Custom column configuration
        self._id_column = _id_col
        self._node_id_column = _node_id_col
        self._text_column = _text_col
        self._embedding_column = _emb_col
        self._metadata_json_column = metadata_json_column
        self._metadata_column_objs = _metadata_col_objs
        self._metadata_column_names = _metadata_col_names

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
        distance_strategy: Literal["euclidean", "cosine", "inner_product"] = "cosine",
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
        id_column: Optional[str] = None,
        node_id_column: Optional[str] = None,
        text_column: Optional[str] = None,
        embedding_column: Optional[str] = None,
        metadata_json_column: Optional[str] = "metadata",
        metadata_columns: Optional[List[Union[Column, str]]] = None,
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
            distance_strategy=distance_strategy,
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
            id_column=id_column,
            node_id_column=node_id_column,
            text_column=text_column,
            embedding_column=embedding_column,
            metadata_json_column=metadata_json_column,
            metadata_columns=metadata_columns,
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
                            self._validate_distance_strategy()
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
        caps["dbms_vidx"] = self._probe_dbms_vidx()
        self._capabilities = caps
        _logger.info("Detected capabilities: %s", self._capabilities)

    def _probe_dbms_vidx(self) -> bool:
        """Probe whether dbms_vidx stored procedures exist.

        Tries calling ``dbms_vidx.preload_check`` with dummy args.
        If the error is ``ERR_PROCEDURE_NOT_FOUND``, the procedures are
        absent. Any other error (invalid args, table not found, etc.)
        means the procedure exists.
        """
        from sqlalchemy import text

        try:
            with self._session() as session:
                session.execute(
                    text("CALL dbms_vidx.preload_check('', '', '')")
                )
                return True
        except Exception as e:
            err_msg = str(e)
            if "PROCEDURE_NOT_FOUND" in err_msg or "Not Found" in err_msg:
                _logger.debug("dbms_vidx procedures not found on this instance")
                return False
            # Other errors mean the procedure exists but rejected the args
            return True

    def _validate_distance_strategy(self) -> None:
        """Validate INNER_PRODUCT requires newer version support."""
        if (
            self.distance_strategy == "inner_product"
            and not self._capabilities.get("vec_dim", False)
        ):
            raise NotSupportedError(
                "distance_strategy='inner_product' requires a newer "
                "PolarDB-X version. Use 'cosine' or 'euclidean' for "
                "older versions."
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
                    "VEC_DISTANCE exists but needs index context: %s",
                    self._sanitize_error(e),
                )
                return True
            _logger.debug(
                "VEC_DISTANCE probe failed: %s", self._sanitize_error(e)
            )
            return False

    def _probe_function(self, sql: str) -> bool:
        """Probe whether a SQL function is available."""
        from sqlalchemy import text

        try:
            with self._session() as session:
                result = session.execute(text(sql))
                return result.fetchone() is not None
        except Exception as e:
            _logger.debug(
                "Function probe failed [%s]: %s",
                sql, self._sanitize_error(e),
            )
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
                "Table probe failed [%s.%s]: %s",
                schema, table, self._sanitize_error(e),
            )
            return False

    def _get_distance_func(self) -> str:
        """Return the optimal distance function for the current instance.

        Prefers VEC_DISTANCE (auto-inference, newer versions) when available;
        falls back to explicit VEC_DISTANCE_COSINE / _EUCLIDEAN /
        _INNER_PRODUCT otherwise.
        """
        if self._capabilities.get("vec_distance", False):
            return "VEC_DISTANCE"
        if self.distance_strategy == "cosine":
            return "VEC_DISTANCE_COSINE"
        if self.distance_strategy == "inner_product":
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

    @property
    def _has_custom_columns(self) -> bool:
        """True if custom column configuration differs from defaults."""
        return (
            self._id_column != "id"
            or self._node_id_column != "node_id"
            or self._text_column != "text"
            or self._embedding_column != "embedding"
            or bool(self._metadata_column_names)
            or self._metadata_json_column != "metadata"
        )

    def _create_table_if_not_exists(self) -> None:
        """Create the vector table if it does not exist."""
        from sqlalchemy import text

        if self._has_custom_columns:
            base_ddl = self._build_create_table_sql_custom()
        else:
            # Build optional EF_CONSTRUCTION clause (v3 only)
            ef_clause = ""
            if (
                self._ef_construction is not None
                and self._capabilities.get("vec_dim", False)
            ):
                ef_clause = f" EF_CONSTRUCTION={self._ef_construction}"

            # When partitioning is enabled, PolarDB-X requires that unique
            # indexes include the partition key. The node_id unique index
            # does not include the partition key (default: id), so
            # downgrade it to a regular INDEX. The id PRIMARY KEY already
            # guarantees row uniqueness, and LlamaIndex handles dedup at
            # the app level.
            node_index_type = (
                "INDEX" if self._has_partition else "UNIQUE INDEX"
            )

            base_ddl = f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                id VARCHAR(36) PRIMARY KEY,
                node_id VARCHAR(255) NOT NULL,
                text LONGTEXT,
                metadata JSON,
                embedding VECTOR({self.embed_dim}) NOT NULL,
                {node_index_type} `node_id_index` (node_id),
                VECTOR INDEX `vi` (embedding) M={self.default_m}{ef_clause} DISTANCE={self.distance_strategy.upper()}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """

        partition_clause = self._build_partition_clause()
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
                            "on older DN versions. Try upgrading the DN "
                            "version, or remove partition parameters "
                            "(partition_by, broadcast, etc.) to create a "
                            "non-partitioned vector table."
                        ) from e
                raise

    def _build_create_table_sql_custom(self) -> str:
        """Build CREATE TABLE SQL with custom column definitions.

        Generates DDL using the user-specified column names and
        metadata column definitions. Falls back to sensible defaults
        for the core columns (id, node_id, text, embedding).
        """
        # Build optional EF_CONSTRUCTION clause (v3 only)
        index_extra = ""
        if (
            self._ef_construction is not None
            and self._capabilities.get("vec_dim", False)
        ):
            index_extra = f" EF_CONSTRUCTION={self._ef_construction}"

        # When partitioning is enabled, PolarDB-X requires that unique
        # indexes include the partition key. The node_id unique index does
        # not include the partition key (default: id), so downgrade it
        # to a regular INDEX.
        node_index_type = (
            "INDEX" if self._has_partition else "UNIQUE INDEX"
        )

        lines: List[str] = []
        lines.append(
            f"    `{self._id_column}` VARCHAR(36) PRIMARY KEY"
        )
        lines.append(
            f"    `{self._node_id_column}` VARCHAR(255) NOT NULL"
        )
        lines.append(
            f"    `{self._text_column}` LONGTEXT"
        )
        if self._metadata_json_column is not None:
            lines.append(
                f"    `{self._metadata_json_column}` JSON"
            )
        lines.append(
            f"    `{self._embedding_column}` "
            f"VECTOR({self.embed_dim}) NOT NULL"
        )
        # node_id index
        lines.append(
            f"    {node_index_type} `node_id_index` "
            f"({self._node_id_column})"
        )
        # vector index
        lines.append(
            f"    VECTOR INDEX `vi` ({self._embedding_column}) "
            f"M={self.default_m}{index_extra} "
            f"DISTANCE={self.distance_strategy.upper()}"
        )
        # Add metadata columns: use Column objects when available
        # (for new tables), fall back to TEXT for string-only names
        col_obj_map = {c.name: c for c in self._metadata_column_objs}
        for name in self._metadata_column_names:
            col = col_obj_map.get(name)
            if col is not None:
                col_def = f"    `{name}` {col.data_type}"
                if not col.nullable:
                    col_def += " NOT NULL"
                if col.default is not None:
                    col_def += f" DEFAULT {col.default}"
            else:
                # String-only column name: use TEXT as default type
                col_def = f"    `{name}` TEXT"
            lines.append(col_def)

        inner = ",\n".join(lines)
        return (
            f"CREATE TABLE IF NOT EXISTS `{self.table_name}` (\n"
            f"{inner}\n"
            f") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 "
            f"COLLATE=utf8mb4_unicode_ci\n"
        )

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

    # ------------------------------------------------------------------
    # SQL builder methods for UPSERT (INSERT + ON DUPLICATE KEY UPDATE)
    # ------------------------------------------------------------------

    def _build_upsert_sql(self, partitioned: bool) -> str:
        """Build INSERT SQL for adding a node.

        When custom columns are not configured, uses the original
        static SQL.  When custom columns are configured, dynamically
        generates the column list and value placeholders.

        Args:
            partitioned: If True, use plain INSERT (no ON DUPLICATE KEY
                UPDATE) because partitioned tables use DELETE-then-INSERT.

        Returns:
            The INSERT SQL statement.
        """
        if not self._has_custom_columns:
            if partitioned:
                return f"""
                INSERT INTO `{self.table_name}` (id, node_id, text, embedding, metadata)
                VALUES (
                    UUID(),
                    :node_id,
                    :text,
                    VEC_FROMTEXT(:embedding),
                    :metadata
                )
                """
            return f"""
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
            """

        # Custom column mode: dynamically build column list
        # Order: id, node_id, text, [metadata_json], embedding,
        #        [metadata_col1, metadata_col2, ...]
        cols: List[str] = [
            f"`{self._id_column}`",
            f"`{self._node_id_column}`",
            f"`{self._text_column}`",
        ]
        placeholders: List[str] = [
            "UUID()",
            ":node_id",
            ":text",
        ]
        if self._metadata_json_column is not None:
            cols.append(f"`{self._metadata_json_column}`")
            placeholders.append(":metadata")
        cols.append(f"`{self._embedding_column}`")
        placeholders.append("VEC_FROMTEXT(:embedding)")

        # Metadata columns
        for name in self._metadata_column_names:
            cols.append(f"`{name}`")
            placeholders.append(f":meta_{name}")

        col_list = ", ".join(cols)
        val_list = ", ".join(placeholders)

        if partitioned:
            return (
                f"INSERT INTO `{self.table_name}` ({col_list})\n"
                f"VALUES ({val_list})\n"
            )

        # ON DUPLICATE KEY UPDATE: all columns except primary key
        update_cols = cols[1:]
        update_clause = ",\n    ".join(
            f"{c} = VALUES({c})" for c in update_cols
        )
        return (
            f"INSERT INTO `{self.table_name}` ({col_list})\n"
            f"VALUES ({val_list})\n"
            f"ON DUPLICATE KEY UPDATE\n    {update_clause}\n"
        )

    def _build_upsert_params(
        self, item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the parameter dict for the INSERT SQL.

        When custom columns are configured, extracts mapped metadata
        keys into their own parameters (prefixed ``meta_``) and puts
        the remaining metadata into the JSON parameter.

        Args:
            item: The table row dict from _node_to_table_row.

        Returns:
            Dict of parameter names to values.
        """
        if not self._has_custom_columns:
            return {
                "node_id": item["node_id"],
                "text": item["text"],
                "embedding": json.dumps(item["embedding"]),
                "metadata": json.dumps(item["metadata"]),
            }

        params: Dict[str, Any] = {
            "node_id": item["node_id"],
            "text": item["text"],
        }
        if self._metadata_json_column is not None:
            # Keys mapped to dedicated columns are excluded from JSON
            remaining = {
                k: v
                for k, v in item["metadata"].items()
                if k not in self._metadata_column_names
            }
            params["metadata"] = json.dumps(remaining)
        params["embedding"] = json.dumps(item["embedding"])

        # Metadata column values with NOT NULL validation
        col_obj_map = {c.name: c for c in self._metadata_column_objs}
        meta = item.get("metadata", {})
        for name in self._metadata_column_names:
            val = meta.get(name)
            if val is None:
                col = col_obj_map.get(name)
                if col is not None and not col.nullable:
                    raise ValueError(
                        f"Column '{name}' is NOT NULL but no value "
                        f"was provided in metadata. Provide a value "
                        f"for '{name}' in the metadata dict, or set "
                        f"nullable=True on the Column definition. "
                        f"Note: Column.default only affects DDL schema "
                        f"definition and does not auto-fill missing "
                        f"values during INSERT."
                    )
            params[f"meta_{name}"] = val
        return params

    # ------------------------------------------------------------------
    # SQL builder methods for SELECT (similarity search)
    # ------------------------------------------------------------------

    def _build_select_columns(self) -> str:
        """Build the SELECT column list with stable aliases.

        Core columns (node_id, text, metadata) are aliased so that
        result-mapping code can always access them as ``item[0]``,
        ``item[1]``, ``item[2]`` regardless of the actual column names.
        Metadata columns are appended after the core columns.

        Returns:
            Column list string for use in a SELECT clause.
        """
        if not self._has_custom_columns:
            return "node_id, text, metadata"

        cols: List[str] = [
            f"`{self._node_id_column}` AS `node_id`",
            f"`{self._text_column}` AS `text`",
        ]
        if self._metadata_json_column is not None:
            cols.append(
                f"`{self._metadata_json_column}` AS `metadata`"
            )
        for name in self._metadata_column_names:
            cols.append(f"`{name}`")
        return ", ".join(cols)

    def _build_search_sql(
        self,
        distance_func: str,
        index_hint: str,
        where_clause: str,
    ) -> str:
        """Build the similarity search SQL.

        Args:
            distance_func: Distance function name (e.g. VEC_DISTANCE).
            index_hint: Index hint clause (e.g. "/*+ ... */").
            where_clause: Optional WHERE clause (may be empty).

        Returns:
            The complete SELECT ... ORDER BY ... LIMIT SQL.
        """
        select_cols = self._build_select_columns()
        emb_col = self._embedding_column

        return (
            f"SELECT\n"
            f"    {select_cols},\n"
            f"    {distance_func}(`{emb_col}`, "
            f"VEC_FROMTEXT(:query_embedding)) AS distance\n"
            f"FROM `{self.table_name}`{index_hint}\n"
            f"{where_clause}\n"
            f"ORDER BY distance\n"
            f"LIMIT :limit"
        )

    def _build_get_nodes_select(self) -> str:
        """Build SELECT columns for get_nodes/search_by_metadata.

        Returns the column list for a metadata-only SELECT (no
        vector distance).  The first three columns are always
        node_id, text, metadata (aliased).
        """
        return self._build_select_columns()

    def _record_to_metadata(self, row: Any) -> dict:
        """Reconstruct metadata dict from a database row.

        For the default schema, simply deserializes the JSON column.
        For custom columns, merges the JSON column (if present) with
        the mapped metadata column values from the row.

        Args:
            row: A SQLAlchemy Row object.  The first columns are
                node_id (0), text (1), then optionally metadata (2),
                and any metadata columns follow.

        Returns:
            The metadata dictionary.
        """
        if not self._has_custom_columns:
            return self._parse_metadata(row[2])

        metadata: dict = {}
        # Determine the starting index for metadata columns
        if self._metadata_json_column is not None:
            json_data = row[2]  # aliased as 'metadata'
            metadata.update(self._parse_metadata(json_data))
            meta_start = 3
        else:
            meta_start = 2
        # Mapped column values override
        for i, name in enumerate(self._metadata_column_names):
            val = row[meta_start + i]
            if val is not None:
                metadata[name] = val
        return metadata

    def _validate_embedding_dimensions(
        self, nodes: Sequence[BaseNode]
    ) -> None:
        """Validate that all node embeddings match the expected dimension.

        Checks each node's embedding length against ``self.embed_dim``.
        On newer versions, additionally cross-checks with DN's
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
                    "VECTOR_DIM cross-check failed: %s",
                    self._sanitize_error(e),
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

        When custom columns are enabled, if the filter key matches a
        mapped metadata column name, the filter uses a direct column
        reference instead of JSON_EXTRACT.
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
        elif filter_.value is None:
            # L2: None values require IS NULL / IS NOT NULL
            if filter_.operator not in (FilterOperator.EQ, FilterOperator.NE):
                raise ValueError(
                    f"Filter '{filter_.key}' uses operator "
                    f"{filter_.operator} with None value. "
                    f"Only EQ and NE support None values."
                )
            filter_value = None
        else:
            param_name = f"param_{global_param_counter[0]}"
            global_param_counter[0] += 1
            filter_value = f":{param_name}"
            params[param_name] = filter_.value

        # Determine the left-hand side of the filter expression
        if self._has_custom_columns:
            if filter_.key in self._metadata_column_names:
                # Mapped column: direct column reference
                lhs = f"`{filter_.key}`"
            elif self._metadata_json_column is not None:
                # JSON column: use JSON_EXTRACT on custom column
                lhs = (
                    f"JSON_UNQUOTE(JSON_EXTRACT("
                    f"`{self._metadata_json_column}`, "
                    f"'$.{filter_.key}'))"
                )
            else:
                raise ValueError(
                    f"Cannot filter on '{filter_.key}': no JSON "
                    f"metadata column configured and "
                    f"'{filter_.key}' is not a mapped metadata "
                    f"column. Either set metadata_json_column to "
                    f"a column name, or add '{filter_.key}' to "
                    f"metadata_columns."
                )
        else:
            # Default schema: use hardcoded metadata column
            lhs = (
                f"JSON_UNQUOTE(JSON_EXTRACT(metadata, "
                f"'$.{filter_.key}'))"
            )

        if filter_value is None:
            # L2: None value — use IS NULL / IS NOT NULL
            if filter_.operator == FilterOperator.EQ:
                clause = f"{lhs} IS NULL"
            else:
                clause = f"{lhs} IS NOT NULL"
        else:
            clause = f"{lhs} {self._to_mysql_operator(filter_.operator)} {filter_value}"
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

    def _reconstruct_node(
        self, metadata: dict, text: str, node_id: Optional[str] = None
    ) -> BaseNode:
        """Reconstruct a node from metadata and text.

        Tries ``metadata_dict_to_node`` first (full reconstruction
        using internal serialization keys like ``_node_content``).
        Falls back to creating a ``TextNode`` directly when those
        keys are absent — this happens when the store has no JSON
        metadata column and only mapped columns are available.

        Args:
            metadata: The metadata dict from the database row.
            text: The text content from the database row.
            node_id: Optional node_id to set on the reconstructed node.

        Returns:
            A ``BaseNode`` with content and metadata populated.
        """
        if metadata and "_node_content" in metadata:
            node = metadata_dict_to_node(metadata)
        else:
            # No internal serialization keys — create a basic TextNode.
            # This happens in no-JSON-column mode where only mapped
            # metadata columns are stored.
            from llama_index.core.schema import TextNode

            node = TextNode(
                id_=node_id or metadata.get("node_id", ""),
                text=text,
                metadata={
                    k: v for k, v in metadata.items()
                    if not k.startswith("_")
                },
            )
        node.set_content(text if text is not None else "")
        if node_id is not None:
            node.node_id = str(node_id)
        return node

    def _db_rows_to_query_result(
        self, rows: List[DBEmbeddingRow]
    ) -> VectorStoreQueryResult:
        """Convert DB rows to a VectorStoreQueryResult."""
        nodes = []
        similarities = []
        ids = []
        for db_row in rows:
            node = self._reconstruct_node(
                metadata=db_row.metadata,
                text=db_row.text,
                node_id=db_row.node_id,
            )
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

        The conversion depends on ``distance_strategy``:
        - COSINE: ``1 - distance``  (range [-1, 1])
        - EUCLIDEAN: ``1 / (1 + distance)``  (range (0, 1])
        - INNER_PRODUCT: ``-distance``  (distance is -dot_product)
        """
        if self.distance_strategy == "cosine":
            return 1.0 - distance
        elif self.distance_strategy == "euclidean":
            return 1.0 / (1.0 + max(0.0, distance))
        elif self.distance_strategy == "inner_product":
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
                # W12: Use DEBUG to avoid leaking sensitive metadata
                _logger.debug(
                    "Corrupted metadata JSON in database row, "
                    "using empty dict. Raw length: %d",
                    len(raw),
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

            # W3: Guard against NaN/Inf in embeddings causing infinite
            # loop — if no candidate scored above -inf, break.
            if best_idx == -1:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    def _fetch_embeddings_by_node_ids(
        self, node_ids: List[str]
    ) -> Dict[str, List[float]]:
        """Fetch stored embedding vectors by node_id.

        Uses ``VEC_TOTEXT`` on newer versions, ``CAST(embedding AS CHAR)``
        on old versions.

        Args:
            node_ids: List of node_id values to fetch.

        Returns:
            Dict mapping node_id to its embedding vector.
        """
        if not node_ids:
            return {}

        emb_expr = (
            f"VEC_TOTEXT(`{self._embedding_column}`)"
            if self._capabilities.get("vec_totext", False)
            else f"CAST(`{self._embedding_column}` AS CHAR)"
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
                f"SELECT `{self._node_id_column}`, {emb_expr} AS emb_str "
                f"FROM `{self.table_name}` "
                f"WHERE `{self._node_id_column}` IN ({placeholders})"
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
                # W5: Sanitize credential info from error before logging
                _logger.warning(
                    "Failed to fetch embeddings: %s",
                    self._sanitize_error(e),
                )
        return result

    async def _afetch_embeddings_by_node_ids(
        self, node_ids: List[str]
    ) -> Dict[str, List[float]]:
        """Async fetch stored embedding vectors by node_id."""
        if not node_ids:
            return {}

        emb_expr = (
            f"VEC_TOTEXT(`{self._embedding_column}`)"
            if self._capabilities.get("vec_totext", False)
            else f"CAST(`{self._embedding_column}` AS CHAR)"
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
                f"SELECT `{self._node_id_column}`, {emb_expr} AS emb_str "
                f"FROM `{self.table_name}` "
                f"WHERE `{self._node_id_column}` IN ({placeholders})"
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
                # W5: Sanitize credential info from error before logging
                _logger.warning(
                    "Failed to fetch embeddings: %s",
                    self._sanitize_error(e),
                )
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
                        self._vector_index_name = self._validate_identifier(row[0])
                        return self._vector_index_name
            except Exception as e:
                _logger.debug(
                    "VECTOR_INDEXES query failed: %s",
                    self._sanitize_error(e),
                )

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
                        self._vector_index_name = self._validate_identifier(m.group(1))
                        return self._vector_index_name
        except Exception as e:
            _logger.debug(
                "Failed to detect vector index name: %s",
                self._sanitize_error(e),
            )
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
        is_partitioned = self._has_partition
        ids: List[str] = []
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            with self._session() as session:
                for node in batch:
                    ids.append(node.node_id)
                    item = self._node_to_table_row(node)

                    if is_partitioned:
                        # W2: Partitioned tables downgrade node_id unique
                        # index to regular INDEX, so ON DUPLICATE KEY
                        # UPDATE cannot detect duplicates. Use
                        # DELETE-then-INSERT to preserve upsert semantics.
                        del_stmt = sqlalchemy.text(
                            f"DELETE FROM `{self.table_name}` "
                            f"WHERE `{self._node_id_column}` = :node_id"
                        )
                        session.execute(
                            del_stmt, {"node_id": item["node_id"]}
                        )
                    stmt = sqlalchemy.text(
                        self._build_upsert_sql(partitioned=is_partitioned)
                    )
                    session.execute(
                        stmt,
                        self._build_upsert_params(item),
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
        is_partitioned = self._has_partition
        ids: List[str] = []
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            async with self._async_session() as session:
                for node in batch:
                    ids.append(node.node_id)
                    item = self._node_to_table_row(node)

                    if is_partitioned:
                        # W2: DELETE-then-INSERT for partitioned tables
                        del_stmt = sqlalchemy.text(
                            f"DELETE FROM `{self.table_name}` "
                            f"WHERE `{self._node_id_column}` = :node_id"
                        )
                        await session.execute(
                            del_stmt, {"node_id": item["node_id"]}
                        )
                    stmt = sqlalchemy.text(
                        self._build_upsert_sql(partitioned=is_partitioned)
                    )
                    await session.execute(
                        stmt,
                        self._build_upsert_params(item),
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
            where_clause = f"WHERE {where_clause}" if where_clause else ""
            params.update(filter_params)

        stmt = sqlalchemy.text(
            self._build_search_sql(
                distance_func=distance_func,
                index_hint=index_hint,
                where_clause=where_clause,
            )
        )

        with self._session() as session:
            self._set_ef_search_sync(session, ef_search)
            result = session.execute(stmt, params)
            results = result.fetchall()

        rows = []
        for item in results:
            meta = self._record_to_metadata(item)
            rows.append(
                DBEmbeddingRow(
                    node_id=item[0],
                    text=item[1],
                    metadata=meta,
                    similarity=self._distance_to_similarity(item[-1])
                    if item[-1] is not None
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
            where_clause = f"WHERE {where_clause}" if where_clause else ""
            params.update(filter_params)

        stmt = sqlalchemy.text(
            self._build_search_sql(
                distance_func=distance_func,
                index_hint=index_hint,
                where_clause=where_clause,
            )
        )

        async with self._async_session() as session:
            await self._set_ef_search_async(session, ef_search)
            result = await session.execute(stmt, params)
            results = result.fetchall()

        rows = []
        for item in results:
            meta = self._record_to_metadata(item)
            rows.append(
                DBEmbeddingRow(
                    node_id=item[0],
                    text=item[1],
                    metadata=meta,
                    similarity=self._distance_to_similarity(item[-1])
                    if item[-1] is not None
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
                        f"SELECT {self._build_get_nodes_select()} FROM `{self.table_name}` "
                        f"WHERE `{self._node_id_column}` IN ({placeholders})"
                    )
                    result = session.execute(stmt, params)
                    for item in result:
                        meta = self._record_to_metadata(item)
                        node = self._reconstruct_node(
                            meta, item[1] if item[1] is not None else "", item[0]
                        )
                        nodes.append(node)
        elif filters and filters.filters:
            global_param_counter = [0]
            where_clause, filter_params = self._filters_to_where_clause(
                filters, global_param_counter
            )
            stmt = sqlalchemy.text(
                f"SELECT {self._build_get_nodes_select()} FROM `{self.table_name}` "
                f"WHERE {where_clause} LIMIT 10000"
            )
            _logger.warning(
                "get_nodes(filters=...) results capped "
                "at 10000 rows."
            )
            with self._session() as session:
                result = session.execute(stmt, filter_params)
                for item in result:
                    meta = self._record_to_metadata(item)
                    node = self._reconstruct_node(
                        meta, item[1] if item[1] is not None else "", item[0]
                    )
                    nodes.append(node)
        else:
            stmt = sqlalchemy.text(
                f"SELECT {self._build_get_nodes_select()} FROM `{self.table_name}` "
                f"LIMIT 10000"
            )
            _logger.warning(
                "get_nodes() called without node_ids or filters; "
                "results capped at 10000 rows."
            )
            with self._session() as session:
                result = session.execute(stmt)
                for item in result:
                    meta = self._record_to_metadata(item)
                    node = self._reconstruct_node(
                        meta, item[1] if item[1] is not None else "", item[0]
                    )
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
                        f"SELECT {self._build_get_nodes_select()} FROM `{self.table_name}` "
                        f"WHERE `{self._node_id_column}` IN ({placeholders})"
                    )
                    result = await session.execute(stmt, params)
                    for item in result:
                        meta = self._record_to_metadata(item)
                        node = self._reconstruct_node(
                            meta, item[1] if item[1] is not None else "", item[0]
                        )
                        nodes.append(node)
        elif filters and filters.filters:
            global_param_counter = [0]
            where_clause, filter_params = self._filters_to_where_clause(
                filters, global_param_counter
            )
            stmt = sqlalchemy.text(
                f"SELECT {self._build_get_nodes_select()} FROM `{self.table_name}` "
                f"WHERE {where_clause} LIMIT 10000"
            )
            _logger.warning(
                "get_nodes(filters=...) results capped "
                "at 10000 rows."
            )
            async with self._async_session() as session:
                result = await session.execute(stmt, filter_params)
                for item in result:
                    meta = self._record_to_metadata(item)
                    node = self._reconstruct_node(
                        meta, item[1] if item[1] is not None else "", item[0]
                    )
                    nodes.append(node)
        else:
            stmt = sqlalchemy.text(
                f"SELECT {self._build_get_nodes_select()} FROM `{self.table_name}` "
                f"LIMIT 10000"
            )
            _logger.warning(
                "aget_nodes() called without node_ids or filters; "
                "results capped at 10000 rows."
            )
            async with self._async_session() as session:
                result = await session.execute(stmt)
                for item in result:
                    meta = self._record_to_metadata(item)
                    node = self._reconstruct_node(
                        meta, item[1] if item[1] is not None else "", item[0]
                    )
                    nodes.append(node)

        return nodes

    # ------------------------------------------------------------------
    # CRUD: delete
    # ------------------------------------------------------------------

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id (stored in metadata)."""
        self._initialize()

        # S2: Use direct column reference when ref_doc_id is mapped
        if "ref_doc_id" in self._metadata_column_names:
            where_clause = f"`ref_doc_id` = :doc_id"
        elif self._metadata_json_column is not None:
            where_clause = (
                f"JSON_UNQUOTE(JSON_EXTRACT("
                f"`{self._metadata_json_column}`, "
                f"'$.ref_doc_id')) = :doc_id"
            )
        else:
            raise ValueError(
                "delete(ref_doc_id) requires a JSON metadata column or "
                "a 'ref_doc_id' mapped column. When metadata_json_column=None, "
                "use delete_nodes(filters=...) or delete_by_metadata(filters=...) "
                "instead, or add 'ref_doc_id' to metadata_columns to filter "
                "on a dedicated column."
            )

        with self._session() as session:
            stmt = sqlalchemy.text(
                f"DELETE FROM `{self.table_name}` "
                f"WHERE {where_clause}"
            )
            session.execute(stmt, {"doc_id": ref_doc_id})
            session.commit()

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async delete nodes by ref_doc_id."""
        self._initialize()

        # S2: Use direct column reference when ref_doc_id is mapped
        if "ref_doc_id" in self._metadata_column_names:
            where_clause = f"`ref_doc_id` = :doc_id"
        elif self._metadata_json_column is not None:
            where_clause = (
                f"JSON_UNQUOTE(JSON_EXTRACT("
                f"`{self._metadata_json_column}`, "
                f"'$.ref_doc_id')) = :doc_id"
            )
        else:
            raise ValueError(
                "adelete(ref_doc_id) requires a JSON metadata column or "
                "a 'ref_doc_id' mapped column. When metadata_json_column=None, "
                "use adelete_nodes(filters=...) or adelete_by_metadata(filters=...) "
                "instead, or add 'ref_doc_id' to metadata_columns to filter "
                "on a dedicated column."
            )

        async with self._async_session() as session:
            stmt = sqlalchemy.text(
                f"DELETE FROM `{self.table_name}` "
                f"WHERE {where_clause}"
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
                        f"WHERE `{self._node_id_column}` IN ({placeholders})"
                    )
                    session.execute(stmt, params)
                session.commit()
            elif filters and filters.filters:
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
            else:
                _logger.warning(
                    "delete_nodes() called without node_ids or "
                    "filters; no rows deleted."
                )

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
                        f"WHERE `{self._node_id_column}` IN ({placeholders})"
                    )
                    await session.execute(stmt, params)
                await session.commit()
            elif filters and filters.filters:
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
            else:
                _logger.warning(
                    "adelete_nodes() called without node_ids or "
                    "filters; no rows deleted."
                )

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

        # M5: Validate limit parameter
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(
                f"limit must be a positive integer, got {limit}"
            )

        if not filters.filters:
            raise ValueError(
                "filters cannot be empty — provide at least one "
                "MetadataFilter condition."
            )

        global_param_counter = [0]
        where_clause, filter_params = self._filters_to_where_clause(
            filters, global_param_counter
        )

        stmt = sqlalchemy.text(
            f"SELECT {self._build_get_nodes_select()} "
            f"FROM `{self.table_name}` "
            f"WHERE {where_clause} LIMIT :limit"
        )
        filter_params["limit"] = limit

        with self._session() as session:
            result = session.execute(stmt, filter_params)
            nodes: List[BaseNode] = []
            for row in result:
                meta = self._record_to_metadata(row)
                node = self._reconstruct_node(
                    meta or {}, row[1] if row[1] is not None else "", str(row[0])
                )
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

        # M5: Validate limit parameter
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(
                f"limit must be a positive integer, got {limit}"
            )

        if not filters.filters:
            raise ValueError(
                "filters cannot be empty — provide at least one "
                "MetadataFilter condition."
            )

        global_param_counter = [0]
        where_clause, filter_params = self._filters_to_where_clause(
            filters, global_param_counter
        )

        stmt = sqlalchemy.text(
            f"SELECT {self._build_get_nodes_select()} "
            f"FROM `{self.table_name}` "
            f"WHERE {where_clause} LIMIT :limit"
        )
        filter_params["limit"] = limit

        async with self._async_session() as session:
            result = await session.execute(stmt, filter_params)
            nodes: List[BaseNode] = []
            for row in result:
                meta = self._record_to_metadata(row)
                node = self._reconstruct_node(
                    meta or {}, row[1] if row[1] is not None else "", str(row[0])
                )
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

        if not filters.filters:
            raise ValueError(
                "filters cannot be empty — provide at least one "
                "MetadataFilter condition."
            )

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

        if not filters.filters:
            raise ValueError(
                "filters cannot be empty — provide at least one "
                "MetadataFilter condition."
            )

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
        # L7: Reset engine/session references for consistency with
        # _cleanup_engines(), so that any stale access raises clearly
        # instead of using a disposed engine.
        self._engine = None
        self._async_engine = None
        self._session = None
        self._async_session = None

    async def aclose(self) -> None:
        """Async close engines."""
        if not self._is_initialized:
            return
        if self._engine:
            self._engine.dispose()
        if self._async_engine:
            await self._async_engine.dispose()
        self._is_initialized = False
        self._engine = None
        self._async_engine = None
        self._session = None
        self._async_session = None

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
            distance: Distance function (``"cosine"``, ``"euclidean"``,
                or ``"inner_product"``). Defaults to the store's strategy.
                ``inner_product`` requires a newer PolarDB-X version.
            ef_construction: HNSW build-time candidate list size (5-1000).
                Newer versions only; silently ignored on older versions.
        """
        self._initialize()
        self._validate_identifier(index_name)
        # S1: Validate m and ef_construction to prevent SQL injection
        m_val = m or self.default_m
        self._validate_positive_int(m_val, "m")
        if not (3 <= m_val <= 200):
            raise ValueError(f"m must be 3-200, got {m_val}")
        dist_val = (distance or self.distance_strategy).upper()
        if dist_val not in ("COSINE", "EUCLIDEAN", "INNER_PRODUCT"):
            raise ValueError(
                f"Invalid distance function: {dist_val}. "
                "Must be 'cosine', 'euclidean', or 'inner_product'."
            )
        if (
            dist_val == "INNER_PRODUCT"
            and not self._capabilities.get("vec_dim", False)
        ):
            raise NotSupportedError(
                "distance='inner_product' requires a newer "
                "PolarDB-X version. Use 'cosine' or 'euclidean' for "
                "older versions."
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
                    f"ADD VECTOR INDEX `{index_name}` (`{self._embedding_column}`) "
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
        dist_val = (distance or self.distance_strategy).upper()
        if dist_val not in ("COSINE", "EUCLIDEAN", "INNER_PRODUCT"):
            raise ValueError(
                f"Invalid distance function: {dist_val}. "
                "Must be 'cosine', 'euclidean', or 'inner_product'."
            )
        if (
            dist_val == "INNER_PRODUCT"
            and not self._capabilities.get("vec_dim", False)
        ):
            raise NotSupportedError(
                "distance='inner_product' requires a newer "
                "PolarDB-X version. Use 'cosine' or 'euclidean' for "
                "older versions."
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
                    f"ADD VECTOR INDEX `{index_name}` (`{self._embedding_column}`) "
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
        """Raise NotSupportedError if newer version capabilities are not available.

        Uses ``vec_dim`` (VECTOR_DIM) as the newer version indicator —
        present iff the instance has newer vector features enabled.
        """
        if not self._capabilities.get("vec_dim", False):
            raise NotSupportedError(
                f"{feature} requires a newer PolarDB-X version with "
                "vector index support. Current instance does not support "
                "these vector features."
            )

    def preload_index(self) -> None:
        """Preload the HNSW vector index into memory cache (newer versions only).

        Loads the entire HNSW auxiliary table graph into the shared
        cache to eliminate cold-start latency on the first query.
        """
        self._initialize()
        self._require_v3("preload_index()")
        if not self._capabilities.get("dbms_vidx", False):
            raise NotSupportedError(
                "preload_index() requires dbms_vidx stored procedures "
                "which are not available on this instance. This may occur "
                "on instances where VECTOR_DIM is present but dbms_vidx "
                "procedures are not installed."
            )
        # S2: Re-validate mutable Pydantic fields before SQL interpolation
        self._validate_identifier(self.database)
        self._validate_identifier(self.table_name)
        with self._session() as session:
            result = session.execute(
                sqlalchemy.text(
                    f"CALL dbms_vidx.preload("
                    f"'{self.database}', '{self.table_name}', '{self._embedding_column}')"
                )
            )
            result.fetchall()
            session.commit()
        _logger.info(
            "Preloaded vector index for table %s", self.table_name
        )

    async def apreload_index(self) -> None:
        """Async preload the HNSW vector index (newer versions only)."""
        self._initialize()
        self._require_v3("preload_index()")
        if not self._capabilities.get("dbms_vidx", False):
            raise NotSupportedError(
                "apreload_index() requires dbms_vidx stored procedures "
                "which are not available on this instance. This may occur "
                "on instances where VECTOR_DIM is present but dbms_vidx "
                "procedures are not installed."
            )
        # S2: Re-validate mutable Pydantic fields before SQL interpolation
        self._validate_identifier(self.database)
        self._validate_identifier(self.table_name)
        async with self._async_session() as session:
            result = await session.execute(
                sqlalchemy.text(
                    f"CALL dbms_vidx.preload("
                    f"'{self.database}', '{self.table_name}', '{self._embedding_column}')"
                )
            )
            result.fetchall()
            await session.commit()
        _logger.info(
            "Preloaded vector index for table %s", self.table_name
        )

    def preload_check(self) -> Dict[str, Any]:
        """Check if preloading would fit in cache (newer versions only).

        Returns:
            Dictionary with check results (rows, memory estimate, etc.).
        """
        self._initialize()
        self._require_v3("preload_check()")
        if not self._capabilities.get("dbms_vidx", False):
            raise NotSupportedError(
                "preload_check() requires dbms_vidx stored procedures "
                "which are not available on this instance. This may occur "
                "on instances where VECTOR_DIM is present but dbms_vidx "
                "procedures are not installed."
            )
        # S2: Re-validate mutable Pydantic fields before SQL interpolation
        self._validate_identifier(self.database)
        self._validate_identifier(self.table_name)
        with self._session() as session:
            result = session.execute(
                sqlalchemy.text(
                    f"CALL dbms_vidx.preload_check("
                    f"'{self.database}', '{self.table_name}', '{self._embedding_column}')"
                )
            )
            rows = result.fetchall()
            session.commit()
        if not rows:
            return {}
        return {str(idx): dict(row._mapping) for idx, row in enumerate(rows)}

    async def apreload_check(self) -> Dict[str, Any]:
        """Async check if preloading would fit in cache (newer versions only)."""
        self._initialize()
        self._require_v3("preload_check()")
        if not self._capabilities.get("dbms_vidx", False):
            raise NotSupportedError(
                "apreload_check() requires dbms_vidx stored procedures "
                "which are not available on this instance. This may occur "
                "on instances where VECTOR_DIM is present but dbms_vidx "
                "procedures are not installed."
            )
        # S2: Re-validate mutable Pydantic fields before SQL interpolation
        self._validate_identifier(self.database)
        self._validate_identifier(self.table_name)
        async with self._async_session() as session:
            result = await session.execute(
                sqlalchemy.text(
                    f"CALL dbms_vidx.preload_check("
                    f"'{self.database}', '{self.table_name}', '{self._embedding_column}')"
                )
            )
            rows = result.fetchall()
            await session.commit()
        if not rows:
            return {}
        return {str(idx): dict(row._mapping) for idx, row in enumerate(rows)}

    def explain_index_health(self) -> Dict[str, Any]:
        """Check vector index health and return diagnostics (newer versions only).

        Combines information_schema.VECTOR_INDEXES metadata with
        EXPLAIN and EXPLAIN ANALYZE output to provide a comprehensive
        health report.

        Returns:
            Dictionary with keys:
                - index_info: VECTOR_INDEXES metadata (name, algorithm,
                  metric, dimension, M, EF_CONSTRUCTION, etc.)
                - explain: plain EXPLAIN output (index selection info)
                - explain_analyze: EXPLAIN ANALYZE output with actual
                  nodes_visited cost (newer versions only)
        """
        self._initialize()
        self._require_v3("explain_index_health()")
        if not self._capabilities.get("vector_indexes_view", False):
            raise NotSupportedError(
                "explain_index_health() requires "
                "information_schema.VECTOR_INDEXES which is not "
                "available on this instance."
            )
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
                    f"EXPLAIN SELECT `{self._id_column}` FROM `{self.table_name}` "
                    f"ORDER BY {dist_func}(`{self._embedding_column}`, "
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
                        f"SELECT `{self._id_column}` FROM `{self.table_name}` "
                        f"ORDER BY {dist_func}(`{self._embedding_column}`, "
                        f"VEC_FROMTEXT('{sample_vec}')) LIMIT 10"
                    )
                )
                result["explain_analyze"] = [
                    dict(r._mapping) for r in result_set.fetchall()
                ]
            except Exception as e:
                _logger.debug(
                    "EXPLAIN ANALYZE failed (may not be supported): %s",
                    self._sanitize_error(e),
                )
                result["explain_analyze"] = None

        return result

    async def aexplain_index_health(self) -> Dict[str, Any]:
        """Async check vector index health (newer versions only)."""
        self._initialize()
        self._require_v3("explain_index_health()")
        if not self._capabilities.get("vector_indexes_view", False):
            raise NotSupportedError(
                "aexplain_index_health() requires "
                "information_schema.VECTOR_INDEXES which is not "
                "available on this instance."
            )
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
                    f"EXPLAIN SELECT `{self._id_column}` FROM `{self.table_name}` "
                    f"ORDER BY {dist_func}(`{self._embedding_column}`, "
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
                        f"SELECT `{self._id_column}` FROM `{self.table_name}` "
                        f"ORDER BY {dist_func}(`{self._embedding_column}`, "
                        f"VEC_FROMTEXT('{sample_vec}')) LIMIT 10"
                    )
                )
                result["explain_analyze"] = [
                    dict(r._mapping) for r in result_set.fetchall()
                ]
            except Exception as e:
                _logger.debug(
                    "EXPLAIN ANALYZE failed (may not be supported): %s",
                    self._sanitize_error(e),
                )
                result["explain_analyze"] = None

        return result
