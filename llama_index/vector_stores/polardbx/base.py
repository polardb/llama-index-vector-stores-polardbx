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

import json
import logging
import re
from typing import (
    Any,
    ClassVar,
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
from llama_index.core.bridge.pydantic import PrivateAttr
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

    stores_text: ClassVar[bool] = True
    flat_metadata: ClassVar[bool] = False

    connection_string: str
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
    _ef_construction: Optional[int] = PrivateAttr(default=None)

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

    # ------------------------------------------------------------------
    # Constructor & factory
    # ------------------------------------------------------------------

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
        """
        self._validate_identifier(table_name)
        self._validate_identifier(database)
        self._validate_positive_int(embed_dim, "embed_dim")
        self._validate_positive_int(default_m, "default_m")

        if distance_method not in ("EUCLIDEAN", "COSINE", "INNER_PRODUCT"):
            raise ValueError(
                f"Invalid distance_method: {distance_method}. "
                "Must be 'COSINE', 'EUCLIDEAN', or 'INNER_PRODUCT'."
            )

        password_safe = quote_plus(password)
        connection_string = (
            f"mysql+pymysql://{user}:{password_safe}@{host}:{port}/{database}"
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
        self._ef_construction = ef_construction

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

        self._engine = create_engine(
            self.connection_string,
            echo=self.debug,
        )

        async_connection_string = self.connection_string.replace(
            "mysql+pymysql://", "mysql+aiomysql://"
        )
        self._async_engine = create_async_engine(
            async_connection_string,
            echo=self.debug,
        )

        self._session = sessionmaker(self._engine)
        self._async_session = sessionmaker(
            self._async_engine, class_=AsyncSession
        )

    def _initialize(self) -> None:
        """Connect, detect capabilities, and optionally create table."""
        if not self._is_initialized:
            self._connect()
            if self.perform_setup:
                self._detect_capabilities()
                self._validate_distance_method()
                self._create_table_if_not_exists()
            self._is_initialized = True

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

        stmt = text(f"""
        CREATE TABLE IF NOT EXISTS `{self.table_name}` (
            id VARCHAR(36) PRIMARY KEY,
            node_id VARCHAR(255) NOT NULL,
            text LONGTEXT,
            metadata JSON,
            embedding VECTOR({self.embed_dim}) NOT NULL,
            UNIQUE INDEX `node_id_index` (node_id),
            VECTOR INDEX `vi` (embedding) M={self.default_m}{ef_clause} DISTANCE={self.distance_method}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """)
        with self._session() as session:
            session.execute(stmt)
            session.commit()

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
            _logger.warning(
                "Unsupported operator: %s, fallback to '='", operator
            )
            return "="
        return mapping[operator]

    def _build_filter_clause(
        self, filter_: MetadataFilter, global_param_counter: List[int]
    ) -> tuple[str, dict]:
        """Build a single filter clause.

        Uses JSON_UNQUOTE(JSON_EXTRACT(...)) instead of JSON_VALUE
        because PolarDB-X does not support the JSON_VALUE function.
        """
        params: Dict[str, Any] = {}

        if filter_.operator in [FilterOperator.IN, FilterOperator.NIN]:
            placeholders = []
            for i in range(len(filter_.value)):
                param_name = f"param_{global_param_counter[0]}"
                global_param_counter[0] += 1
                placeholders.append(f":{param_name}")
                params[param_name] = filter_.value[i]
            filter_value = f"({','.join(placeholders)})"
        elif isinstance(filter_.value, (list, tuple)):
            placeholders = []
            for i in range(len(filter_.value)):
                param_name = f"param_{global_param_counter[0]}"
                global_param_counter[0] += 1
                placeholders.append(f":{param_name}")
                params[param_name] = filter_.value[i]
            filter_value = f"({','.join(placeholders)})"
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
    # Index hint & ef_search helpers
    # ------------------------------------------------------------------

    def _detect_vector_index_name(self) -> Optional[str]:
        """Auto-detect the vector index name on the table.

        v3 path: query information_schema.VECTOR_INDEXES.
        Fallback: parse SHOW CREATE TABLE with regex.
        """
        if self._vector_index_name is not None:
            return self._vector_index_name

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
            session.execute(
                sqlalchemy.text(
                    f"SET SESSION vidx_hnsw_ef_search = {int(ef_search)}"
                )
            )

    async def _set_ef_search_async(
        self, session: Any, ef_search: Optional[int]
    ) -> None:
        """Set ef_search session variable (async)."""
        if ef_search is not None:
            await session.execute(
                sqlalchemy.text(
                    f"SET SESSION vidx_hnsw_ef_search = {int(ef_search)}"
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
        """
        self._initialize()

        if not nodes:
            return []

        ids: List[str] = []
        with self._session() as session:
            for node in nodes:
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
        """Async add nodes to the vector store."""
        self._initialize()

        if not nodes:
            return []

        ids: List[str] = []
        async with self._async_session() as session:
            for node in nodes:
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
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"Query mode {query.mode} not available.")

        self._initialize()

        ef_search = kwargs.get("ef_search")
        search_type = kwargs.get("search_type")

        distance_func = self._get_distance_func()
        index_hint = self._build_index_hint(search_type)

        where_clause = ""
        params: Dict[str, Any] = {
            "query_embedding": json.dumps(query.query_embedding),
            "limit": query.similarity_top_k,
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
            meta = item[2]
            if isinstance(meta, str):
                meta = json.loads(meta)
            rows.append(
                DBEmbeddingRow(
                    node_id=item[0],
                    text=item[1],
                    metadata=meta,
                    similarity=(1 - item[3]) if item[3] is not None else 0,
                )
            )

        return self._db_rows_to_query_result(rows)

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Async query the vector store."""
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError(f"Query mode {query.mode} not available.")

        self._initialize()

        ef_search = kwargs.get("ef_search")
        search_type = kwargs.get("search_type")

        distance_func = self._get_distance_func()
        index_hint = self._build_index_hint(search_type)

        where_clause = ""
        params: Dict[str, Any] = {
            "query_embedding": json.dumps(query.query_embedding),
            "limit": query.similarity_top_k,
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
            meta = item[2]
            if isinstance(meta, str):
                meta = json.loads(meta)
            rows.append(
                DBEmbeddingRow(
                    node_id=item[0],
                    text=item[1],
                    metadata=meta,
                    similarity=(1 - item[3]) if item[3] is not None else 0,
                )
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
            placeholders = ",".join(
                [f":node_id_{i}" for i in range(len(node_ids))]
            )
            params = {
                f"node_id_{i}": nid for i, nid in enumerate(node_ids)
            }
            stmt = sqlalchemy.text(
                f"SELECT text, metadata FROM `{self.table_name}` "
                f"WHERE node_id IN ({placeholders})"
            )
            with self._session() as session:
                result = session.execute(stmt, params)
                for item in result:
                    meta = item[1]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
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
                    meta = item[1]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    node = metadata_dict_to_node(meta)
                    node.set_content(str(item[0]))
                    nodes.append(node)
        else:
            stmt = sqlalchemy.text(
                f"SELECT text, metadata FROM `{self.table_name}`"
            )
            with self._session() as session:
                result = session.execute(stmt)
                for item in result:
                    meta = item[1]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
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
            placeholders = ",".join(
                [f":node_id_{i}" for i in range(len(node_ids))]
            )
            params = {
                f"node_id_{i}": nid for i, nid in enumerate(node_ids)
            }
            stmt = sqlalchemy.text(
                f"SELECT text, metadata FROM `{self.table_name}` "
                f"WHERE node_id IN ({placeholders})"
            )
            async with self._async_session() as session:
                result = await session.execute(stmt, params)
                for item in result:
                    meta = item[1]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
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
                    meta = item[1]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    node = metadata_dict_to_node(meta)
                    node.set_content(str(item[0]))
                    nodes.append(node)
        else:
            stmt = sqlalchemy.text(
                f"SELECT text, metadata FROM `{self.table_name}`"
            )
            async with self._async_session() as session:
                result = await session.execute(stmt)
                for item in result:
                    meta = item[1]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
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
                placeholders = ",".join(
                    [f":node_id_{i}" for i in range(len(node_ids))]
                )
                params = {
                    f"node_id_{i}": nid for i, nid in enumerate(node_ids)
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
                placeholders = ",".join(
                    [f":node_id_{i}" for i in range(len(node_ids))]
                )
                params = {
                    f"node_id_{i}": nid for i, nid in enumerate(node_ids)
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
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    asyncio.run(self._async_engine.dispose())
                else:
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, self._async_engine.dispose()
                        )
                        future.result()
            except RuntimeError:
                asyncio.run(self._async_engine.dispose())
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
        m_val = m or self.default_m
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
        m_val = m or self.default_m
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
        with self._session() as session:
            session.execute(
                sqlalchemy.text(
                    f"ALTER TABLE `{self.table_name}` DROP INDEX `{name}`"
                )
            )
            session.commit()
        self._vector_index_name = None
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
        async with self._async_session() as session:
            await session.execute(
                sqlalchemy.text(
                    f"ALTER TABLE `{self.table_name}` DROP INDEX `{name}`"
                )
            )
            await session.commit()
        self._vector_index_name = None
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
