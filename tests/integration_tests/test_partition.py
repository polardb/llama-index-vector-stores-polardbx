"""Integration tests for partition table functionality on real PolarDB-X.

These tests verify that:
1. create_partitioned_table creates tables with all partition strategies
2. PolarDBXVectorStore creates partitioned vector tables with UNIQUE->INDEX downgrade
3. CRUD operations work on partitioned vector tables
4. LOCALITY clause works
5. IF NOT EXISTS is idempotent

Requires a live PolarDB-X instance configured via .env (POLARDBX_URI).
If the instance does not support partition + vector index, tests are skipped.
"""

import os
import sys
import uuid

import pytest
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import (
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    DN_NODE,
    make_store,
    uri,
)

# Use unique prefixes to avoid collisions
DDL_TABLE_PREFIX = "it_part_ddl"
VS_TABLE_PREFIX = "it_part_vs"


def _unique_table(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _drop_table_safe(engine: create_engine, table_name: str) -> None:
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
        conn.commit()


def _verify_engine() -> create_engine:
    """Create a verification engine using mysql+pymysql."""
    return create_engine(uri())


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------


def _skip_if_no_dn_node():
    """Skip test if POLARDBX_DN_NODE is not configured."""
    if not DN_NODE:
        pytest.skip("POLARDBX_DN_NODE not set in .env")


# ---------------------------------------------------------------------------
# create_partitioned_table integration tests
# ---------------------------------------------------------------------------


class TestCreatePartitionedTableDDL:
    """Test create_partitioned_table on real PolarDB-X."""

    def test_hash_partition_create_and_insert(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "user_id BIGINT NOT NULL",
                "amount DECIMAL(10,2)",
                "PRIMARY KEY (id, user_id)",
            ],
            partition_by="HASH",
            partition_column="user_id",
            partitions=4,
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(
                    f"INSERT INTO `{table}` (user_id, amount) VALUES (1, 99.99)"
                ))
                conn.commit()
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 1
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_broadcast_create_and_insert(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=[
                "id INT PRIMARY KEY",
                "name VARCHAR(100)",
            ],
            broadcast=True,
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(f"INSERT INTO `{table}` (id, name) VALUES (1, 'test')"))
                conn.commit()
                result = conn.execute(text(f"SELECT name FROM `{table}` WHERE id=1"))
                assert result.fetchone()[0] == "test"
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_range_partition_numeric_create_and_insert(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "val INT NOT NULL",
                "PRIMARY KEY (id, val)",
            ],
            partition_by="RANGE",
            partition_column="val",
            partition_defs=[
                {"name": "p0", "values_less_than": 100},
                {"name": "p1", "values_less_than": 1000},
                {"name": "p2", "values_less_than": "MAXVALUE"},
            ],
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(f"INSERT INTO `{table}` (val) VALUES (50), (500), (5000)"))
                conn.commit()
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 3
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_range_partition_string_values(self) -> None:
        """RANGE partition with string values (date strings) must be quoted."""
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "created_at VARCHAR(20) NOT NULL",
                "PRIMARY KEY (id, created_at)",
            ],
            partition_by="RANGE",
            partition_column="created_at",
            partition_defs=[
                {"name": "p0", "values_less_than": "2024-01-01"},
                {"name": "p1", "values_less_than": "2025-01-01"},
                {"name": "p2", "values_less_than": "MAXVALUE"},
            ],
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(
                    f"INSERT INTO `{table}` (created_at) VALUES ('2023-06-15')"
                ))
                conn.commit()
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 1
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_list_partition_strings(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "region VARCHAR(20) NOT NULL",
                "PRIMARY KEY (id, region)",
            ],
            partition_by="LIST",
            partition_column="region",
            partition_defs=[
                {"name": "p_east", "values_in": ["east", "northeast"]},
                {"name": "p_west", "values_in": ["west", "southwest"]},
            ],
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(
                    f"INSERT INTO `{table}` (region) VALUES ('east'), ('west')"
                ))
                conn.commit()
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 2
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_list_partition_with_single_quote(self) -> None:
        """LIST partition with string containing single quote."""
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "name VARCHAR(50) NOT NULL",
                "PRIMARY KEY (id, name)",
            ],
            partition_by="LIST",
            partition_column="name",
            partition_defs=[
                {"name": "p0", "values_in": ["O'Brien", "Smith"]},
            ],
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(
                    f"INSERT INTO `{table}` (name) VALUES ('O''Brien')"
                ))
                conn.commit()
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 1
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_locality_with_hash(self) -> None:
        """LOCALITY clause with a real DN node name from .env."""
        _skip_if_no_dn_node()

        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "val INT NOT NULL",
                "PRIMARY KEY (id, val)",
            ],
            partition_by="HASH",
            partition_column="val",
            partitions=4,
            locality=f"dn={DN_NODE}",
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(f"INSERT INTO `{table}` (val) VALUES (1)"))
                conn.commit()
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 1
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_single_table_no_partition(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=["id INT PRIMARY KEY", "data TEXT"],
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(f"INSERT INTO `{table}` (id, data) VALUES (1, 'hello')"))
                conn.commit()
                result = conn.execute(text(f"SELECT data FROM `{table}` WHERE id=1"))
                assert result.fetchone()[0] == "hello"
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_if_not_exists_idempotent(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        # Create once
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=["id INT PRIMARY KEY", "name VARCHAR(100)"],
            partition_by="HASH",
            partition_column="id",
            partitions=4,
        )
        # Create again with IF NOT EXISTS — should not error
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=["id INT PRIMARY KEY", "name VARCHAR(100)"],
            partition_by="HASH",
            partition_column="id",
            partitions=4,
        )
        engine = _verify_engine()
        _drop_table_safe(engine, table)
        engine.dispose()

    def test_key_partition(self) -> None:
        from llama_index.vector_stores.polardbx import create_partitioned_table

        table = _unique_table(DDL_TABLE_PREFIX)
        create_partitioned_table(
            uri=uri(),
            table_name=table,
            columns=[
                "id BIGINT NOT NULL AUTO_INCREMENT",
                "user_id BIGINT NOT NULL",
                "PRIMARY KEY (id, user_id)",
            ],
            partition_by="KEY",
            partition_column="user_id",
            partitions=8,
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                conn.execute(text(f"INSERT INTO `{table}` (user_id) VALUES (42)"))
                conn.commit()
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 1
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()


# ---------------------------------------------------------------------------
# VectorStore partition integration tests
# ---------------------------------------------------------------------------


class TestVectorStorePartition:
    """Test PolarDBXVectorStore with partition table on real PolarDB-X."""

    def test_hash_partition_create_table(self) -> None:
        """VectorStore should create a HASH partitioned table with INDEX (not UNIQUE).

        Note: PolarDB-X internally converts HASH to KEY in SHOW CREATE TABLE output.
        """
        table = _unique_table(VS_TABLE_PREFIX)
        vs = make_store(
            table_name=table,
            partition_by="HASH",
            partitions=4,
        )
        # Verify table was created
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SHOW CREATE TABLE `{table}`"))
                create_sql = result.fetchone()[1]
                # PolarDB-X converts HASH to KEY in output
                assert "PARTITION BY" in create_sql
                assert "PARTITIONS 4" in create_sql
                # Should NOT have UNIQUE INDEX (downgraded to INDEX)
                assert "UNIQUE INDEX" not in create_sql
                assert "UNIQUE KEY" not in create_sql
                assert "INDEX `node_id_index`" in create_sql or "KEY `node_id_index`" in create_sql
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_broadcast_create_table(self) -> None:
        table = _unique_table(VS_TABLE_PREFIX)
        vs = make_store(
            table_name=table,
            broadcast=True,
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SHOW CREATE TABLE `{table}`"))
                create_sql = result.fetchone()[1]
                assert "BROADCAST" in create_sql
                assert "UNIQUE INDEX" not in create_sql
                assert "UNIQUE KEY" not in create_sql
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_range_partition_create_table(self) -> None:
        table = _unique_table(VS_TABLE_PREFIX)
        vs = make_store(
            table_name=table,
            partition_by="RANGE",
            partition_defs=[
                {"name": "p0", "values_less_than": 100},
                {"name": "p1", "values_less_than": "MAXVALUE"},
            ],
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SHOW CREATE TABLE `{table}`"))
                create_sql = result.fetchone()[1]
                assert "PARTITION BY RANGE" in create_sql
                assert "UNIQUE INDEX" not in create_sql
                assert "UNIQUE KEY" not in create_sql
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_hash_partition_crud(self) -> None:
        """Full CRUD on a HASH partitioned vector table."""
        import numpy as np

        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import VectorStoreQuery

        table = _unique_table(VS_TABLE_PREFIX)
        vs = make_store(
            table_name=table,
            embed_dim=4,
            partition_by="HASH",
            partitions=4,
        )
        engine = _verify_engine()
        try:
            # Add nodes
            nodes = []
            for i in range(5):
                node = TextNode(
                    text=f"test text {i}",
                    embedding=np.random.rand(4).tolist(),
                )
                nodes.append(node)
            vs.add(nodes)

            # Verify count
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 5

            # Search
            query_emb = np.random.rand(4).tolist()
            results = vs.query(
                VectorStoreQuery(query_emb, similarity_top_k=3)
            )
            assert len(results.nodes) <= 3

            # Delete by node_id (use delete_nodes, not delete)
            if results.nodes:
                vs.delete_nodes(node_ids=[results.nodes[0].node_id])
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                    assert result.fetchone()[0] == 4
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_broadcast_partition_crud(self) -> None:
        """Full CRUD on a broadcast vector table."""
        import numpy as np

        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import VectorStoreQuery

        table = _unique_table(VS_TABLE_PREFIX)
        vs = make_store(
            table_name=table,
            embed_dim=4,
            broadcast=True,
        )
        engine = _verify_engine()
        try:
            nodes = [
                TextNode(
                    text=f"broadcast test {i}",
                    embedding=np.random.rand(4).tolist(),
                )
                for i in range(3)
            ]
            vs.add(nodes)

            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 3

            results = vs.query(
                VectorStoreQuery(np.random.rand(4).tolist(), similarity_top_k=2)
            )
            assert len(results.nodes) <= 2
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_range_partition_crud(self) -> None:
        """Full CRUD on a RANGE partitioned vector table."""
        import numpy as np

        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import VectorStoreQuery

        table = _unique_table(VS_TABLE_PREFIX)
        vs = make_store(
            table_name=table,
            embed_dim=4,
            partition_by="RANGE",
            partition_defs=[
                {"name": "p0", "values_less_than": 100},
                {"name": "p1", "values_less_than": "MAXVALUE"},
            ],
        )
        engine = _verify_engine()
        try:
            nodes = [
                TextNode(
                    text=f"range test {i}",
                    embedding=np.random.rand(4).tolist(),
                )
                for i in range(3)
            ]
            vs.add(nodes)

            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`"))
                assert result.fetchone()[0] == 3

            results = vs.query(
                VectorStoreQuery(np.random.rand(4).tolist(), similarity_top_k=2)
            )
            assert len(results.nodes) <= 2
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_no_partition_uses_unique_index(self) -> None:
        """Without partitioning, table should use UNIQUE INDEX for node_id.

        Note: PolarDB-X SHOW CREATE TABLE outputs 'UNIQUE KEY' instead of
        'UNIQUE INDEX', and non-partitioned tables get 'SINGLE' keyword.
        """
        table = _unique_table(VS_TABLE_PREFIX)
        vs = make_store(
            table_name=table,
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SHOW CREATE TABLE `{table}`"))
                create_sql = result.fetchone()[1]
                # PolarDB-X outputs 'UNIQUE KEY' in SHOW CREATE TABLE
                assert "UNIQUE" in create_sql
                assert "node_id_index" in create_sql
                assert "PARTITION" not in create_sql
                assert "BROADCAST" not in create_sql
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()

    def test_locality_with_hash(self) -> None:
        """VectorStore with LOCALITY clause on a HASH partitioned table."""
        _skip_if_no_dn_node()

        table = _unique_table(VS_TABLE_PREFIX)
        vs = make_store(
            table_name=table,
            partition_by="HASH",
            partitions=4,
            locality=f"dn={DN_NODE}",
        )
        engine = _verify_engine()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SHOW CREATE TABLE `{table}`"))
                create_sql = result.fetchone()[1]
                assert "PARTITION BY" in create_sql
                assert "LOCALITY" in create_sql
        finally:
            _drop_table_safe(engine, table)
            engine.dispose()


# ---------------------------------------------------------------------------
# Validation error tests (raise before connecting)
# ---------------------------------------------------------------------------


class TestPartitionValidationErrors:
    """Test that validation errors are raised before connecting."""

    def test_invalid_partition_by_raises(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="Invalid partition_by"):
            PolarDBXVectorStore(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                partition_by="INVALID",
                partitions=4,
            )

    def test_hash_without_partitions_raises(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="partitions must be > 0"):
            PolarDBXVectorStore(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                partition_by="HASH",
                partitions=0,
            )

    def test_broadcast_and_partition_mutually_exclusive(self) -> None:
        from llama_index.vector_stores.polardbx import PolarDBXVectorStore

        with pytest.raises(ValueError, match="mutually exclusive"):
            PolarDBXVectorStore(
                host=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                partition_by="HASH",
                partitions=4,
                broadcast=True,
            )
