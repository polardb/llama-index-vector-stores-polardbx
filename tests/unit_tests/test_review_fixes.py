"""Unit tests for review bug fixes (Bug-1, Bug-2)."""

import pytest
from unittest.mock import patch

from llama_index.vector_stores.polardbx import PolarDBXVectorStore


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


# ==================================================================
# Bug-1: delete/adelete in no-JSON-column mode
# ==================================================================


class TestBug1DeleteNoJsonColumn:
    """Verify delete/adelete raise ValueError when metadata_json_column=None."""

    @patch.object(PolarDBXVectorStore, "_initialize", lambda self: None)
    def test_delete_raises_when_no_json_column(self):
        """delete(ref_doc_id) raises ValueError when metadata_json_column=None."""
        vs = _make_store(
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        with pytest.raises(ValueError, match="requires a JSON metadata column"):
            vs.delete(ref_doc_id="doc-001")

    @patch.object(PolarDBXVectorStore, "_initialize", lambda self: None)
    def test_adelete_raises_when_no_json_column(self):
        """adelete(ref_doc_id) raises ValueError when metadata_json_column=None."""
        vs = _make_store(
            metadata_json_column=None,
            metadata_columns=["category", "lang"],
        )
        with pytest.raises(ValueError, match="requires a JSON metadata column"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                vs.adelete(ref_doc_id="doc-001")
            )

    @patch.object(PolarDBXVectorStore, "_initialize", lambda self: None)
    def test_delete_works_with_json_column(self):
        """delete(ref_doc_id) does not raise ValueError about JSON column
        when metadata_json_column is set (it may fail at DB level instead)."""
        vs = _make_store(
            metadata_json_column="my_meta",
            metadata_columns=["category"],
        )
        # Should not raise our ValueError — it will fail at DB level
        # (no real connection), but the None check should pass.
        with pytest.raises(Exception) as exc_info:
            vs.delete(ref_doc_id="doc-001")
        # The error should NOT be our ValueError about JSON column
        assert "requires a JSON metadata column" not in str(exc_info.value)


# ==================================================================
# Bug-2: partition_column hardcoded "id"
# ==================================================================


class TestBug2PartitionColumnIdColumn:
    """Verify partition_column uses id_column value instead of hardcoded "id"."""

    def test_custom_id_column_with_partition_no_partition_column(self):
        """Scenario B: id_column="my_id", partition_by="HASH", no partition_column.

        Should default partition_column to id_column value ("my_id"),
        not hardcoded "id".
        """
        vs = _make_store(
            id_column="my_id",
            partition_by="HASH",
            partitions=8,
            # partition_column not set — should default to "my_id"
        )
        assert vs._partition_column == "my_id"
        assert vs._id_column == "my_id"

    def test_custom_id_column_with_matching_partition_column(self):
        """Scenario A: id_column="my_id", partition_by="HASH", partition_column="my_id".

        Should pass validation (partition_column matches id_column).
        """
        vs = _make_store(
            id_column="my_id",
            partition_by="HASH",
            partitions=8,
            partition_column="my_id",
        )
        assert vs._partition_column == "my_id"
        assert vs._id_column == "my_id"

    def test_custom_id_column_with_mismatched_partition_column(self):
        """id_column="my_id", partition_column="other" should fail."""
        with pytest.raises(ValueError, match="partition_column must match"):
            _make_store(
                id_column="my_id",
                partition_by="HASH",
                partitions=8,
                partition_column="other",
            )

    def test_default_id_column_with_partition(self):
        """Default id_column="id" with partition_by still works."""
        vs = _make_store(
            partition_by="HASH",
            partitions=8,
        )
        assert vs._partition_column == "id"
        assert vs._id_column == "id"

    def test_default_id_column_with_partition_column_id(self):
        """Explicit partition_column="id" with default id_column works."""
        vs = _make_store(
            partition_by="HASH",
            partitions=8,
            partition_column="id",
        )
        assert vs._partition_column == "id"

    def test_error_message_includes_both_column_names(self):
        """Error message should mention both partition_column and id_column."""
        with pytest.raises(ValueError) as exc_info:
            _make_store(
                id_column="my_id",
                partition_by="HASH",
                partitions=8,
                partition_column="wrong_col",
            )
        msg = str(exc_info.value)
        assert "partition_column" in msg
        assert "id_column" in msg
        assert "my_id" in msg
