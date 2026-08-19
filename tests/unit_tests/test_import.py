"""Unit tests — import and basic class structure validation (no DB required)."""

from llama_index.vector_stores.polardbx import (
    NotSupportedError,
    PolarDBXVectorStore,
)


def test_import_polar_dbx_vector_store():
    """PolarDBXVectorStore can be imported."""
    assert PolarDBXVectorStore is not None


def test_import_not_supported_error():
    """NotSupportedError can be imported."""
    assert NotSupportedError is not None
    assert issubclass(NotSupportedError, Exception)


def test_class_name():
    """class_name() returns the correct class name."""
    assert PolarDBXVectorStore.class_name() == "PolarDBXVectorStore"


def test_class_attributes():
    """Class-level attributes have expected values."""
    # stores_text and flat_metadata are regular Pydantic fields (not ClassVar),
    # so check defaults via model_fields instead of class-level access.
    fields = PolarDBXVectorStore.model_fields
    assert fields["stores_text"].default is True
    assert fields["flat_metadata"].default is False


def test_is_subclass_of_base():
    """PolarDBXVectorStore is a subclass of BasePydanticVectorStore."""
    from llama_index.core.vector_stores.types import BasePydanticVectorStore

    assert issubclass(PolarDBXVectorStore, BasePydanticVectorStore)


def test_validate_identifier():
    """_validate_identifier accepts valid identifiers, rejects invalid."""
    assert PolarDBXVectorStore._validate_identifier("valid_name") == "valid_name"
    assert PolarDBXVectorStore._validate_identifier("_under") == "_under"
    assert PolarDBXVectorStore._validate_identifier("CamelCase") == "CamelCase"

    for bad in ["invalid-name!", "123start", "has space", "", "drop;table"]:
        try:
            PolarDBXVectorStore._validate_identifier(bad)
            assert False, f"Expected ValueError for: {bad}"
        except ValueError:
            pass


def test_validate_positive_int():
    """_validate_positive_int accepts positive ints, rejects others."""
    assert PolarDBXVectorStore._validate_positive_int(1, "test") == 1
    assert PolarDBXVectorStore._validate_positive_int(100, "test") == 100

    for bad in [0, -1, -100, "1", 1.5, None]:
        try:
            PolarDBXVectorStore._validate_positive_int(bad, "test")
            assert False, f"Expected ValueError for: {bad}"
        except ValueError:
            pass
