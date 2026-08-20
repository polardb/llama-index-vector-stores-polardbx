"""Unit tests for Column dataclass and custom column constructor validation."""

import pytest

from llama_index.vector_stores.polardbx.column import (
    Column,
    _validate_sql_literal,
)


class TestColumn:
    """Tests for the Column dataclass."""

    def test_basic_column(self):
        """A column with name and data_type is valid."""
        col = Column("price", "DECIMAL(10,2)")
        assert col.name == "price"
        assert col.data_type == "DECIMAL(10,2)"
        assert col.nullable is True
        assert col.default is None

    def test_column_with_options(self):
        """A column with nullable=False and default is valid."""
        col = Column(
            "category",
            "VARCHAR(100)",
            nullable=False,
            default="'unknown'",
        )
        assert col.name == "category"
        assert col.data_type == "VARCHAR(100)"
        assert col.nullable is False
        assert col.default == "'unknown'"

    def test_injection_semicolon_blocked(self):
        """Semicolon in data_type is rejected."""
        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("evil", "VARCHAR(10); DROP TABLE users--")

    def test_injection_comment_blocked(self):
        """SQL comment markers in data_type are rejected."""
        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("evil", "VARCHAR(10) /* comment */")

    def test_injection_dash_dash_blocked(self):
        """Double-dash comment in data_type is rejected."""
        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("evil", "INT--")

    def test_injection_newline_blocked(self):
        """Newline in data_type is rejected."""
        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("evil", "VARCHAR(10)\nDROP TABLE")

    def test_injection_in_default_blocked(self):
        """Injection via default field is rejected."""
        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("x", "INT", default="0; DROP TABLE")

    def test_empty_data_type_blocked(self):
        """Empty data_type is rejected."""
        with pytest.raises(ValueError, match="non-empty string"):
            Column("bad", "")

    def test_whitespace_data_type_blocked(self):
        """Whitespace-only data_type is rejected."""
        with pytest.raises(ValueError, match="non-empty string"):
            Column("bad", "   ")

    def test_non_string_data_type_blocked(self):
        """Non-string data_type is rejected."""
        with pytest.raises(ValueError, match="non-empty string"):
            Column("bad", 123)  # type: ignore[arg-type]


class TestValidateSqlLiteral:
    """Tests for the _validate_sql_literal helper."""

    def test_valid_literal(self):
        """A valid SQL type string passes validation."""
        assert _validate_sql_literal("VARCHAR(255)", "test") == "VARCHAR(255)"

    def test_carriage_return_blocked(self):
        r"""Carriage return is rejected."""
        with pytest.raises(ValueError, match="forbidden sequence"):
            _validate_sql_literal("INT\r", "test")

    def test_star_slash_blocked(self):
        r"""*/ sequence is rejected."""
        with pytest.raises(ValueError, match="forbidden sequence"):
            _validate_sql_literal("INT*/", "test")
