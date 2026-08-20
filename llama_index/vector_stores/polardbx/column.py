"""Column definition for custom table schema support.

Used when creating a new table with custom metadata columns.
When connecting to an existing table, only column name strings
are needed (the data_type is ignored).
"""

from dataclasses import dataclass
from typing import Optional


def _validate_sql_literal(value: str, label: str) -> str:
    """Validate a raw SQL expression string to prevent injection.

    Used for Column.data_type and Column.default which are directly
    interpolated into DDL (not parameterized).  Forbids characters
    that could be used for SQL injection.

    Args:
        value: The SQL expression string to validate.
        label: Human-readable label for error messages.

    Returns:
        The validated string.

    Raises:
        ValueError: If the value is empty or contains forbidden sequences.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Invalid {label}: must be a non-empty string"
        )
    # Forbid SQL comment markers, statement separators, and newlines
    dangerous = [";", "--", "/*", "*/", "\n", "\r"]
    for marker in dangerous:
        if marker in value:
            raise ValueError(
                f"Invalid {label}: contains forbidden sequence "
                f"'{marker}'. Only SQL type expressions are allowed."
            )
    return value


@dataclass
class Column:
    """Table column definition for custom schema.

    Used when creating a new table with custom metadata columns.
    When connecting to an existing table, only column name strings
    are needed (the data_type is ignored).

    Example:
        .. code-block:: python

            from llama_index.vector_stores.polardbx import (
                PolarDBXVectorStore,
                Column,
            )

            store = PolarDBXVectorStore(
                ...,
                metadata_columns=[
                    Column("price", "DECIMAL(10,2)"),
                    Column("category", "VARCHAR(100)"),
                ],
            )
    """

    name: str
    data_type: str  # SQL type, e.g. "VARCHAR(255)", "INT", "DECIMAL(10,2)"
    nullable: bool = True
    default: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate data_type and default to prevent SQL injection."""
        self.data_type = _validate_sql_literal(self.data_type, "data_type")
        if self.default is not None:
            self.default = _validate_sql_literal(self.default, "default")
