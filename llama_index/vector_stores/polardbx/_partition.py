"""Partition clause helpers for PolarDB-X CREATE TABLE statements.

This module provides the shared partition-clause builder and identifier
validation utilities used by both ``PolarDBXVectorStore`` and
``create_partitioned_table``.

Keeping these functions in a standalone module ensures a single source of
truth for partition clause generation across the package.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str, label: str = "identifier") -> None:
    """Validate a SQL identifier to prevent SQL injection.

    Raises ValueError if the name is not a valid identifier.
    """
    if not name or not _IDENT_RE.match(name):
        raise ValueError(
            f"Invalid {label}: {name!r}. "
            "Must start with a letter or underscore, "
            "and contain only alphanumeric characters and underscores."
        )
    if len(name) > 64:
        raise ValueError(
            f"{label.capitalize()} too long: {name!r}. Maximum length is 64 characters."
        )


def _sql_quote_string(value: str) -> str:
    """Quote a string literal using SQL-standard single-quote doubling.

    Also escapes backslashes to prevent MySQL escape interpretation.
    """
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def _build_partition_clause(
    partition_by: Optional[str] = None,
    partition_column: str = "id",
    partitions: int = 0,
    broadcast: bool = False,
    locality: Optional[str] = None,
    partition_defs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the PARTITION/BROADCAST/LOCALITY clause for CREATE TABLE.

    Returns an empty string for a single (non-partitioned) table.
    This is the single source of truth for partition clause generation.
    """
    parts: List[str] = []

    if broadcast:
        parts.append("BROADCAST")
    elif partition_by:
        pby = partition_by.upper()
        # M4: Validate partition_column as identifier (defense in depth)
        _validate_identifier(partition_column, "partition column")
        if pby in ("HASH", "KEY"):
            parts.append(
                f"PARTITION BY {pby}({partition_column}) PARTITIONS {partitions}"
            )
        elif pby == "RANGE":
            items = []
            for d in partition_defs or []:
                name = d.get("name")
                if not name:
                    raise ValueError("Each RANGE partition def must have a 'name' key.")
                # S1: Validate partition name as identifier
                _validate_identifier(name, "partition name")
                vlt = d.get("values_less_than")
                if vlt is None:
                    raise ValueError(
                        f"RANGE partition '{name}' is missing 'values_less_than' key."
                    )
                if isinstance(vlt, str) and vlt.upper() == "MAXVALUE":
                    items.append(f"PARTITION {name} VALUES LESS THAN (MAXVALUE)")
                elif isinstance(vlt, str):
                    items.append(
                        f"PARTITION {name} VALUES LESS THAN ({_sql_quote_string(vlt)})"
                    )
                else:
                    items.append(f"PARTITION {name} VALUES LESS THAN ({vlt})")
            parts.append(
                f"PARTITION BY RANGE({partition_column}) (" + ", ".join(items) + ")"
            )
        elif pby == "LIST":
            items = []
            for d in partition_defs or []:
                name = d.get("name")
                if not name:
                    raise ValueError("Each LIST partition def must have a 'name' key.")
                # S1: Validate partition name as identifier
                _validate_identifier(name, "partition name")
                vals = d.get("values_in")
                if vals is None:
                    raise ValueError(
                        f"LIST partition '{name}' is missing 'values_in' key."
                    )
                # M1: Reject empty values_in list to prevent invalid SQL
                if not vals:
                    raise ValueError(
                        f"LIST partition '{name}' has empty 'values_in' list."
                    )
                val_str = ", ".join(
                    _sql_quote_string(v) if isinstance(v, str) else str(v) for v in vals
                )
                items.append(f"PARTITION {name} VALUES IN ({val_str})")
            parts.append(
                f"PARTITION BY LIST({partition_column}) (" + ", ".join(items) + ")"
            )

    if locality:
        # Escape single quotes to prevent injection via LOCALITY value
        safe_locality = locality.replace("'", "''")
        parts.append(f"LOCALITY='{safe_locality}'")

    return "".join(f" {p}" for p in parts) if parts else ""
