"""Unit tests for _validate_kwargs — no DB required."""

import pytest

from llama_index.vector_stores.polardbx import PolarDBXVectorStore


class TestKwargsValidation:
    """Tests for the _validate_kwargs static method."""

    def test_valid_ssl_cert_passes(self):
        """ssl_cert is a known pymysql parameter and should pass."""
        PolarDBXVectorStore._validate_kwargs({"ssl_cert": "/path/cert.pem"})

    def test_valid_ssl_key_passes(self):
        """ssl_key is a known pymysql parameter and should pass."""
        PolarDBXVectorStore._validate_kwargs({"ssl_key": "/path/key.pem"})

    def test_valid_connect_timeout_passes(self):
        """connect_timeout is a known pymysql parameter and should pass."""
        PolarDBXVectorStore._validate_kwargs({"connect_timeout": 10})

    def test_valid_read_timeout_passes(self):
        """read_timeout is a known pymysql parameter and should pass."""
        PolarDBXVectorStore._validate_kwargs({"read_timeout": 30})

    def test_valid_charset_passes(self):
        """charset is a known pymysql parameter and should pass."""
        PolarDBXVectorStore._validate_kwargs({"charset": "utf8mb4"})

    def test_typo_embed_dim_suggests_embed_dim(self):
        """Typo 'embed_dims' should suggest 'embed_dim'."""
        with pytest.raises(TypeError) as exc_info:
            PolarDBXVectorStore._validate_kwargs({"embed_dims": 768})
        assert "embed_dim" in str(exc_info.value)

    def test_typo_table_suggests_table_name(self):
        """Typo 'table' should suggest 'table_name'."""
        with pytest.raises(TypeError) as exc_info:
            PolarDBXVectorStore._validate_kwargs({"table": "my_table"})
        assert "table_name" in str(exc_info.value)

    def test_typo_distance_suggests_distance_strategy(self):
        """Typo 'distance' should suggest 'distance_strategy'."""
        with pytest.raises(TypeError) as exc_info:
            PolarDBXVectorStore._validate_kwargs({"distance": "cosine"})
        assert "distance_strategy" in str(exc_info.value)

    def test_unknown_kwarg_raises_typeerror(self):
        """Completely unknown kwarg should raise TypeError."""
        with pytest.raises(TypeError):
            PolarDBXVectorStore._validate_kwargs({"totally_unknown": True})

    def test_empty_kwargs_passes(self):
        """Empty kwargs dict should pass without error."""
        PolarDBXVectorStore._validate_kwargs({})

    def test_multiple_valid_kwargs_pass(self):
        """Multiple valid kwargs should all pass."""
        PolarDBXVectorStore._validate_kwargs(
            {
                "ssl_cert": "/cert.pem",
                "ssl_key": "/key.pem",
                "connect_timeout": 15,
                "read_timeout": 30,
                "write_timeout": 30,
                "charset": "utf8mb4",
            }
        )

    def test_mixed_valid_and_invalid_raises(self):
        """One invalid kwarg among valid ones should still raise."""
        with pytest.raises(TypeError):
            PolarDBXVectorStore._validate_kwargs(
                {
                    "ssl_cert": "/cert.pem",
                    "bogus_param": True,
                }
            )
