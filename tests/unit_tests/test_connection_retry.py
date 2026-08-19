"""Unit tests for connection retry logic — no DB required."""

from unittest.mock import patch, MagicMock

import pytest

from llama_index.vector_stores.polardbx import PolarDBXVectorStore
from llama_index.vector_stores.polardbx.base import NotSupportedError


class TestConnectionRetry:
    """Tests for the connection retry mechanism in _initialize."""

    def test_retry_params_stored(self):
        """Verify connection_retries and retry_delay are stored."""
        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                connection_retries=5,
                retry_delay=0.5,
            )
            assert store._connection_retries == 5
            assert store._retry_delay == 0.5

    def test_retry_params_default(self):
        """Verify default retry params."""
        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
            )
            assert store._connection_retries == 3
            assert store._retry_delay == 1.0

    def test_from_params_passes_retry_params(self):
        """Verify from_params passes retry params."""
        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore.from_params(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                connection_retries=7,
                retry_delay=2.0,
            )
            assert store._connection_retries == 7
            assert store._retry_delay == 2.0

    def test_retry_succeeds_on_second_attempt(self):
        """First attempt fails, second succeeds."""
        call_count = {"connect": 0, "detect": 0}

        def fake_connect(self):
            call_count["connect"] += 1

        def fake_detect(self):
            call_count["detect"] += 1
            if call_count["detect"] == 1:
                raise ConnectionError("DB temporarily unavailable")

        def fake_validate(self):
            pass

        def fake_create(self):
            pass

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                connection_retries=3,
                retry_delay=0.01,
            )

        # Now manually test _initialize with mocked internals
        with patch.object(
            PolarDBXVectorStore, "_connect", fake_connect
        ), patch.object(
            PolarDBXVectorStore, "_detect_capabilities", fake_detect
        ), patch.object(
            PolarDBXVectorStore, "_validate_distance_method", fake_validate
        ), patch.object(
            PolarDBXVectorStore, "_create_table_if_not_exists", fake_create
        ), patch.object(
            PolarDBXVectorStore, "_cleanup_engines", lambda self: None
        ):
            store._is_initialized = False
            store._initialize()

        assert call_count["connect"] == 2
        assert call_count["detect"] == 2
        assert store._is_initialized is True

    def test_retry_exhausted_raises_last_error(self):
        """All retry attempts fail — last exception is raised."""
        call_count = {"connect": 0}

        def fake_connect(self):
            call_count["connect"] += 1

        def fake_detect(self):
            raise ConnectionError("DB unavailable")

        def fake_validate(self):
            pass

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                connection_retries=3,
                retry_delay=0.01,
            )

        with patch.object(
            PolarDBXVectorStore, "_connect", fake_connect
        ), patch.object(
            PolarDBXVectorStore, "_detect_capabilities", fake_detect
        ), patch.object(
            PolarDBXVectorStore, "_validate_distance_method", fake_validate
        ), patch.object(
            PolarDBXVectorStore, "_create_table_if_not_exists",
            lambda self: None,
        ), patch.object(
            PolarDBXVectorStore, "_cleanup_engines", lambda self: None
        ):
            store._is_initialized = False
            with pytest.raises(ConnectionError, match="DB unavailable"):
                store._initialize()

        assert call_count["connect"] == 3
        assert store._is_initialized is False

    def test_value_error_not_retried(self):
        """ValueError (configuration error) is not retried."""
        call_count = {"connect": 0}

        def fake_connect(self):
            call_count["connect"] += 1

        def fake_detect(self):
            raise ValueError("Invalid configuration")

        def fake_validate(self):
            pass

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                connection_retries=3,
                retry_delay=0.01,
            )

        with patch.object(
            PolarDBXVectorStore, "_connect", fake_connect
        ), patch.object(
            PolarDBXVectorStore, "_detect_capabilities", fake_detect
        ), patch.object(
            PolarDBXVectorStore, "_validate_distance_method", fake_validate
        ), patch.object(
            PolarDBXVectorStore, "_create_table_if_not_exists",
            lambda self: None,
        ), patch.object(
            PolarDBXVectorStore, "_cleanup_engines", lambda self: None
        ):
            store._is_initialized = False
            with pytest.raises(ValueError, match="Invalid configuration"):
                store._initialize()

        # Should have only tried once
        assert call_count["connect"] == 1

    def test_not_supported_error_not_retried(self):
        """NotSupportedError is not retried."""
        call_count = {"connect": 0}

        def fake_connect(self):
            call_count["connect"] += 1

        def fake_detect(self):
            raise NotSupportedError("Feature not supported")

        def fake_validate(self):
            pass

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                connection_retries=3,
                retry_delay=0.01,
            )

        with patch.object(
            PolarDBXVectorStore, "_connect", fake_connect
        ), patch.object(
            PolarDBXVectorStore, "_detect_capabilities", fake_detect
        ), patch.object(
            PolarDBXVectorStore, "_validate_distance_method", fake_validate
        ), patch.object(
            PolarDBXVectorStore, "_create_table_if_not_exists",
            lambda self: None,
        ), patch.object(
            PolarDBXVectorStore, "_cleanup_engines", lambda self: None
        ):
            store._is_initialized = False
            with pytest.raises(NotSupportedError):
                store._initialize()

        assert call_count["connect"] == 1

    def test_cleanup_engines_called_between_retries(self):
        """_cleanup_engines is called between retry attempts."""
        cleanup_count = {"count": 0}

        def fake_connect(self):
            pass

        def fake_detect(self):
            raise ConnectionError("DB unavailable")

        def fake_validate(self):
            pass

        def fake_cleanup(self):
            cleanup_count["count"] += 1

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                connection_retries=3,
                retry_delay=0.01,
            )

        with patch.object(
            PolarDBXVectorStore, "_connect", fake_connect
        ), patch.object(
            PolarDBXVectorStore, "_detect_capabilities", fake_detect
        ), patch.object(
            PolarDBXVectorStore, "_validate_distance_method", fake_validate
        ), patch.object(
            PolarDBXVectorStore, "_create_table_if_not_exists",
            lambda self: None,
        ), patch.object(
            PolarDBXVectorStore, "_cleanup_engines", fake_cleanup
        ):
            store._is_initialized = False
            with pytest.raises(ConnectionError):
                store._initialize()

        # 3 retries, cleanup called after every failed attempt
        assert cleanup_count["count"] == 3

    def test_single_retry(self):
        """connection_retries=1 means no retry."""
        call_count = {"connect": 0}

        def fake_connect(self):
            call_count["connect"] += 1

        def fake_detect(self):
            raise ConnectionError("DB unavailable")

        def fake_validate(self):
            pass

        with patch.object(PolarDBXVectorStore, "_initialize") as mock_init:
            mock_init.return_value = None
            store = PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="root",
                password="test",
                database="testdb",
                connection_retries=1,
                retry_delay=0.01,
            )

        with patch.object(
            PolarDBXVectorStore, "_connect", fake_connect
        ), patch.object(
            PolarDBXVectorStore, "_detect_capabilities", fake_detect
        ), patch.object(
            PolarDBXVectorStore, "_validate_distance_method", fake_validate
        ), patch.object(
            PolarDBXVectorStore, "_create_table_if_not_exists",
            lambda self: None,
        ), patch.object(
            PolarDBXVectorStore, "_cleanup_engines", lambda self: None
        ):
            store._is_initialized = False
            with pytest.raises(ConnectionError):
                store._initialize()

        assert call_count["connect"] == 1
