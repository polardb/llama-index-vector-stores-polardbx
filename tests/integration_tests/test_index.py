"""Tests for LlamaIndex PolarDBXVectorStore — index management (sync + async)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, METADATAS, TEXTS, is_v3, make_nodes, make_store
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.polardbx import NotSupportedError


def _build_query(text="database", k=3):
    """Build a VectorStoreQuery for testing."""
    return VectorStoreQuery(
        query_embedding=EMB.embed_query(text),
        similarity_top_k=k,
        mode=VectorStoreQueryMode.DEFAULT,
    )


def _drop_existing_vi(vs):
    """Drop the default vector index created with the table."""
    existing = vs._detect_vector_index_name()
    if existing:
        vs.drop_vector_index(index_name=existing)
        return existing
    return None


# ==================== SYNC: basic index management ====================


def test_sync_apply_and_drop_vector_index():
    """apply_vector_index() and drop_vector_index() work end-to-end."""
    vs = make_store()
    vs.add(make_nodes(TEXTS, METADATAS))

    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_test", m=8)

    detected = vs._detect_vector_index_name()
    assert detected == "vi_test"

    vs.drop_vector_index(index_name="vi_test")
    assert vs._detect_vector_index_name() is None

    vs.drop()
    vs.close()


def test_sync_drop_vector_index_auto_detect():
    """drop_vector_index() without index_name auto-detects."""
    vs = make_store(table_name="test_li_idx_autodrop")
    vs.add(make_nodes(TEXTS, METADATAS))

    detected = vs._detect_vector_index_name()
    assert detected is not None

    vs.drop_vector_index()
    assert vs._detect_vector_index_name() is None

    vs.drop()
    vs.close()


def test_sync_detect_vector_index_name():
    """_detect_vector_index_name() finds the default index."""
    vs = make_store(table_name="test_li_idx_detect")
    vs.add(make_nodes(TEXTS[:1], METADATAS[:1]))

    vs._vector_index_name = None  # force re-detection
    detected = vs._detect_vector_index_name()
    assert detected is not None

    vs.drop()
    vs.close()


def test_sync_get_stats():
    """get_stats() returns a dict with Vidx* status variables."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:2], METADATAS[:2]))

    stats = vs.get_stats()
    assert isinstance(stats, dict)
    # Should contain some Vidx* status variables
    assert len(stats) > 0

    vs.drop()
    vs.close()


def test_sync_optimize():
    """optimize() runs OPTIMIZE TABLE without error."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:2], METADATAS[:2]))

    vs.optimize()  # should not raise

    vs.drop()
    vs.close()


def test_sync_apply_vector_index_variants():
    """apply_vector_index() with different M values."""
    # M=4
    vs = make_store(table_name="test_li_idx_m4")
    vs.add(make_nodes(TEXTS, METADATAS))
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m4", m=4)
    assert vs._detect_vector_index_name() == "vi_m4"
    vs.drop_vector_index(index_name="vi_m4")
    vs.drop()
    vs.close()

    # M=16
    vs = make_store(table_name="test_li_idx_m16")
    vs.add(make_nodes(TEXTS, METADATAS))
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m16", m=16)
    assert vs._detect_vector_index_name() == "vi_m16"
    vs.drop_vector_index(index_name="vi_m16")
    vs.drop()
    vs.close()

    # M=32 with distance=EUCLIDEAN
    vs = make_store(
        table_name="test_li_idx_m32", distance_method="EUCLIDEAN"
    )
    vs.add(make_nodes(TEXTS, METADATAS))
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m32", m=32, distance="EUCLIDEAN")
    assert vs._detect_vector_index_name() == "vi_m32"
    vs.drop_vector_index(index_name="vi_m32")
    vs.drop()
    vs.close()


# ==================== SYNC: query with index ====================


def test_sync_query_with_ef_search_and_index():
    """query() with ef_search after apply_vector_index."""
    vs = make_store()
    vs.add(make_nodes(TEXTS, METADATAS))
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_ef", m=8)

    # query with ef_search
    result = vs.query(
        _build_query("database", k=3),
        ef_search=50,
    )
    assert len(result.nodes) <= 3

    # query with search_type=ann
    result = vs.query(
        _build_query("database", k=3),
        search_type="ann",
        ef_search=80,
    )
    assert len(result.nodes) <= 3

    vs.drop_vector_index(index_name="vi_ef")
    vs.drop()
    vs.close()


def test_sync_query_search_type_knn_with_index():
    """query() with search_type='knn' after apply_vector_index."""
    vs = make_store(table_name="test_li_idx_knn")
    vs.add(make_nodes(TEXTS, METADATAS))
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_knn", m=8)

    result = vs.query(_build_query("database", k=3), search_type="knn")
    assert len(result.nodes) <= 3
    assert len(result.nodes) > 0

    vs.drop_vector_index(index_name="vi_knn")
    vs.drop()
    vs.close()


# ==================== SYNC: ef_search boundary ====================


def test_sync_ef_search_boundary():
    """ef_search at extreme values after apply_vector_index."""
    vs = make_store(table_name="test_li_idx_efbd")
    vs.add(make_nodes(TEXTS, METADATAS))
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_efbd", m=8)

    # ef_search=1
    result = vs.query(_build_query("database", k=3), ef_search=1)
    assert len(result.nodes) <= 3

    # ef_search=10000
    result = vs.query(_build_query("database", k=3), ef_search=10000)
    assert len(result.nodes) <= 3

    vs.drop_vector_index(index_name="vi_efbd")
    vs.drop()
    vs.close()


# ==================== v3: EF_CONSTRUCTION in DDL ====================


def test_ef_construction_in_ddl():
    """ef_construction parameter in CREATE TABLE DDL (v3 only)."""
    vs = make_store(table_name="test_li_idx_efcddl", ef_construction=40)
    vs.add(make_nodes(TEXTS[:2], METADATAS[:2]))

    if is_v3(vs):
        import sqlalchemy

        with vs._session() as session:
            result = session.execute(
                sqlalchemy.text("SHOW CREATE TABLE `test_li_idx_efcddl`")
            )
            row = result.fetchone()
            create_sql = str(row[1]) if row else ""
        assert "EF_CONSTRUCTION=40" in create_sql.upper(), (
            f"EF_CONSTRUCTION=40 not found in DDL: {create_sql[-120:]}"
        )
    else:
        # Old version: ef_construction silently ignored, table still works
        assert vs.count() == 2

    vs.drop()
    vs.close()


# ==================== v3: EF_CONSTRUCTION in apply_vector_index ====================


def test_apply_vector_index_with_ef_construction():
    """apply_vector_index() with ef_construction (v3 only)."""
    vs = make_store(table_name="test_li_idx_efcapply")
    vs.add(make_nodes(TEXTS[:2], METADATAS[:2]))
    _drop_existing_vi(vs)

    if is_v3(vs):
        vs.apply_vector_index(index_name="vi_efc", m=8, ef_construction=64)
        assert vs._detect_vector_index_name() == "vi_efc"
    else:
        vs.apply_vector_index(index_name="vi_efc", m=8)
        assert vs._detect_vector_index_name() == "vi_efc"

    vs.drop_vector_index(index_name="vi_efc")
    vs.drop()
    vs.close()


# ==================== v3: INNER_PRODUCT in apply_vector_index ====================


def test_apply_vector_index_inner_product():
    """apply_vector_index() with INNER_PRODUCT requires v3."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:2], METADATAS[:2]))
    _drop_existing_vi(vs)

    if is_v3(vs):
        vs.apply_vector_index(
            index_name="vi_ip", m=8, distance="INNER_PRODUCT"
        )
        assert vs._detect_vector_index_name() == "vi_ip"
        vs.drop_vector_index(index_name="vi_ip")
    else:
        try:
            vs.apply_vector_index(
                index_name="vi_ip", m=8, distance="INNER_PRODUCT"
            )
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass

    vs.drop()
    vs.close()


# ==================== v3: preload_index / preload_check ====================


def test_preload_index():
    """preload_index() succeeds on v3, raises NotSupportedError on old."""
    vs = make_store(table_name="test_li_idx_preload")
    vs.add(make_nodes(TEXTS[:1], METADATAS[:1]))

    if is_v3(vs):
        vs.preload_index()  # should not raise
    else:
        try:
            vs.preload_index()
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass

    vs.drop()
    vs.close()


def test_preload_check():
    """preload_check() returns dict on v3, raises NotSupportedError on old."""
    vs = make_store(table_name="test_li_idx_plchk")
    vs.add(make_nodes(TEXTS[:1], METADATAS[:1]))

    if is_v3(vs):
        result = vs.preload_check()
        assert isinstance(result, dict)
    else:
        try:
            vs.preload_check()
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass

    vs.drop()
    vs.close()


# ==================== v3: explain_index_health ====================


def test_explain_index_health():
    """explain_index_health() returns metadata on v3, raises on old."""
    vs = make_store(table_name="test_li_idx_health")
    vs.add(make_nodes(TEXTS[:1], METADATAS[:1]))

    if is_v3(vs):
        result = vs.explain_index_health()
        assert "index_info" in result
        assert "explain" in result
        assert result["index_info"] is not None
    else:
        try:
            vs.explain_index_health()
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass

    vs.drop()
    vs.close()


# ==================== ASYNC ====================


async def test_async_apply_and_drop_vector_index():
    """aapply_vector_index() and adrop_vector_index() work end-to-end."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS, METADATAS))

    old = vs._detect_vector_index_name()
    if old:
        await vs.adrop_vector_index(index_name=old)

    await vs.aapply_vector_index(index_name="vi_async", m=8)
    assert vs._detect_vector_index_name() == "vi_async"

    await vs.adrop_vector_index(index_name="vi_async")
    assert vs._detect_vector_index_name() is None

    vs.drop()
    vs.close()


async def test_async_get_stats():
    """aget_stats() returns a dict with Vidx* status variables."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS[:2], METADATAS[:2]))

    stats = await vs.aget_stats()
    assert isinstance(stats, dict)
    assert len(stats) > 0

    vs.drop()
    vs.close()


async def test_async_optimize():
    """aoptimize() runs OPTIMIZE TABLE without error."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS[:2], METADATAS[:2]))

    await vs.aoptimize()  # should not raise

    vs.drop()
    vs.close()


async def test_async_query_with_ef_search_and_index():
    """aquery() with ef_search after aapply_vector_index."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS, METADATAS))

    old = vs._detect_vector_index_name()
    if old:
        await vs.adrop_vector_index(index_name=old)
    await vs.aapply_vector_index(index_name="vi_aef", m=8)

    from llama_index.core.vector_stores.types import (
        VectorStoreQuery,
        VectorStoreQueryMode,
    )

    query = VectorStoreQuery(
        query_embedding=EMB.embed_query("database"),
        similarity_top_k=3,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    result = await vs.aquery(query, ef_search=50)
    assert len(result.nodes) <= 3

    await vs.adrop_vector_index(index_name="vi_aef")
    vs.drop()
    vs.close()


# ==================== v3: async preload / explain_index_health ====================


async def test_async_preload_and_health():
    """Async preload_index/preload_check/explain_index_health on v3."""
    vs = make_store(table_name="test_li_idx_asyncv3")
    await vs.async_add(make_nodes(TEXTS[:1], METADATAS[:1]))

    if is_v3(vs):
        await vs.apreload_index()
        result = await vs.apreload_check()
        assert isinstance(result, dict)
        health = await vs.aexplain_index_health()
        assert "index_info" in health
    else:
        try:
            await vs.apreload_index()
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass

    vs.drop()
    vs.close()
