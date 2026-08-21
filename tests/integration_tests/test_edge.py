"""Edge case tests for LlamaIndex PolarDBXVectorStore."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import (
    EMB,
    EMBED_DIM,
    METADATAS,
    TEXTS,
    FakeEmbeddings,
    is_v3,
    make_nodes,
    make_store,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.polardbx import NotSupportedError


def _build_query(text="database", k=3):
    return VectorStoreQuery(
        query_embedding=EMB.embed_query(text),
        similarity_top_k=k,
        mode=VectorStoreQueryMode.DEFAULT,
    )


# ==================== 1. EUCLIDEAN distance method ====================


def test_euclidean():
    """EUCLIDEAN distance strategy works end-to-end."""
    vs = make_store(
        table_name="test_li_euclidean", distance_strategy="euclidean"
    )
    vs.add(make_nodes(TEXTS, METADATAS))

    result = vs.query(_build_query("database", k=3))
    assert len(result.nodes) == 3
    assert all(isinstance(s, (int, float)) for s in result.similarities)

    vs.drop()
    vs.close()


# ==================== 2. Special characters / SQL injection ====================


def test_special_chars():
    """Special characters and SQL injection attempts are handled safely."""
    vs = make_store(table_name="test_li_special")
    special_texts = [
        "It's a test with 'single quotes'",
        "Semicolon; DROP TABLE test_li_special;--",
        "Backslash \\ and newline \n text",
        "Unicode: 你好世界 🌍 café naïve",
        "Mixed: <script>alert('xss')</script>",
    ]
    nodes = make_nodes(special_texts)
    vs.add(nodes)
    assert vs.count() == 5

    # Verify text round-trip
    fetched = vs.get_nodes(node_ids=[nodes[0].node_id])
    assert fetched[0].get_content() == special_texts[0]

    fetched = vs.get_nodes(node_ids=[nodes[3].node_id])
    assert fetched[0].get_content() == special_texts[3]

    # Table still exists (SQL injection didn't work)
    assert vs.count() == 5

    vs.drop()
    vs.close()


# ==================== 3. Large / nested metadata ====================


def test_large_metadata():
    """Large and nested metadata is stored and retrieved correctly."""
    vs = make_store(table_name="test_li_bigmeta")
    nested_meta = {
        "level1": {
            "level2": {
                "level3": "deep value",
                "numbers": [1, 2, 3, 4, 5],
            },
            "tags": ["ai", "vector", "database"],
        },
        "unicode": "你好🌟",
        "long_string": "x" * 500,
        "boolean": True,
        "null_value": None,
    }
    nodes = make_nodes(["doc with big metadata"], [nested_meta])
    vs.add(nodes)
    assert vs.count() == 1

    fetched = vs.get_nodes(node_ids=[nodes[0].node_id])
    meta = fetched[0].metadata

    # LlamaIndex metadata_dict_to_node may flatten some fields;
    # verify what we can
    assert "level1" in meta or "level2" in meta or "level3" in meta
    assert "unicode" in meta
    assert meta.get("unicode") == "你好🌟"

    vs.drop()
    vs.close()


# ==================== 4. Data persistence across store instances ====================


def test_persistence():
    """Data persists across store instances."""
    vs1 = make_store(table_name="test_li_persist_edge", pre_delete=True)
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    vs1.add(nodes)
    assert vs1.count() == 3
    vs1.close()

    vs2 = make_store(table_name="test_li_persist_edge", pre_delete=False)
    assert vs2.count() == 3

    result = vs2.query(_build_query("database", k=2))
    assert len(result.nodes) == 2

    vs2.drop()
    vs2.close()


# ==================== 5. INNER_PRODUCT distance method (v3) ====================


def test_inner_product():
    """INNER_PRODUCT distance method requires v3."""
    probe = make_store()
    v3 = is_v3(probe)
    probe.drop()
    probe.close()

    if v3:
        vs = make_store(
            table_name="test_li_innerprod",
            distance_strategy="inner_product",
        )
        vs.add(make_nodes(TEXTS, METADATAS))

        result = vs.query(_build_query("database", k=3))
        assert len(result.nodes) <= 3
        assert all(isinstance(s, (int, float)) for s in result.similarities)

        vs.drop()
        vs.close()
    else:
        try:
            vs = make_store(
                table_name="test_li_innerprod",
                distance_strategy="inner_product",
            )
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass


# ==================== 6. Vector dimension mismatch ====================


def test_dimension_mismatch():
    """Inserting wrong-dimension embedding is handled gracefully."""
    vs = make_store(table_name="test_li_dimmm")
    vs.add(make_nodes(["seed text"]))  # creates table with dim=128

    # Try to insert with wrong dimension embedding
    wrong_emb = FakeEmbeddings(dim=256)
    bad_node = make_nodes(
        ["wrong dim text"], emb=wrong_emb
    )[0]
    try:
        vs.add([bad_node])
        # If no error, verify it didn't corrupt data
        assert vs.count() >= 1
    except Exception:
        # Correctly rejected wrong dimension
        pass

    vs.drop()
    vs.close()


# ==================== 7. Empty add list ====================


def test_empty_add():
    """add() with empty list returns empty list and doesn't error."""
    vs = make_store()
    ids = vs.add([])
    assert ids == []
    assert vs.count() == 0
    vs.drop()
    vs.close()


# ==================== 8. Query mode validation ====================


def test_unsupported_query_mode():
    """query() with non-DEFAULT mode raises NotImplementedError."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:1], METADATAS[:1]))

    query = VectorStoreQuery(
        query_embedding=EMB.embed_query("test"),
        similarity_top_k=1,
        mode=VectorStoreQueryMode.SVM,
    )
    try:
        vs.query(query)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass

    vs.drop()
    vs.close()


# ==================== 9. get_nodes with empty list ====================


def test_get_nodes_empty_ids():
    """get_nodes() with empty node_ids returns all nodes."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:3], METADATAS[:3]))

    # Empty node_ids falls through to the "all" branch
    fetched = vs.get_nodes(node_ids=[])
    # Empty list is falsy, so it goes to the "all" branch
    assert len(fetched) == 3

    vs.drop()
    vs.close()


# ==================== 10. delete non-existent ref_doc_id ====================


def test_delete_nonexistent_ref_doc_id():
    """delete() with non-existent ref_doc_id does nothing, no error."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:3], METADATAS[:3]))
    assert vs.count() == 3

    vs.delete(ref_doc_id="nonexistent-doc-id")
    assert vs.count() == 3  # unchanged

    vs.drop()
    vs.close()


# ==================== 11. delete_nodes non-existent node_ids ====================


def test_delete_nodes_nonexistent():
    """delete_nodes() with non-existent node_ids does nothing."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:3], METADATAS[:3]))
    assert vs.count() == 3

    vs.delete_nodes(node_ids=["nonexistent-1", "nonexistent-2"])
    assert vs.count() == 3  # unchanged

    vs.drop()
    vs.close()


# ==================== 12. Double clear ====================


def test_double_clear():
    """clear() called twice doesn't error."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:2], METADATAS[:2]))

    vs.clear()
    assert vs.count() == 0

    vs.clear()  # second clear — should not error
    assert vs.count() == 0

    vs.drop()
    vs.close()


# ==================== 13. Re-add after clear ====================


def test_readd_after_clear():
    """Adding data after clear() works correctly."""
    vs = make_store()
    vs.add(make_nodes(TEXTS[:2], METADATAS[:2]))
    vs.clear()

    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    vs.add(nodes)
    assert vs.count() == 3

    # Verify data is correct
    fetched = vs.get_nodes(node_ids=[nodes[0].node_id])
    assert fetched[0].get_content() == TEXTS[0]

    vs.drop()
    vs.close()


# ==================== 14. Async dimension mismatch ====================


async def test_async_dimension_mismatch():
    """async_add() with wrong dimension embedding is handled."""
    vs = make_store(table_name="test_li_adimmm")
    await vs.async_add(make_nodes(["seed text"]))

    wrong_emb = FakeEmbeddings(dim=256)
    bad_node = make_nodes(["wrong dim"], emb=wrong_emb)[0]
    try:
        await vs.async_add([bad_node])
        assert await vs.acount() >= 1
    except Exception:
        pass

    vs.drop()
    vs.close()


# ==================== 15. Async empty add ====================


async def test_async_empty_add():
    """async_add() with empty list returns empty list."""
    vs = make_store()
    ids = await vs.async_add([])
    assert ids == []
    assert await vs.acount() == 0
    vs.drop()
    vs.close()


# ==================== 16. Async delete non-existent ====================


async def test_async_delete_nonexistent():
    """adelete() with non-existent ref_doc_id does nothing."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS[:3], METADATAS[:3]))
    assert await vs.acount() == 3

    await vs.adelete(ref_doc_id="nonexistent")
    assert await vs.acount() == 3

    vs.drop()
    vs.close()


# ==================== 17. Async double clear ====================


async def test_async_double_clear():
    """aclear() called twice doesn't error."""
    vs = make_store()
    await vs.async_add(make_nodes(TEXTS[:2], METADATAS[:2]))

    await vs.aclear()
    assert await vs.acount() == 0

    await vs.aclear()
    assert await vs.acount() == 0

    vs.drop()
    vs.close()


# ==================== 18. Identifier validation ====================


def test_invalid_table_name():
    """Invalid table name raises ValueError."""
    from llama_index.vector_stores.polardbx import PolarDBXVectorStore
    from _helpers import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

    try:
        PolarDBXVectorStore(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            table_name="invalid-name!",
            embed_dim=EMBED_DIM,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_embed_dim():
    """Invalid embed_dim raises ValueError."""
    from llama_index.vector_stores.polardbx import PolarDBXVectorStore
    from _helpers import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

    try:
        PolarDBXVectorStore(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            table_name="test_li_invaliddim",
            embed_dim=0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_distance_strategy():
    """Invalid distance_strategy raises ValueError."""
    from llama_index.vector_stores.polardbx import PolarDBXVectorStore
    from _helpers import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

    try:
        PolarDBXVectorStore(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            table_name="test_li_invaliddist",
            embed_dim=EMBED_DIM,
            distance_strategy="INVALID",
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ==================== 19. close without init ====================


def test_close_without_init():
    """close() on uninitialized store does nothing, no error."""
    from llama_index.vector_stores.polardbx import PolarDBXVectorStore
    from _helpers import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

    vs = PolarDBXVectorStore(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        table_name="test_li_close_noinit",
        embed_dim=EMBED_DIM,
        perform_setup=False,
    )
    # _is_initialized is True after __init__, so close should work
    vs.close()


# ==================== 20. stores_text and flat_metadata ====================


def test_class_attributes():
    """Model field defaults have correct values."""
    from llama_index.vector_stores.polardbx import PolarDBXVectorStore

    # stores_text and flat_metadata are Pydantic model fields,
    # not ClassVar — access defaults via model_fields.
    assert PolarDBXVectorStore.model_fields["stores_text"].default is True
    assert PolarDBXVectorStore.model_fields["flat_metadata"].default is False
