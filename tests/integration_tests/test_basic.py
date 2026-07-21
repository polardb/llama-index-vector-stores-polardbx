"""Tests for LlamaIndex PolarDBXVectorStore — basic CRUD operations (sync + async)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import (
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    EMB,
    EMBED_DIM,
    METADATAS,
    TEXTS,
    make_nodes,
    make_store,
)


# ==================== SYNC ====================


def test_sync_add_and_count():
    """add() inserts nodes and count() returns correct count."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    ids = vs.add(nodes)
    assert len(ids) == 5
    assert vs.count() == 5
    vs.drop()
    vs.close()


def test_sync_add_empty():
    """add() with empty list returns empty list."""
    vs = make_store()
    ids = vs.add([])
    assert len(ids) == 0
    vs.drop()
    vs.close()


def test_sync_get_nodes_by_ids():
    """get_nodes() by node_ids returns correct nodes."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    vs.add(nodes)

    fetched = vs.get_nodes(node_ids=[nodes[0].node_id, nodes[2].node_id])
    assert len(fetched) == 2
    vs.drop()
    vs.close()


def test_sync_get_nodes_all():
    """get_nodes() without args returns all nodes."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    vs.add(nodes)

    fetched = vs.get_nodes()
    assert len(fetched) == 5
    vs.drop()
    vs.close()


def test_sync_get_nodes_nonexistent():
    """get_nodes() with nonexistent IDs returns empty list."""
    vs = make_store()
    nodes = make_nodes(TEXTS[:2], METADATAS[:2])
    vs.add(nodes)

    fetched = vs.get_nodes(node_ids=["nonexistent-id-1", "nonexistent-id-2"])
    assert len(fetched) == 0
    vs.drop()
    vs.close()


def test_sync_delete_by_ref_doc_id():
    """delete() removes nodes by ref_doc_id."""
    vs = make_store()
    ref_ids = ["doc-1", "doc-2", "doc-3"]
    nodes = make_nodes(TEXTS[:3], METADATAS[:3], ref_doc_ids=ref_ids)
    vs.add(nodes)
    assert vs.count() == 3

    vs.delete(ref_doc_id="doc-2")
    assert vs.count() == 2

    # Verify the right node was deleted
    remaining = vs.get_nodes(node_ids=[nodes[0].node_id, nodes[1].node_id, nodes[2].node_id])
    remaining_ids = {n.node_id for n in remaining}
    assert nodes[1].node_id not in remaining_ids
    assert nodes[0].node_id in remaining_ids
    vs.drop()
    vs.close()


def test_sync_delete_nodes_by_ids():
    """delete_nodes() removes nodes by node_ids."""
    vs = make_store()
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    vs.add(nodes)
    assert vs.count() == 3

    vs.delete_nodes(node_ids=[nodes[0].node_id])
    assert vs.count() == 2
    vs.drop()
    vs.close()


def test_sync_delete_nodes_no_args():
    """delete_nodes() with no args does nothing."""
    vs = make_store()
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    vs.add(nodes)
    assert vs.count() == 3

    vs.delete_nodes()
    assert vs.count() == 3  # unchanged
    vs.drop()
    vs.close()


def test_sync_clear():
    """clear() truncates the table."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    vs.add(nodes)
    assert vs.count() == 5

    vs.clear()
    assert vs.count() == 0
    vs.drop()
    vs.close()


def test_sync_upsert_via_add():
    """add() with same node_id should UPSERT, not duplicate."""
    vs = make_store()
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    vs.add(nodes)
    assert vs.count() == 3

    # Re-add same nodes with updated text
    updated_nodes = make_nodes(
        ["UPDATED " + t for t in TEXTS[:3]],
        METADATAS[:3],
    )
    # Use the same node_ids as the original nodes
    for i, n in enumerate(updated_nodes):
        n.id_ = nodes[i].node_id
    vs.add(updated_nodes)
    assert vs.count() == 3  # still 3, not 6

    # Verify content was updated
    fetched = vs.get_nodes(node_ids=[nodes[0].node_id])
    assert "UPDATED" in fetched[0].get_content()
    vs.drop()
    vs.close()


def test_sync_persistence():
    """Data persists across store instances."""
    vs1 = make_store(table_name="test_li_persist", pre_delete=True)
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    vs1.add(nodes)
    assert vs1.count() == 3
    vs1.close()

    # New store pointing to same table — don't delete
    vs2 = make_store(table_name="test_li_persist", pre_delete=False)
    assert vs2.count() == 3
    vs2.drop()
    vs2.close()


def test_sync_from_params():
    """from_params() factory creates a working store."""
    from llama_index.vector_stores.polardbx import PolarDBXVectorStore

    vs = PolarDBXVectorStore.from_params(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        table_name="test_li_fromparams",
        embed_dim=EMBED_DIM,
    )
    nodes = make_nodes(TEXTS[:2], METADATAS[:2])
    vs.add(nodes)
    assert vs.count() == 2
    vs.drop()
    vs.close()


def test_sync_drop_and_close():
    """drop() drops table and close() cleans up."""
    vs = make_store(table_name="test_li_dropclose")
    nodes = make_nodes(TEXTS[:1], METADATAS[:1])
    vs.add(nodes)
    assert vs.count() == 1

    vs.drop()
    vs.close()

    # After drop, creating a new store with same name should work (table gone)
    vs2 = make_store(table_name="test_li_dropclose", pre_delete=False)
    assert vs2.count() == 0
    vs2.drop()
    vs2.close()


def test_sync_client_property():
    """client property returns the engine after init."""
    vs = make_store()
    assert vs.client is not None
    vs.drop()
    vs.close()


def test_sync_class_name():
    """class_name() returns the correct class name."""
    from llama_index.vector_stores.polardbx import PolarDBXVectorStore

    assert PolarDBXVectorStore.class_name() == "PolarDBXVectorStore"


# ==================== ASYNC ====================


async def test_async_add_and_count():
    """async_add() inserts nodes and acount() returns correct count."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    ids = await vs.async_add(nodes)
    assert len(ids) == 5

    cnt = await vs.acount()
    assert cnt == 5
    vs.drop()
    vs.close()


async def test_async_add_empty():
    """async_add() with empty list returns empty list."""
    vs = make_store()
    ids = await vs.async_add([])
    assert len(ids) == 0
    vs.drop()
    vs.close()


async def test_async_get_nodes_by_ids():
    """aget_nodes() by node_ids returns correct nodes."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    await vs.async_add(nodes)

    fetched = await vs.aget_nodes(
        node_ids=[nodes[1].node_id, nodes[3].node_id]
    )
    assert len(fetched) == 2
    vs.drop()
    vs.close()


async def test_async_get_nodes_all():
    """aget_nodes() without args returns all nodes."""
    vs = make_store()
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    await vs.async_add(nodes)

    fetched = await vs.aget_nodes()
    assert len(fetched) == 3
    vs.drop()
    vs.close()


async def test_async_delete_by_ref_doc_id():
    """adelete() removes nodes by ref_doc_id."""
    vs = make_store()
    ref_ids = ["adoc-1", "adoc-2", "adoc-3"]
    nodes = make_nodes(TEXTS[:3], METADATAS[:3], ref_doc_ids=ref_ids)
    await vs.async_add(nodes)
    assert await vs.acount() == 3

    await vs.adelete(ref_doc_id="adoc-1")
    assert await vs.acount() == 2
    vs.drop()
    vs.close()


async def test_async_delete_nodes_by_ids():
    """adelete_nodes() removes nodes by node_ids."""
    vs = make_store()
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    await vs.async_add(nodes)
    assert await vs.acount() == 3

    await vs.adelete_nodes(node_ids=[nodes[0].node_id, nodes[1].node_id])
    assert await vs.acount() == 1
    vs.drop()
    vs.close()


async def test_async_clear():
    """aclear() truncates the table."""
    vs = make_store()
    nodes = make_nodes(TEXTS, METADATAS)
    await vs.async_add(nodes)
    assert await vs.acount() == 5

    await vs.aclear()
    assert await vs.acount() == 0
    vs.drop()
    vs.close()


async def test_async_upsert_via_add():
    """async_add() with same node_id should UPSERT, not duplicate."""
    vs = make_store()
    nodes = make_nodes(TEXTS[:3], METADATAS[:3])
    await vs.async_add(nodes)
    assert await vs.acount() == 3

    # Re-add same nodes with updated text
    updated_nodes = make_nodes(
        ["ASYNC UPDATED " + t for t in TEXTS[:3]],
        METADATAS[:3],
    )
    for i, n in enumerate(updated_nodes):
        n.id_ = nodes[i].node_id
    await vs.async_add(updated_nodes)
    assert await vs.acount() == 3  # still 3, not 6

    fetched = await vs.aget_nodes(node_ids=[nodes[0].node_id])
    assert "ASYNC UPDATED" in fetched[0].get_content()
    vs.drop()
    vs.close()


async def test_async_drop_and_close():
    """adrop() drops table and aclose() cleans up."""
    vs = make_store(table_name="test_li_adropclose")
    nodes = make_nodes(TEXTS[:1], METADATAS[:1])
    await vs.async_add(nodes)
    assert await vs.acount() == 1

    await vs.adrop()
    # aclose() called inside adrop, but we can still call it
    # (it's idempotent when _is_initialized is False)
