"""Standard LlamaIndex partner package test.

Verifies that PolarDBXVectorStore inherits from BasePydanticVectorStore,
as required by the LlamaIndex integration package specification.
"""

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.polardbx import PolarDBXVectorStore


def test_class():
    names_of_base_classes = [
        b.__name__ for b in PolarDBXVectorStore.__mro__
    ]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes
