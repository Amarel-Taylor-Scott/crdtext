"""crdtext — a convergent replicated text type (RGA CRDT) in pure Python.

N replicas edit the same text concurrently, exchange operations in any order
(delayed, duplicated, interleaved), and provably converge to identical text.
Tombstone deletes, causal buffering for out-of-order delivery, version
vectors, and safe tombstone compaction. Zero dependencies.

    >>> from crdtext import Doc
    >>> a, b = Doc("a"), Doc("b")
    >>> ops = a.insert(0, "hello")
    >>> _ = b.apply(ops)
    >>> b.text()
    'hello'
"""

from .core import (
    BEGIN,
    Doc,
    Op,
    op_from_json,
    op_to_json,
    ops_from_json,
    ops_to_json,
)

__version__ = "0.1.0"

__all__ = [
    "BEGIN",
    "Doc",
    "Op",
    "op_from_json",
    "op_to_json",
    "ops_from_json",
    "ops_to_json",
    "__version__",
]
