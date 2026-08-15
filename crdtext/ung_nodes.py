"""UNG node adapters for crdtext: RGA CRDT ops over JSON-serialized replica state."""

from __future__ import annotations

from typing import Any, Dict

from .core import Doc, ops_from_json, ops_to_json

_TAGS = ["license.mit", "runtime.python", "dependency-free"]


def ung_new_doc(replica_id: str) -> Dict[str, str]:
    """Create an empty replica and return its serialized state."""
    doc = Doc(replica_id)
    return {"state": doc.state_to_json(), "text": doc.text()}


def ung_edit(
    state: str, op: str, index: int, text: str = "", length: int = 1
) -> Dict[str, Any]:
    """Insert or delete at a visible index; returns the new state and the op batch to ship."""
    doc = Doc.state_from_json(state)
    if op == "insert":
        batch = doc.insert(index, text)
    elif op == "delete":
        batch = doc.delete(index, length)
    else:
        raise ValueError("op must be 'insert' or 'delete'")
    return {"state": doc.state_to_json(), "ops": ops_to_json(batch), "text": doc.text()}


def ung_apply(state: str, ops: str) -> Dict[str, Any]:
    """Apply a remote op batch (idempotent, causally buffered) to a serialized replica."""
    doc = Doc.state_from_json(state)
    applied = doc.apply(ops_from_json(ops))
    return {
        "state": doc.state_to_json(),
        "applied": applied,
        "text": doc.text(),
        "pending": doc.pending_count,
    }


def ung_merge(state_a: str, state_b: str) -> Dict[str, Any]:
    """State-based merge (union) of two replicas; converges from either direction."""
    doc = Doc.state_from_json(state_a)
    doc.merge(state_b)
    return {"state": doc.state_to_json(), "text": doc.text()}


def ung_info(state: str) -> Dict[str, Any]:
    """Report a replica's visible text, length, version vector, tombstones, and pending ops."""
    doc = Doc.state_from_json(state)
    return {
        "text": doc.text(),
        "length": len(doc),
        "version_vector": doc.version_vector(),
        "tombstones": doc.tombstone_count,
        "pending": doc.pending_count,
    }


def ung_compact(state: str, min_vv: Dict[str, int]) -> Dict[str, Any]:
    """Remove tombstones covered by an agreed minimum version vector; returns count removed."""
    doc = Doc.state_from_json(state)
    removed = doc.compact(min_vv)
    return {"state": doc.state_to_json(), "removed": removed, "text": doc.text()}


def _port(name: str, type_id: str, description: str) -> Dict[str, str]:
    return {"name": name, "type_id": type_id, "description": description}


def _param(name, value_type, default, required=False, choices=None):
    spec = {"name": name, "value_type": value_type, "default": default, "required": required}
    if choices is not None:
        spec["choices"] = choices
    return spec


def _node(fn, action, capability, summary, inputs, outputs, parameters=()):
    return {
        "fn": fn,
        "id": "amarel.crdtext." + action,
        "capabilities": [capability],
        "summary": summary,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "parameters": list(parameters),
        "effects": [],
        "determinism": "deterministic",
        "idempotency": "idempotent",
        "tags": list(_TAGS),
    }


_TEXT = "amarel.types.text"
_JSON = "amarel.types.json-value"
_STATE = "amarel.types.state"

NODES = [
    _node(
        ung_new_doc,
        "new-doc",
        "crdt.new-doc",
        "Create an empty RGA replica and return its serialized state.",
        [],
        [_port("state", _STATE, "Serialized replica state (JSON string)."),
         _port("text", _TEXT, "Visible text (empty).")],
        [_param("replica_id", "string", None, required=True)],
    ),
    _node(
        ung_edit,
        "edit-doc",
        "crdt.edit",
        "Insert or delete at a visible index; returns the new state and the op batch to ship.",
        [_port("state", _STATE, "Serialized replica state.")],
        [_port("state", _STATE, "Updated replica state."),
         _port("ops", _JSON, "The generated op batch (JSON string) to ship to other replicas."),
         _port("text", _TEXT, "Visible text after the edit.")],
        [_param("op", "string", None, required=True, choices=["insert", "delete"]),
         _param("index", "integer", None, required=True),
         _param("text", "string", ""),
         _param("length", "integer", 1)],
    ),
    _node(
        ung_apply,
        "apply-ops",
        "crdt.apply-ops",
        "Apply a remote op batch (idempotent, causally buffered) to a serialized replica.",
        [_port("state", _STATE, "Serialized replica state."),
         _port("ops", _JSON, "Op batch (JSON string) from another replica.")],
        [_port("state", _STATE, "Updated replica state."),
         _port("applied", _JSON, "How many ops were newly applied."),
         _port("text", _TEXT, "Visible text after application."),
         _port("pending", _JSON, "Ops still buffered awaiting causal dependencies.")],
    ),
    _node(
        ung_merge,
        "merge-docs",
        "crdt.merge",
        "State-based merge (union) of two replicas; converges from either direction.",
        [_port("state_a", _STATE, "Serialized state of the local replica."),
         _port("state_b", _STATE, "Serialized state of the remote replica.")],
        [_port("state", _STATE, "Merged replica state."),
         _port("text", _TEXT, "Visible text after the merge.")],
    ),
    _node(
        ung_info,
        "doc-info",
        "crdt.inspect",
        "Report a replica's visible text, length, version vector, tombstones, and pending ops.",
        [_port("state", _STATE, "Serialized replica state.")],
        [_port("text", _TEXT, "Visible text."),
         _port("length", _JSON, "Visible length."),
         _port("version_vector", _JSON, "{replica_id: max counter applied}."),
         _port("tombstones", _JSON, "Stored tombstone count."),
         _port("pending", _JSON, "Buffered op count.")],
    ),
    _node(
        ung_compact,
        "compact-doc",
        "crdt.compact",
        "Remove tombstones covered by an externally agreed minimum version vector.",
        [_port("state", _STATE, "Serialized replica state."),
         _port("min_vv", _JSON, "Version vector every replica has met.")],
        [_port("state", _STATE, "Compacted replica state."),
         _port("removed", _JSON, "Number of tombstones removed."),
         _port("text", _TEXT, "Visible text (unchanged by compaction).")],
    ),
]
