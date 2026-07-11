"""crdtext core — an RGA (Replicated Growable Array) text CRDT in pure Python.

Model
-----
A document is a sequence of single-code-point *elements*. Every element has a
globally unique id ``(replica_id, counter)`` where ``counter`` comes from a
Lamport clock (each replica's clock is bumped to at least the counter of every
op it applies, so a freshly generated op always carries a counter strictly
greater than every counter its replica has seen). Deleting never removes an
element immediately — it flips a *tombstone* flag, so concurrent operations
that still reference the element keep working.

Ordering rule (the RGA tie-break)
---------------------------------
An insert names an ``origin``: the id of the element that was its visible
predecessor at generation time (or ``BEGIN``, i.e. ``None``, for index 0).
The element is placed *after* its origin and *before* any existing sibling
with the same origin whose id ``(counter, replica_id)`` is smaller — i.e.
siblings sort by ``(counter, replica_id)`` **descending**. Because counters
are Lamport, a later insert that already saw a sibling always outranks it,
and truly concurrent inserts at the same spot are ordered the same way on
every replica (counter first, replica id as the deterministic tie-break).

Integration uses the classic RGA scan: start just after the origin and skip
every element whose ``(counter, replica_id)`` is greater than the new op's,
then insert. Under the Lamport property, every element of a higher-priority
sibling's subtree has a greater id and every element past the origin's region
has a smaller id, so the first smaller id is exactly the right slot.

Delivery
--------
``apply`` tolerates any delivery order and duplicates. Each op carries a dense
per-replica sequence number ``seq`` (1, 2, 3, ...): ops from one replica are
applied in ``seq`` order (a gap is buffered), an op whose ``seq`` is already
covered is a duplicate no-op, and an insert whose origin (or delete whose
target) is unknown is buffered until the dependency arrives. Buffered ops are
drained automatically. ``version_vector()`` maps each replica id to the
highest *counter* applied from it; because application is per-replica FIFO,
``vv[r] >= c`` really does mean op ``(r, c)`` has been applied.

Compaction
----------
``compact(min_vv)`` physically removes tombstones whose delete op is covered
by ``min_vv`` and that are not referenced by any locally buffered op. Because
the document order is materialized as a flat list (origins are never re-read
to order elements that are already integrated), removing an invisible element
never changes the text. ``min_vv`` must come from an external agreement made
at a quiescent point — see the README for the exact contract.

No wall-clock time, no randomness, no threads, no I/O. Python 3.9+ stdlib.
"""

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

__all__ = [
    "BEGIN",
    "Doc",
    "Op",
    "op_from_json",
    "op_to_json",
    "ops_from_json",
    "ops_to_json",
]

#: An element / op id: ``(replica_id, counter)``.
ElemId = Tuple[str, int]

#: Sentinel origin meaning "insert at the very beginning of the document".
BEGIN = None  # type: Optional[ElemId]

_INS = "ins"
_DEL = "del"

# _apply_one statuses
_APPLIED = "applied"
_DUP = "dup"
_PENDING = "pending"

_STATE_FORMAT = "crdtext-state-1"


@dataclass(frozen=True)
class Op:
    """One replicated operation: a single code point inserted or deleted.

    Fields
    ------
    kind:    ``"ins"`` or ``"del"``.
    replica: id of the replica that generated the op.
    counter: Lamport counter; ``(replica, counter)`` is the op's globally
             unique id (and, for inserts, the id of the element it creates).
    seq:     dense per-replica sequence number (1, 2, 3, ...) used for
             idempotence and in-order application.
    origin:  (inserts) id of the visible predecessor element at generation
             time, or ``None`` (= ``BEGIN``) for the start of the document.
    ch:      (inserts) the inserted code point (a length-1 ``str``).
    target:  (deletes) id of the element being tombstoned.
    """

    kind: str
    replica: str
    counter: int
    seq: int
    origin: Optional[ElemId] = None
    ch: Optional[str] = None
    target: Optional[ElemId] = None

    @property
    def id(self) -> ElemId:
        """Globally unique id of this op: ``(replica, counter)``."""
        return (self.replica, self.counter)


class _Element(object):
    """One document element (a code point), possibly tombstoned."""

    __slots__ = ("eid", "seq", "origin", "ch", "deleted", "deleted_by")

    def __init__(self, eid, seq, origin, ch, deleted=False, deleted_by=None):
        self.eid = eid                # (replica, counter)
        self.seq = seq                # seq of the insert op
        self.origin = origin          # (replica, counter) or None (BEGIN)
        self.ch = ch                  # single code point
        self.deleted = deleted        # tombstone flag
        self.deleted_by = deleted_by  # (replica, counter, seq) of delete op


class Doc(object):
    """A replicated text document (RGA CRDT).

    Create one ``Doc`` per replica with a unique ``replica_id``. Local edits
    (:meth:`insert` / :meth:`delete`) return op batches; ship every batch to
    every other replica (in any order, possibly duplicated or delayed) and
    feed them to :meth:`apply`. Once every replica has received every batch,
    all replicas hold byte-identical text.
    """

    def __init__(self, replica_id: str):
        if not isinstance(replica_id, str) or not replica_id:
            raise ValueError("replica_id must be a non-empty string")
        self.replica_id = replica_id
        self._clock = 0                 # Lamport clock: max counter applied
        self._elems = {}                # type: Dict[ElemId, _Element]
        self._order = []                # type: List[ElemId]  # doc order incl. tombstones
        self._visible = 0               # number of non-deleted elements
        self._tombstones = 0            # number of deleted elements still stored
        self._vv = {}                   # type: Dict[str, int]  # replica -> max counter applied
        self._seen_seq = {}             # type: Dict[str, int]  # replica -> contiguous seq applied
        # (replica, seq) -> (op, fifo_flag): ops waiting for a dependency
        self._pending = {}              # type: Dict[Tuple[str, int], Tuple[Op, bool]]

    # ------------------------------------------------------------------ #
    # local editing                                                       #
    # ------------------------------------------------------------------ #

    def insert(self, index: int, text: str) -> List[Op]:
        """Insert ``text`` at visible ``index``; return the op batch.

        Multi-character text becomes a chain of one-code-point ops: the first
        op's origin is the visible predecessor (or ``BEGIN``), each following
        op's origin is the previous op's element. The whole batch is returned
        as one list — ship it to the other replicas.
        """
        if not isinstance(text, str):
            raise TypeError("text must be str, got %r" % type(text).__name__)
        if not 0 <= index <= self._visible:
            raise IndexError(
                "insert index %d out of range 0..%d" % (index, self._visible))
        if not text:
            return []
        origin = BEGIN if index == 0 else self._visible_id_at(index - 1)
        ops = []
        for ch in text:
            self._clock += 1
            seq = self._seen_seq.get(self.replica_id, 0) + 1
            op = Op(_INS, self.replica_id, self._clock, seq, origin=origin, ch=ch)
            status = self._apply_one(op, True)
            if status != _APPLIED:  # pragma: no cover - internal invariant
                raise AssertionError("local insert did not apply: " + status)
            ops.append(op)
            origin = op.id
        return ops

    def delete(self, index: int, length: int = 1) -> List[Op]:
        """Tombstone ``length`` visible elements starting at ``index``.

        Returns the op batch (one delete op per element, each carrying the
        target element id).
        """
        if not isinstance(length, int) or length < 0:
            raise ValueError("length must be a non-negative int")
        if not 0 <= index <= self._visible - length or index < 0:
            raise IndexError(
                "delete range [%d, %d) out of range for visible length %d"
                % (index, index + length, self._visible))
        if length == 0:
            return []
        targets = self._visible_ids_range(index, length)
        ops = []
        for tid in targets:
            self._clock += 1
            seq = self._seen_seq.get(self.replica_id, 0) + 1
            op = Op(_DEL, self.replica_id, self._clock, seq, target=tid)
            status = self._apply_one(op, True)
            if status != _APPLIED:  # pragma: no cover - internal invariant
                raise AssertionError("local delete did not apply: " + status)
            ops.append(op)
        return ops

    # ------------------------------------------------------------------ #
    # remote application                                                  #
    # ------------------------------------------------------------------ #

    def apply(self, op_or_batch: Union[Op, Iterable[Op]]) -> int:
        """Apply a remote op or a batch (iterable) of ops.

        Idempotent: an op already applied is a no-op. Causally buffered: an
        op that arrives before its dependency (per-replica predecessor,
        insert origin, or delete target) is held in the pending buffer and
        applied automatically once the dependency arrives. Returns the number
        of ops actually applied now (including drained pending ops).
        """
        if isinstance(op_or_batch, Op):
            ops = [op_or_batch]  # type: List[Op]
        else:
            ops = list(op_or_batch)
        applied = 0
        for op in ops:
            if not isinstance(op, Op):
                raise TypeError("apply() takes an Op or an iterable of Ops")
            if self._apply_one(op, True) == _APPLIED:
                applied += 1 + self._drain_pending()
        return applied

    @property
    def pending_count(self) -> int:
        """Number of ops buffered while waiting for a dependency."""
        return len(self._pending)

    @property
    def tombstone_count(self) -> int:
        """Number of deleted elements still physically stored."""
        return self._tombstones

    # ------------------------------------------------------------------ #
    # reading                                                             #
    # ------------------------------------------------------------------ #

    def text(self) -> str:
        """The visible text (tombstones excluded)."""
        elems = self._elems
        return "".join(
            elems[eid].ch for eid in self._order if not elems[eid].deleted)

    def __len__(self) -> int:
        return self._visible

    def __repr__(self) -> str:
        return "Doc(replica_id=%r, len=%d, tombstones=%d, pending=%d)" % (
            self.replica_id, self._visible, self._tombstones, len(self._pending))

    def version_vector(self) -> Dict[str, int]:
        """``{replica_id: max counter applied from that replica}``.

        Ops from one replica are applied in order, so ``vv[r] >= c`` means op
        ``(r, c)`` (and every earlier op from ``r``) has been applied.
        """
        return dict(self._vv)

    # ------------------------------------------------------------------ #
    # compaction                                                          #
    # ------------------------------------------------------------------ #

    def compact(self, min_vv: Dict[str, int]) -> int:
        """Physically remove tombstones that are safe to forget.

        A tombstone is removed iff:

        * its delete op ``(r, c)`` is covered by ``min_vv`` (``min_vv[r] >= c``),
          i.e. every replica has applied the delete (which implies every
          replica has also applied the insert), and
        * no locally buffered (pending) op references it as origin or target.

        ``min_vv`` must be externally agreed at a quiescent point — a moment
        where every op generated by any replica has been delivered to every
        replica (e.g. the pointwise minimum of all replicas' version vectors
        after a full sync). Ops generated *after* that point carry Lamport
        counters larger than every compacted id and integrate identically on
        compacted and uncompacted replicas. Returns the number of tombstones
        removed.
        """
        if not isinstance(min_vv, dict):
            raise TypeError("min_vv must be a dict {replica_id: counter}")
        blocked = set()
        for op, _fifo in self._pending.values():
            if op.origin is not None:
                blocked.add(op.origin)
            if op.target is not None:
                blocked.add(op.target)
        removed = []
        for eid in self._order:
            el = self._elems[eid]
            if not el.deleted or el.deleted_by is None:
                continue
            dr, dc = el.deleted_by[0], el.deleted_by[1]
            if min_vv.get(dr, 0) < dc:
                continue  # some replica may not have seen the delete yet
            if eid in blocked:
                continue  # a buffered op still needs this element
            removed.append(eid)
        if removed:
            gone = set(removed)
            self._order = [eid for eid in self._order if eid not in gone]
            for eid in removed:
                del self._elems[eid]
            self._tombstones -= len(removed)
        return len(removed)

    # ------------------------------------------------------------------ #
    # serialization                                                       #
    # ------------------------------------------------------------------ #

    def state_to_json(self) -> str:
        """Full state as canonical JSON: elements (incl. tombstones) in
        document order, version vector, sequence high-water marks, Lamport
        clock, and the pending buffer."""
        els = []
        for eid in self._order:
            el = self._elems[eid]
            els.append([
                eid[0], eid[1], el.seq,
                list(el.origin) if el.origin is not None else None,
                el.ch,
                1 if el.deleted else 0,
                list(el.deleted_by) if el.deleted_by is not None else None,
            ])
        pend = []
        for key in sorted(self._pending):
            op, fifo = self._pending[key]
            pend.append([_op_to_dict(op), 1 if fifo else 0])
        state = {
            "format": _STATE_FORMAT,
            "replica_id": self.replica_id,
            "clock": self._clock,
            "vv": self._vv,
            "seen_seq": self._seen_seq,
            "elements": els,
            "pending": pend,
        }
        return json.dumps(state, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))

    @classmethod
    def state_from_json(cls, s: str) -> "Doc":
        """Reconstruct a :class:`Doc` exactly from :meth:`state_to_json`."""
        st = _parse_state(s)
        doc = cls(st["replica_id"])
        doc._clock = st["clock"]
        doc._vv = dict(st["vv"])
        doc._seen_seq = dict(st["seen_seq"])
        for (rid, counter, seq, origin, ch, deleted, deleted_by) in st["elements"]:
            eid = (rid, counter)
            if eid in doc._elems:
                raise ValueError("duplicate element id in state: %r" % (eid,))
            doc._elems[eid] = _Element(eid, seq, origin, ch, deleted, deleted_by)
            doc._order.append(eid)
            if deleted:
                doc._tombstones += 1
            else:
                doc._visible += 1
        for op, fifo in st["pending"]:
            doc._pending[(op.replica, op.seq)] = (op, fifo)
        return doc

    def merge(self, other_state_json: str) -> None:
        """State-based merge: union this replica with a peer's full state.

        Every element (live or tombstone) present in either state is present
        afterwards; an element deleted in either state is deleted; version
        vectors, sequence marks and the Lamport clock take the pointwise
        maximum; the peer's pending ops are ingested. Merging A into B and B
        into A yields identical text. Peers must be uncompacted or compacted
        at the same agreed point (see README) — otherwise elements whose
        origin was compacted away on both sides stay buffered.
        """
        st = _parse_state(other_state_json)
        # 1. union of elements, walked in the peer's document order so that
        #    an element's origin (its parent) is always ingested before it.
        for (rid, counter, seq, origin, ch, deleted, deleted_by) in st["elements"]:
            eid = (rid, counter)
            if eid not in self._elems:
                self._apply_one(
                    Op(_INS, rid, counter, seq, origin=origin, ch=ch), False)
            if deleted and deleted_by is not None:
                self._apply_one(
                    Op(_DEL, deleted_by[0], deleted_by[1], deleted_by[2],
                       target=eid), False)
        # 2. pointwise maxima of the delivery bookkeeping.
        for rid, counter in st["vv"].items():
            if counter > self._vv.get(rid, 0):
                self._vv[rid] = counter
        for rid, seq in st["seen_seq"].items():
            if seq > self._seen_seq.get(rid, 0):
                self._seen_seq[rid] = seq
        if st["clock"] > self._clock:
            self._clock = st["clock"]
        # 3. the peer's pending ops are real ops we may not have seen.
        for op, fifo in st["pending"]:
            self._apply_one(op, fifo)
        # 4. anything unblocked by the union applies now.
        self._drain_pending()

    # ------------------------------------------------------------------ #
    # internals                                                           #
    # ------------------------------------------------------------------ #

    def _apply_one(self, op: Op, fifo: bool) -> str:
        """Apply one op. Returns ``"applied"``, ``"dup"`` or ``"pending"``.

        ``fifo=True`` is the op-based path: per-replica in-order delivery is
        enforced via ``seq`` and duplicates are dropped via the seq
        high-water mark. ``fifo=False`` is the state-merge path: dedup is by
        element presence and bookkeeping is handled by :meth:`merge`.
        """
        if op.kind not in (_INS, _DEL):
            raise ValueError("unknown op kind: %r" % (op.kind,))
        replica = op.replica
        if fifo:
            seen = self._seen_seq.get(replica, 0)
            if op.seq <= seen:
                return _DUP
            if op.seq != seen + 1:
                self._pending[(replica, op.seq)] = (op, True)
                return _PENDING
        if op.kind == _INS:
            if op.id in self._elems:
                # element already here (e.g. it arrived via a state merge);
                # advance the op bookkeeping so FIFO progress isn't stuck.
                if fifo:
                    self._bookkeep(op)
                return _DUP
            if op.origin is not None and op.origin not in self._elems:
                self._pending[(replica, op.seq)] = (op, fifo)
                return _PENDING
            self._integrate_insert(op)
        else:
            if op.target not in self._elems:
                self._pending[(replica, op.seq)] = (op, fifo)
                return _PENDING
            self._integrate_delete(op)
        if op.counter > self._clock:
            self._clock = op.counter
        if fifo:
            self._bookkeep(op)
        return _APPLIED

    def _bookkeep(self, op: Op) -> None:
        self._seen_seq[op.replica] = op.seq
        if op.counter > self._vv.get(op.replica, 0):
            self._vv[op.replica] = op.counter
        if op.counter > self._clock:
            self._clock = op.counter

    def _integrate_insert(self, op: Op) -> None:
        """Place the new element with the RGA rule (see module docstring)."""
        if not isinstance(op.ch, str) or len(op.ch) != 1:
            raise ValueError("insert op must carry exactly one code point")
        order = self._order
        if op.origin is None:
            i = 0
        else:
            i = order.index(op.origin) + 1
        key = (op.counter, op.replica)
        n = len(order)
        while i < n:
            other = order[i]  # (replica, counter)
            if (other[1], other[0]) > key:
                i += 1  # higher-priority sibling or part of its subtree
            else:
                break
        eid = op.id
        order.insert(i, eid)
        self._elems[eid] = _Element(eid, op.seq, op.origin, op.ch)
        self._visible += 1

    def _integrate_delete(self, op: Op) -> None:
        el = self._elems[op.target]
        stamp = (op.replica, op.counter, op.seq)
        if not el.deleted:
            el.deleted = True
            el.deleted_by = stamp
            self._visible -= 1
            self._tombstones += 1
        elif stamp < el.deleted_by:
            # concurrent deletes of the same element: keep the smallest
            # delete-op id so every replica stores the identical tombstone.
            el.deleted_by = stamp

    def _drain_pending(self) -> int:
        """Re-try buffered ops until no more progress. Returns ops applied."""
        applied = 0
        progress = True
        while progress and self._pending:
            progress = False
            for key in list(self._pending):
                entry = self._pending.pop(key, None)
                if entry is None:  # pragma: no cover - defensive
                    continue
                op, fifo = entry
                status = self._apply_one(op, fifo)  # re-buffers if still blocked
                if status == _APPLIED:
                    applied += 1
                    progress = True
                elif status == _DUP:
                    progress = True  # buffer shrank; keep sweeping
        return applied

    def _visible_id_at(self, vindex: int) -> ElemId:
        n = -1
        elems = self._elems
        for eid in self._order:
            if not elems[eid].deleted:
                n += 1
                if n == vindex:
                    return eid
        raise IndexError("visible index %d out of range" % vindex)

    def _visible_ids_range(self, start: int, length: int) -> List[ElemId]:
        out = []  # type: List[ElemId]
        n = -1
        elems = self._elems
        for eid in self._order:
            if not elems[eid].deleted:
                n += 1
                if n >= start:
                    out.append(eid)
                    if len(out) == length:
                        return out
        raise IndexError(
            "visible range [%d, %d) out of range" % (start, start + length))


# ---------------------------------------------------------------------- #
# op / state JSON                                                         #
# ---------------------------------------------------------------------- #

def _op_to_dict(op: Op) -> Dict[str, object]:
    d = {
        "kind": op.kind,
        "replica": op.replica,
        "counter": op.counter,
        "seq": op.seq,
    }  # type: Dict[str, object]
    if op.kind == _INS:
        d["origin"] = list(op.origin) if op.origin is not None else None
        d["ch"] = op.ch
    else:
        d["target"] = list(op.target) if op.target is not None else None
    return d


def _elem_id_from_json(v, what):
    if v is None:
        return None
    if (not isinstance(v, (list, tuple)) or len(v) != 2
            or not isinstance(v[0], str) or not v[0]
            or not isinstance(v[1], int)):
        raise ValueError("bad %s in op/state JSON: %r" % (what, v))
    return (v[0], v[1])


def _op_from_dict(d) -> Op:
    if not isinstance(d, dict):
        raise ValueError("op JSON must be an object, got %r" % type(d).__name__)
    kind = d.get("kind")
    replica = d.get("replica")
    counter = d.get("counter")
    seq = d.get("seq")
    if kind not in (_INS, _DEL):
        raise ValueError("bad op kind: %r" % (kind,))
    if not isinstance(replica, str) or not replica:
        raise ValueError("bad op replica: %r" % (replica,))
    if not isinstance(counter, int) or counter < 1:
        raise ValueError("bad op counter: %r" % (counter,))
    if not isinstance(seq, int) or seq < 1:
        raise ValueError("bad op seq: %r" % (seq,))
    if kind == _INS:
        ch = d.get("ch")
        if not isinstance(ch, str) or len(ch) != 1:
            raise ValueError("bad op ch (must be one code point): %r" % (ch,))
        origin = _elem_id_from_json(d.get("origin"), "origin")
        return Op(_INS, replica, counter, seq, origin=origin, ch=ch)
    target = _elem_id_from_json(d.get("target"), "target")
    if target is None:
        raise ValueError("delete op is missing its target")
    return Op(_DEL, replica, counter, seq, target=target)


def op_to_json(op: Op) -> str:
    """Serialize one :class:`Op` to a JSON string."""
    if not isinstance(op, Op):
        raise TypeError("op_to_json() takes an Op")
    return json.dumps(_op_to_dict(op), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def op_from_json(s: str) -> Op:
    """Parse one :class:`Op` from :func:`op_to_json` output."""
    return _op_from_dict(json.loads(s))


def ops_to_json(ops: Iterable[Op]) -> str:
    """Serialize an op batch (any iterable of :class:`Op`) to JSON."""
    out = []
    for op in ops:
        if not isinstance(op, Op):
            raise TypeError("ops_to_json() takes an iterable of Ops")
        out.append(_op_to_dict(op))
    return json.dumps({"ops": out}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def ops_from_json(s: str) -> List[Op]:
    """Parse an op batch from :func:`ops_to_json` output."""
    d = json.loads(s)
    if not isinstance(d, dict) or not isinstance(d.get("ops"), list):
        raise ValueError("batch JSON must be an object with an 'ops' list")
    return [_op_from_dict(o) for o in d["ops"]]


def _int_map_from_json(v, what) -> Dict[str, int]:
    if not isinstance(v, dict):
        raise ValueError("bad %s in state JSON" % what)
    out = {}
    for k, n in v.items():
        if not isinstance(k, str) or not k or not isinstance(n, int) or n < 0:
            raise ValueError("bad %s entry in state JSON: %r -> %r" % (what, k, n))
        out[k] = n
    return out


def _parse_state(s: str) -> Dict[str, object]:
    d = json.loads(s)
    if not isinstance(d, dict) or d.get("format") != _STATE_FORMAT:
        raise ValueError("not a %s document" % _STATE_FORMAT)
    replica_id = d.get("replica_id")
    if not isinstance(replica_id, str) or not replica_id:
        raise ValueError("bad replica_id in state JSON")
    clock = d.get("clock")
    if not isinstance(clock, int) or clock < 0:
        raise ValueError("bad clock in state JSON")
    elements = []
    raw_els = d.get("elements")
    if not isinstance(raw_els, list):
        raise ValueError("bad elements in state JSON")
    for row in raw_els:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("bad element row in state JSON: %r" % (row,))
        rid, counter, seq, origin, ch, deleted, deleted_by = row
        if (not isinstance(rid, str) or not rid
                or not isinstance(counter, int) or counter < 1
                or not isinstance(seq, int) or seq < 1
                or not isinstance(ch, str) or len(ch) != 1
                or deleted not in (0, 1)):
            raise ValueError("bad element row in state JSON: %r" % (row,))
        origin_t = _elem_id_from_json(origin, "element origin")
        if deleted_by is None:
            db_t = None
        else:
            if (not isinstance(deleted_by, list) or len(deleted_by) != 3
                    or not isinstance(deleted_by[0], str) or not deleted_by[0]
                    or not isinstance(deleted_by[1], int)
                    or not isinstance(deleted_by[2], int)):
                raise ValueError("bad deleted_by in state JSON: %r" % (deleted_by,))
            db_t = (deleted_by[0], deleted_by[1], deleted_by[2])
        if deleted == 1 and db_t is None:
            raise ValueError("tombstone without deleted_by in state JSON")
        elements.append((rid, counter, seq, origin_t, ch, deleted == 1, db_t))
    pending = []
    raw_pend = d.get("pending")
    if not isinstance(raw_pend, list):
        raise ValueError("bad pending in state JSON")
    for row in raw_pend:
        if not isinstance(row, list) or len(row) != 2 or row[1] not in (0, 1):
            raise ValueError("bad pending row in state JSON: %r" % (row,))
        pending.append((_op_from_dict(row[0]), row[1] == 1))
    return {
        "replica_id": replica_id,
        "clock": clock,
        "vv": _int_map_from_json(d.get("vv"), "vv"),
        "seen_seq": _int_map_from_json(d.get("seen_seq"), "seen_seq"),
        "elements": elements,
        "pending": pending,
    }
