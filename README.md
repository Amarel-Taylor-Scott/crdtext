# crdtext

> A convergent replicated text type (**RGA CRDT**) in pure Python. N replicas
> edit the same text concurrently, exchange operations in **any order**
> (delayed, duplicated, interleaved), and **provably converge** to identical
> text. Tombstone deletes, causal buffering, version vectors, safe compaction.
> Zero dependencies — the test suite is the proof.

```python
from crdtext import Doc

a, b = Doc("a"), Doc("b")
ops = a.insert(0, "hello")     # -> a batch of ops
b.apply(ops)                   # ship the batch to b (any order works)
a.apply(b.insert(5, " world"))
assert a.text() == b.text() == "hello world"
```

## Why

Local-first apps, offline editors, and multi-agent systems all need the same
thing: several parties edit shared text without a central server, go offline,
reconnect, and end up agreeing — no lost edits, no "resolve conflict" dialog.
A **CRDT** guarantees that mathematically: apply the same set of operations in
any order and every replica lands on the same document.

`crdtext` implements the classic **RGA** (Replicated Growable Array). It is
deliberately small and readable — the second reason it exists is that the best
way to *understand* CRDTs is one you can read end to end.

## Install

```bash
pip install -e .
```

Pure standard library, Python 3.9+. No dependencies.

## Quickstart — concurrent edits converge

```python
from crdtext import Doc

a, b, c = Doc("A"), Doc("B"), Doc("C")

base = a.insert(0, "the quick brown fox")   # A writes a line
b.apply(base); c.apply(base)                # everyone syncs

# three concurrent edits, each on a different replica:
e_a = a.delete(4, 6)              # A removes "quick "
e_b = b.insert(len(b), " jumps") # B appends
e_c = c.insert(0, "Wow: ")       # C prepends

# ship every batch to the replicas that haven't seen it — order irrelevant:
for op in reversed(e_a): b.apply(op); c.apply(op)   # even reversed
for op in e_b:           a.apply(op); c.apply(op)
for op in e_c:           a.apply(op); b.apply(op)

assert a.text() == b.text() == c.text()   # "Wow: the brown fox jumps"
```

Run the built-in three-replica demo (out-of-order + duplicated delivery,
self-checking):

```bash
crdtext demo        # or: python3 -m crdtext demo
```

## The ordering rule (the RGA tie-break)

Every element is a single code point with a globally unique id
`(replica_id, counter)`, where `counter` is a **Lamport clock** (bumped past
every counter the replica has seen). An insert names an **origin** — the id of
its visible predecessor at the moment it was typed (or `BEGIN` for index 0).

Integration places the new element **after its origin** and **before** any
existing sibling of the same origin whose id is *smaller*, where ids compare by
`(counter, replica_id)`:

```
origin O
  ├─ sibling with larger (counter, replica_id)   ← earlier in the text
  ├─ NEW element
  └─ sibling with smaller (counter, replica_id)   ← later in the text
```

Because counters are Lamport, an insert that already saw a sibling always
outranks it, and two *truly concurrent* inserts at the same spot are ordered
the same way on every replica (counter first, replica id breaks ties). That is
what makes concurrent edits commute. Two people typing whole words at the same
position get one word fully before the other — never interleaved letters
(`test_no_interleaving`).

## Delivery guarantees

`apply(op_or_batch)` tolerates the worst a network can do:

| Situation | What happens |
|---|---|
| Ops arrive **out of order** | Ops whose dependency (per-replica predecessor, insert origin, delete target) hasn't arrived are **buffered** and applied automatically once it does. |
| Ops arrive **duplicated** | Idempotent — a duplicate is a no-op (dense per-replica `seq` numbers detect it). |
| Ops arrive **interleaved** across replicas | Fine — each op integrates independently. |
| A whole batch is **delayed** for a while | Held in the pending buffer; `doc.pending_count` tells you how many. |

`version_vector()` returns `{replica_id: max counter applied}`; because
application is per-replica FIFO, `vv[r] >= c` means op `(r, c)` has landed.

## Compaction (forgetting tombstones)

Deletes leave **tombstones** so concurrent ops that still reference an element
keep working. `doc.compact(min_vv)` physically removes tombstones whose delete
op is covered by `min_vv` — a version vector every replica has met — and that
no locally buffered op still needs. `min_vv` must be **externally agreed at a
quiescent point** (e.g. the pointwise minimum of all replicas' version vectors
after a full sync). Ops generated afterward carry larger counters and integrate
identically on compacted and uncompacted replicas — verified by re-running a
convergence round *after* compaction (`test_new_edits_converge_after_compaction`).

## API

| Symbol | Purpose |
|---|---|
| `Doc(replica_id)` | A replica. |
| `.insert(index, text) -> [Op]` | Insert at a visible index; returns the op batch to ship. |
| `.delete(index, length=1) -> [Op]` | Tombstone visible elements; returns the op batch. |
| `.apply(op_or_batch) -> int` | Apply a remote op/batch (idempotent, causally buffered). |
| `.text()` / `len(doc)` | Visible text / visible length. |
| `.pending_count` / `.tombstone_count` | Buffered ops / stored tombstones. |
| `.version_vector()` | `{replica_id: max counter}`. |
| `.compact(min_vv) -> int` | Remove safe tombstones; returns count removed. |
| `.state_to_json()` / `Doc.state_from_json(s)` | Full-state (de)serialization. |
| `.merge(other_state_json)` | State-based merge (union) — converges both directions. |
| `op_to_json`/`op_from_json`, `ops_to_json`/`ops_from_json` | Op (batch) (de)serialization. |
| `BEGIN` | Origin sentinel for index 0. |

## Guarantees (each mapped to a test)

- **Convergence** — 100 seeded scenarios, N ∈ {2,3,5} replicas, 10–40 concurrent
  edits each, delivered in fully random order with 20% duplicates → byte-identical
  text, zero pending, equal version vectors on all replicas
  (`ConvergenceProperty.test_convergence_100_scenarios`).
- **Single-replica correctness** — 50-seed fuzz checked against a plain-string
  oracle after every edit (`test_fuzz_against_oracle_50_seeds`).
- **Idempotence** — applying a batch three times equals once (`Idempotence`).
- **Commutativity** — concurrent inserts at the same index, insert-vs-delete
  overlap, and double-delete of the same char converge regardless of apply order
  (`Commutativity`).
- **Causal buffering** — children delivered before origins, deletes before
  targets, and seq gaps all buffer then drain to the right text (`CausalBuffering`).
- **No interleaving** — concurrent contiguous runs stay contiguous (`NoInterleaving`).
- **Compaction safety** — text unchanged, tombstones→0, post-compaction edits
  still converge; uncovered deletes are kept (`Compaction`).
- **Serialization & merge** — state and op JSON roundtrip; state merge converges
  both directions (`Serialization`).
- **Unicode** — emoji + CJK converge at code-point granularity (`Unicode`).

## Complexity — reference-grade, not a rope

The document order is a Python list of elements plus an id→element dict. Inserts
and deletes are **O(n)** in the document length (list splice + a linear scan for
the RGA slot). That is perfectly fine for documents and teaching, and it keeps
the code readable. It is **not** a production rope/tree — for megabyte documents
or high-frequency streams you want a balanced-tree RGA or a block-wise CRDT.
This library optimizes for correctness you can read and prove, not throughput.

## Limitations

- **Character granularity** — one element per code point; no rich text,
  formatting, or block structure.
- **O(n) edits** — see above; not a rope.
- **Single-threaded** — no internal locking; drive one `Doc` from one thread.
- **Compaction needs external agreement** — `min_vv` must come from a real
  quiescent point; passing an over-eager vector is your responsibility (the
  library still won't remove a tombstone a pending op needs).
- **Whole-document state merge** is O(state size); it's for bootstrapping a new
  replica or reconciling two, not a per-keystroke sync path (use op batches for that).

## License

MIT — see [LICENSE](LICENSE).
