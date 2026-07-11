"""crdtext test suite.

The convergence property suite (``test_convergence_*``) is the proof: many
seeded scenarios with N replicas making concurrent edits delivered in random
orders, with duplicates and delays, must all converge to byte-identical text.
Everything is deterministic — randomness comes only from seeded
``random.Random`` instances.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crdtext import (  # noqa: E402
    BEGIN,
    Doc,
    Op,
    op_from_json,
    op_to_json,
    ops_from_json,
    ops_to_json,
)

ALPHABET = "abcde \n"


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def do_random_local(doc, rng, n, alphabet=ALPHABET):
    """Perform ``n`` random local edits on ``doc``; return the list of batches
    (each batch a list of Ops). Also maintain nothing else — the doc is the
    truth for its own text."""
    batches = []
    for _ in range(n):
        vis = len(doc)
        if vis == 0 or rng.random() < 0.68:
            idx = rng.randint(0, vis)
            length = rng.randint(1, 4)
            text = "".join(rng.choice(alphabet) for _ in range(length))
            batch = doc.insert(idx, text)
        else:
            start = rng.randint(0, vis - 1)
            length = rng.randint(1, min(3, vis - start))
            batch = doc.delete(start, length)
        if batch:
            batches.append(batch)
    return batches


def deliver_scrambled(target, ops, rng, dup_rate=0.2):
    """Deliver ``ops`` (a flat list) to ``target`` in fully random order, with
    a fraction duplicated — the hardest delivery discipline: any order, dupes,
    interleaving. apply() must buffer and de-dupe."""
    stream = list(ops)
    rng.shuffle(stream)
    dupes = [op for op in stream if rng.random() < dup_rate]
    stream.extend(dupes)
    rng.shuffle(stream)
    for op in stream:
        target.apply(op)


class OracleString:
    """A plain-string mirror of single-replica edits, for cross-checking."""

    def __init__(self):
        self.s = ""

    def insert(self, idx, text):
        self.s = self.s[:idx] + text + self.s[idx:]

    def delete(self, idx, length):
        self.s = self.s[:idx] + self.s[idx + length:]


# --------------------------------------------------------------------------- #
# 1. single-replica correctness vs a plain-string oracle                      #
# --------------------------------------------------------------------------- #

class SingleReplicaOracle(unittest.TestCase):
    def test_basic_insert_delete(self):
        d = Doc("r")
        d.insert(0, "hello world")
        self.assertEqual(d.text(), "hello world")
        d.delete(5, 6)
        self.assertEqual(d.text(), "hello")
        d.insert(5, "!")
        self.assertEqual(d.text(), "hello!")

    def test_insert_middle_and_ends(self):
        d = Doc("r")
        d.insert(0, "AC")
        d.insert(1, "B")
        self.assertEqual(d.text(), "ABC")
        d.insert(3, "D")
        self.assertEqual(d.text(), "ABCD")
        d.insert(0, "Z")
        self.assertEqual(d.text(), "ZABCD")

    def test_len_tracks_visible(self):
        d = Doc("r")
        d.insert(0, "abcdef")
        self.assertEqual(len(d), 6)
        d.delete(2, 3)
        self.assertEqual(len(d), 3)
        self.assertEqual(d.text(), "abf")

    def test_out_of_range_raises(self):
        d = Doc("r")
        d.insert(0, "abc")
        with self.assertRaises(IndexError):
            d.insert(4, "x")
        with self.assertRaises(IndexError):
            d.delete(2, 5)
        with self.assertRaises(IndexError):
            d.delete(3, 1)

    def test_empty_ops(self):
        d = Doc("r")
        self.assertEqual(d.insert(0, ""), [])
        d.insert(0, "ab")
        self.assertEqual(d.delete(0, 0), [])

    def test_fuzz_against_oracle_50_seeds(self):
        for seed in range(50):
            rng = random.Random(seed)
            d = Doc("r%d" % seed)
            oracle = OracleString()
            for _ in range(60):
                vis = len(d)
                if vis == 0 or rng.random() < 0.68:
                    idx = rng.randint(0, vis)
                    text = "".join(
                        rng.choice(ALPHABET) for _ in range(rng.randint(1, 4)))
                    d.insert(idx, text)
                    oracle.insert(idx, text)
                else:
                    start = rng.randint(0, vis - 1)
                    length = rng.randint(1, min(3, vis - start))
                    d.delete(start, length)
                    oracle.delete(start, length)
                self.assertEqual(d.text(), oracle.s,
                                 "seed %d diverged from oracle" % seed)


# --------------------------------------------------------------------------- #
# 2. two-replica sync                                                          #
# --------------------------------------------------------------------------- #

class TwoReplicaSync(unittest.TestCase):
    def test_sequential_exchange(self):
        a, b = Doc("a"), Doc("b")
        b.apply(a.insert(0, "hello"))
        self.assertEqual(b.text(), "hello")
        a.apply(b.insert(5, " world"))
        self.assertEqual(a.text(), "hello world")
        self.assertEqual(a.text(), b.text())

    def test_bidirectional_concurrent_then_sync(self):
        a, b = Doc("a"), Doc("b")
        base = a.insert(0, "shared")
        b.apply(base)
        # concurrent
        oa = a.insert(6, "-A")
        ob = b.insert(0, "B-")
        a.apply(ob)
        b.apply(oa)
        self.assertEqual(a.text(), b.text())
        self.assertEqual(a.version_vector(), b.version_vector())


# --------------------------------------------------------------------------- #
# 3. the convergence property (flagship)                                       #
# --------------------------------------------------------------------------- #

class ConvergenceProperty(unittest.TestCase):
    def _run_scenario(self, seed, n_replicas, shared_prefix):
        rng = random.Random(seed)
        replicas = [Doc("R%d" % i) for i in range(n_replicas)]

        if shared_prefix:
            prefix = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(4, 12)))
            base = replicas[0].insert(0, prefix)
            for r in replicas[1:]:
                r.apply(base)

        # each replica makes concurrent local edits on its own copy
        all_ops = []
        for r in replicas:
            for batch in do_random_local(r, rng, rng.randint(10, 40)):
                all_ops.extend(batch)

        # cross-deliver: each replica receives every op it did not originate,
        # in a fully random order with duplicates.
        for target in replicas:
            foreign = [op for op in all_ops if op.replica != target.replica_id]
            deliver_scrambled(target, foreign, rng)

        texts = {r.text() for r in replicas}
        self.assertEqual(
            len(texts), 1,
            "seed=%d N=%d prefix=%s DIVERGED: %r"
            % (seed, n_replicas, shared_prefix, sorted(texts)))
        for r in replicas:
            self.assertEqual(r.pending_count, 0,
                             "seed=%d left pending ops on %s" % (seed, r.replica_id))
        vvs = [r.version_vector() for r in replicas]
        for vv in vvs[1:]:
            self.assertEqual(vv, vvs[0],
                             "seed=%d version vectors differ" % seed)

    def test_convergence_100_scenarios(self):
        counts = [2, 3, 5]
        for seed in range(100):
            n = counts[seed % len(counts)]
            self._run_scenario(seed, n, shared_prefix=(seed % 2 == 0))


# --------------------------------------------------------------------------- #
# 4. idempotence                                                               #
# --------------------------------------------------------------------------- #

class Idempotence(unittest.TestCase):
    def test_apply_thrice_equals_once(self):
        a, b = Doc("a"), Doc("b")
        ops = a.insert(0, "duplicated")
        b.apply(ops)
        once = b.text()
        vv_once = b.version_vector()
        b.apply(ops)
        b.apply(ops)
        self.assertEqual(b.text(), once)
        self.assertEqual(b.version_vector(), vv_once)
        self.assertEqual(b.pending_count, 0)

    def test_duplicate_single_ops_interleaved(self):
        a, b = Doc("a"), Doc("b")
        ops = a.insert(0, "abcdef")
        for op in ops + ops + ops:
            b.apply(op)
        self.assertEqual(b.text(), "abcdef")


# --------------------------------------------------------------------------- #
# 5. commutativity spot-checks                                                 #
# --------------------------------------------------------------------------- #

class Commutativity(unittest.TestCase):
    def test_concurrent_insert_same_index_order_independent(self):
        a, b = Doc("a"), Doc("b")
        oa = a.insert(0, "A")
        ob = b.insert(0, "B")
        # apply in opposite orders on two fresh replicas
        x = Doc("x")
        x.apply(oa)
        x.apply(ob)
        y = Doc("y")
        y.apply(ob)
        y.apply(oa)
        self.assertEqual(x.text(), y.text())

    def test_insert_vs_delete_overlap(self):
        a, b = Doc("a"), Doc("b")
        base = a.insert(0, "xyz")
        b.apply(base)
        od = a.delete(1, 1)          # delete 'y'
        oi = b.insert(1, "-")        # insert before 'y'
        a.apply(oi)
        b.apply(od)
        self.assertEqual(a.text(), b.text())

    def test_delete_same_char_twice_concurrently(self):
        a, b = Doc("a"), Doc("b")
        base = a.insert(0, "hi")
        b.apply(base)
        da = a.delete(0, 1)
        db = b.delete(0, 1)          # both delete 'h'
        a.apply(db)
        b.apply(da)
        self.assertEqual(a.text(), "i")
        self.assertEqual(b.text(), "i")
        self.assertEqual(a.text(), b.text())


# --------------------------------------------------------------------------- #
# 6. causal buffering                                                          #
# --------------------------------------------------------------------------- #

class CausalBuffering(unittest.TestCase):
    def test_child_before_origin_buffers_then_drains(self):
        a, b = Doc("a"), Doc("b")
        ops = a.insert(0, "chain")     # 5 chained ops
        # deliver strictly reversed: every op's origin arrives after it
        for op in reversed(ops):
            b.apply(op)
        self.assertEqual(b.text(), "chain")
        self.assertEqual(b.pending_count, 0)

    def test_delete_before_target_buffers(self):
        a, b = Doc("a"), Doc("b")
        ins = a.insert(0, "gone")
        dele = a.delete(0, 4)
        # deliver the deletes first (targets unknown), then the inserts
        for op in dele:
            b.apply(op)
        self.assertGreater(b.pending_count, 0)
        for op in ins:
            b.apply(op)
        self.assertEqual(b.text(), "")
        self.assertEqual(b.pending_count, 0)

    def test_seq_gap_buffers(self):
        a, b = Doc("a"), Doc("b")
        ops = a.insert(0, "abcdef")
        # skip the first op (seq gap) then deliver it last
        for op in ops[1:]:
            b.apply(op)
        self.assertGreater(b.pending_count, 0)
        b.apply(ops[0])
        self.assertEqual(b.text(), "abcdef")
        self.assertEqual(b.pending_count, 0)


# --------------------------------------------------------------------------- #
# 7. no interleaving of concurrent contiguous runs                             #
# --------------------------------------------------------------------------- #

class NoInterleaving(unittest.TestCase):
    def test_two_runs_stay_contiguous(self):
        a, b = Doc("a"), Doc("b")
        oa = a.insert(0, "abc")
        ob = b.insert(0, "xyz")
        a.apply(ob)
        b.apply(oa)
        self.assertEqual(a.text(), b.text())
        self.assertIn(a.text(), ("abcxyz", "xyzabc"))

    def test_many_runs_contiguous_property(self):
        for seed in range(20):
            rng = random.Random(1000 + seed)
            reps = [Doc("R%d" % i) for i in range(3)]
            words = ["aaa", "bbb", "ccc"]
            batches = [reps[i].insert(0, words[i]) for i in range(3)]
            flat = [op for batch in batches for op in batch]
            for target in reps:
                foreign = [op for op in flat if op.replica != target.replica_id]
                deliver_scrambled(target, foreign, rng)
            self.assertEqual(len({r.text() for r in reps}), 1)
            text = reps[0].text()
            # every word must appear as a contiguous block
            for w in words:
                self.assertIn(w, text, "word %r interleaved in %r" % (w, text))


# --------------------------------------------------------------------------- #
# 8. compaction                                                                #
# --------------------------------------------------------------------------- #

class Compaction(unittest.TestCase):
    def _min_vv(self, replicas):
        keys = set()
        for r in replicas:
            keys |= set(r.version_vector().keys())
        out = {}
        for k in keys:
            out[k] = min(r.version_vector().get(k, 0) for r in replicas)
        return out

    def test_compact_preserves_text_and_drops_tombstones(self):
        a, b = Doc("a"), Doc("b")
        base = a.insert(0, "keep and drop this")
        b.apply(base)
        dele = a.delete(4, 9)        # remove "and drop "
        b.apply(dele)
        self.assertEqual(a.text(), b.text())
        before = a.text()
        self.assertGreater(a.tombstone_count, 0)
        mvv = self._min_vv([a, b])
        a.compact(mvv)
        b.compact(mvv)
        self.assertEqual(a.text(), before)
        self.assertEqual(b.text(), before)
        self.assertEqual(a.tombstone_count, 0)
        self.assertEqual(b.tombstone_count, 0)

    def test_new_edits_converge_after_compaction(self):
        for seed in range(20):
            rng = random.Random(5000 + seed)
            a, b, c = Doc("a"), Doc("b"), Doc("c")
            base = a.insert(0, "the base text here")
            b.apply(base)
            c.apply(base)
            d = a.delete(0, 4)
            for r in (b, c):
                r.apply(d)
            mvv = self._min_vv([a, b, c])
            for r in (a, b, c):
                r.compact(mvv)
            self.assertEqual(a.tombstone_count, 0)
            # new concurrent edits post-compaction
            reps = [a, b, c]
            new_ops = []
            for r in reps:
                for batch in do_random_local(r, rng, rng.randint(3, 10)):
                    new_ops.extend(batch)
            for target in reps:
                foreign = [op for op in new_ops
                           if op.replica != target.replica_id]
                deliver_scrambled(target, foreign, rng)
            self.assertEqual(len({r.text() for r in reps}), 1,
                             "post-compaction seed %d diverged" % seed)

    def test_compact_uncovered_delete_is_kept(self):
        a, b = Doc("a"), Doc("b")
        base = a.insert(0, "abc")
        b.apply(base)
        a.delete(0, 1)               # b has NOT seen this delete
        mvv = self._min_vv([a, b])   # min excludes the unsynced delete
        removed = a.compact(mvv)
        self.assertEqual(removed, 0)
        self.assertGreater(a.tombstone_count, 0)


# --------------------------------------------------------------------------- #
# 9. serialization + state merge                                              #
# --------------------------------------------------------------------------- #

class Serialization(unittest.TestCase):
    def test_state_roundtrip(self):
        d = Doc("r")
        d.insert(0, "round trip me")
        d.delete(0, 6)
        s = d.state_to_json()
        d2 = Doc.state_from_json(s)
        self.assertEqual(d2.text(), d.text())
        self.assertEqual(d2.version_vector(), d.version_vector())
        self.assertEqual(d2.state_to_json(), s)

    def test_op_json_roundtrip(self):
        d = Doc("r")
        ops = d.insert(0, "xy")
        for op in ops:
            self.assertEqual(op_from_json(op_to_json(op)), op)
        blob = ops_to_json(ops)
        self.assertEqual(ops_from_json(blob), ops)

    def test_merge_converges_both_directions(self):
        a = Doc("a")
        b = Doc("b")
        base = a.insert(0, "common")
        b.apply(base)
        a.insert(6, "-A-end")
        b.insert(0, "start-B-")
        sa = a.state_to_json()
        sb = b.state_to_json()
        a.merge(sb)
        b.merge(sa)
        self.assertEqual(a.text(), b.text())
        self.assertEqual(a.pending_count, 0)
        self.assertEqual(b.pending_count, 0)

    def test_merge_property_random(self):
        for seed in range(20):
            rng = random.Random(9000 + seed)
            a, b = Doc("a"), Doc("b")
            base = a.insert(0, "".join(rng.choice(ALPHABET) for _ in range(6)))
            b.apply(base)
            do_random_local(a, rng, rng.randint(4, 12))
            do_random_local(b, rng, rng.randint(4, 12))
            sa, sb = a.state_to_json(), b.state_to_json()
            a.merge(sb)
            b.merge(sa)
            self.assertEqual(a.text(), b.text(),
                             "merge seed %d diverged" % seed)


# --------------------------------------------------------------------------- #
# 10. unicode                                                                  #
# --------------------------------------------------------------------------- #

class Unicode(unittest.TestCase):
    def test_emoji_and_cjk_converge(self):
        a, b = Doc("a"), Doc("b")
        base = a.insert(0, "café☕中文")
        b.apply(base)
        oa = a.insert(len(a), "🐉")
        ob = b.insert(0, "漢")
        a.apply(ob)
        b.apply(oa)
        self.assertEqual(a.text(), b.text())
        self.assertIn("🐉", a.text())
        self.assertIn("漢", a.text())

    def test_codepoint_granularity(self):
        d = Doc("r")
        d.insert(0, "a🐉b")
        self.assertEqual(len(d), 3)   # one code point per element
        d.delete(1, 1)
        self.assertEqual(d.text(), "ab")


# --------------------------------------------------------------------------- #
# 11. op structure sanity                                                      #
# --------------------------------------------------------------------------- #

class OpStructure(unittest.TestCase):
    def test_op_ids_unique_and_origin_chaining(self):
        d = Doc("r")
        ops = d.insert(0, "abc")
        ids = [op.id for op in ops]
        self.assertEqual(len(set(ids)), 3)
        self.assertEqual(ops[0].origin, BEGIN)
        self.assertEqual(ops[1].origin, ops[0].id)
        self.assertEqual(ops[2].origin, ops[1].id)

    def test_reject_bad_apply_argument(self):
        d = Doc("r")
        with self.assertRaises(TypeError):
            d.apply("not an op")


if __name__ == "__main__":
    unittest.main(verbosity=2)
