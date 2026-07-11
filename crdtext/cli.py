"""crdtext command-line interface.

``crdtext demo`` runs a scripted three-replica editing session with concurrent
edits and deliberately out-of-order + duplicated delivery, prints each
replica's text, and self-checks that all replicas converged (exit 1 if not).

The delivery scrambler here is *deterministic* — no randomness — so the demo
is reproducible. The randomized stress lives in the test suite.
"""

import argparse
import sys

from .core import Doc


def _scramble(ops):
    """A fixed out-of-order + duplicated delivery order (no randomness).

    Reverses the batch (so children arrive before their origins — exercising
    causal buffering) and re-appends every other op as a duplicate (exercising
    idempotence). ``apply`` buffers and de-duplicates, so the end state is
    unaffected.
    """
    ops = list(ops)
    return list(reversed(ops)) + ops[::2]


def _deliver(targets, ops):
    for target in targets:
        for op in _scramble(ops):
            target.apply(op)


def demo(out=sys.stdout):
    """Scripted 3-replica convergence demo. Returns 0 if converged, else 1."""
    a, b, c = Doc("A"), Doc("B"), Doc("C")

    out.write("1. A writes a base line; B and C sync (scrambled delivery).\n")
    base = a.insert(0, "the quick brown fox")
    _deliver([b, c], base)

    out.write("2. Three concurrent edits on the shared base:\n")
    out.write("   A deletes 'quick '  |  B appends ' jumps'  |  C prepends 'Wow: '\n")
    e_a = a.delete(4, 6)                       # remove "quick "
    e_b = b.insert(len(b), " jumps")           # append at B's end
    e_c = c.insert(0, "Wow: ")                 # prepend at C

    out.write("3. Cross-deliver every batch (out of order, with duplicates).\n")
    _deliver([b, c], e_a)
    _deliver([a, c], e_b)
    _deliver([a, b], e_c)

    out.write("\nReplica texts after full sync:\n")
    texts = {}
    for d in (a, b, c):
        texts[d.replica_id] = d.text()
        out.write("  %s: %r  (pending=%d)\n" % (
            d.replica_id, d.text(), d.pending_count))

    converged = (
        len({t for t in texts.values()}) == 1
        and all(d.pending_count == 0 for d in (a, b, c))
        and a.version_vector() == b.version_vector() == c.version_vector()
    )
    if converged:
        out.write("\nCONVERGED: %r\n" % a.text())
        return 0
    out.write("\nDIVERGED\n")
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="crdtext",
        description="A convergent replicated text type (RGA CRDT).")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="run a scripted 3-replica convergence demo")
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        return demo()
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
