"""Tests for the UNG node adapters in crdtext/ung_nodes.py."""

import copy
import inspect
import json
import math
import re
from pathlib import Path

import pytest

from crdtext import ung_nodes
from crdtext.ung_nodes import NODES

FIXTURE_DIR = Path(ung_nodes.__file__).resolve().parent / "ung_fixtures"
PREFIX = "amarel.crdtext."
ID_TAIL = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
REQUIRED_TAGS = {"license.mit", "runtime.python", "dependency-free"}
NODE_IDS = [n["id"] for n in NODES]


def _approx_eq(a, b, rel=1e-9, abs_=1e-12):
    """Recursive equality with float tolerance (bools stay exact)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return math.isclose(fa, fb, rel_tol=rel, abs_tol=abs_)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_approx_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_approx_eq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _load_cases(node_id):
    path = FIXTURE_DIR / (node_id + ".json")
    assert path.is_file(), "missing fixture file for %s" % node_id
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(cases, list) and len(cases) >= 2, (
        "%s: fixtures must be a list of >= 2 cases" % node_id
    )
    for case in cases:
        assert set(case) == {"inputs", "parameters", "expect"}, node_id
    return cases


def _call(node, case):
    kwargs = {**copy.deepcopy(case["inputs"]), **copy.deepcopy(case["parameters"])}
    return node["fn"](**kwargs)


def _node(node_id):
    return next(n for n in NODES if n["id"] == node_id)


def test_module_importable_and_nonempty():
    assert isinstance(NODES, list) and NODES
    assert len(set(NODE_IDS)) == len(NODE_IDS), "duplicate node ids"


def test_metadata_sanity():
    for node in NODES:
        nid = node["id"]
        assert nid.startswith(PREFIX), nid
        assert ID_TAIL.match(nid[len(PREFIX):]), nid
        assert callable(node["fn"])
        assert isinstance(node["summary"], str) and node["summary"].strip()
        assert isinstance(node["capabilities"], list) and node["capabilities"]
        for cap in node["capabilities"]:
            assert re.match(r"^[a-z0-9-]+\.[a-z0-9-]+$", cap), (nid, cap)
        for port in node["inputs"] + node["outputs"]:
            assert set(port) >= {"name", "type_id", "description"}, (nid, port)
            assert port["type_id"].startswith("amarel.types."), (nid, port)
            assert isinstance(port["description"], str) and port["description"]
        for param in node["parameters"]:
            assert set(param) >= {"name", "value_type", "default", "required"}, (
                nid,
                param,
            )
        assert node["effects"] == []
        assert node["determinism"] == "deterministic"
        assert node["idempotency"] == "idempotent"
        assert REQUIRED_TAGS <= set(node["tags"]), nid


def test_declared_names_match_signature():
    for node in NODES:
        sig = set(inspect.signature(node["fn"]).parameters)
        input_names = {p["name"] for p in node["inputs"]}
        param_names = {p["name"] for p in node["parameters"]}
        assert input_names <= sig, (node["id"], input_names - sig)
        assert param_names <= sig, (node["id"], param_names - sig)
        assert not input_names & param_names, node["id"]
        output_names = [p["name"] for p in node["outputs"]]
        assert len(set(output_names)) == len(output_names), node["id"]


def test_no_orphan_fixture_files():
    on_disk = {p.stem for p in FIXTURE_DIR.glob("*.json")}
    assert on_disk == set(NODE_IDS), on_disk ^ set(NODE_IDS)


@pytest.mark.parametrize("node_id", NODE_IDS)
def test_fixture_cases(node_id):
    node = _node(node_id)
    for i, case in enumerate(_load_cases(node_id)):
        got = _call(node, case)
        assert _approx_eq(got, case["expect"]), (
            "%s case %d:\n got: %r\nwant: %r" % (node_id, i, got, case["expect"])
        )
        output_names = {p["name"] for p in node["outputs"]}
        assert isinstance(got, dict) and set(got) == output_names, node_id


@pytest.mark.parametrize("node_id", NODE_IDS)
def test_double_run_deterministic(node_id):
    node = _node(node_id)
    for case in _load_cases(node_id):
        first = _call(node, case)
        second = _call(node, case)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


@pytest.mark.parametrize("node_id", NODE_IDS)
def test_inputs_not_mutated(node_id):
    node = _node(node_id)
    for case in _load_cases(node_id):
        kwargs = {**case["inputs"], **case["parameters"]}
        snapshot = copy.deepcopy(kwargs)
        node["fn"](**kwargs)
        assert kwargs == snapshot, "%s mutated its arguments" % node_id


@pytest.mark.parametrize("node_id", NODE_IDS)
def test_json_round_trip(node_id):
    node = _node(node_id)
    for case in _load_cases(node_id):
        got = _call(node, case)
        wire = json.dumps(got, allow_nan=False)
        assert json.loads(wire) == got, "%s output does not round-trip" % node_id
