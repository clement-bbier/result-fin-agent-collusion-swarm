import json

import numpy as np

from finswarm.core.rng import STREAMS, RNGRegistry


def test_streams_deterministic_and_seed_sensitive() -> None:
    a, b = RNGRegistry(123), RNGRegistry(123)
    for s in STREAMS:
        assert a[s].random() == b[s].random()
    assert RNGRegistry(123)["swarm"].random() != RNGRegistry(124)["swarm"].random()


def test_streams_are_independent() -> None:
    r = RNGRegistry(5)
    x = r["swarm"].random(4)
    y = r["matching"].random(4)
    assert not np.allclose(x, y)


def test_capture_restore_roundtrip_is_json_serialisable() -> None:
    r = RNGRegistry(7)
    r["swarm"].random(5)
    state = r.capture()
    json.dumps(state)
    x = r["swarm"].random(3)
    r.restore(state)
    y = r["swarm"].random(3)
    assert np.array_equal(x, y)


def test_respawn_is_deterministic_and_salt_sensitive() -> None:
    r1, r2, r3 = RNGRegistry(1), RNGRegistry(1), RNGRegistry(1)
    r1.respawn("interventions", 42)
    r2.respawn("interventions", 42)
    r3.respawn("interventions", 43)
    v1, v2, v3 = (r["interventions"].random() for r in (r1, r2, r3))
    assert v1 == v2
    assert v1 != v3
