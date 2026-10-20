"""Independent random streams, one per (seed, purpose, index) coordinate.

Its own module rather than part of `common` so that a light CLI command does not import
numpy to read a config file.

The scheme this replaces derived every stream by *adding* a coordinate to the base seed.
Addition is not injective on pairs, so coordinates collided by construction: pool p at
round r and pool p + r at initialisation received the same integer and therefore the same
draws. That was verified directly -- pool 0 round 1 and pool 1 initialisation emitted
identical sequences, as did pool 1 round 2 and pool 3 initialisation -- which means the
twenty nominally independent evaluation pools shared a correlation band along the
diagonal, and the `random` baseline's draws were among the affected.

The same integer was additionally handed to the surrogate as `random_state` and used for
the acquisition tie-break. The tie-break exists to stop large tied plateaus in the
surrogate's predictions from resolving in index order, so drawing it from the stream that
also fixed the surrogate's bootstrap partly defeated its purpose.

SeedSequence derives coordinates by hashing rather than addition, so distinct coordinates
give distinct streams and neither collision can recur.
"""
import hashlib

import numpy as np


def _coordinate(label):
    """A stable integer for one coordinate.

    Python's `hash` is salted per process, so a string coordinate passed through it would
    select a different stream on every run and nothing would reproduce. blake2b does not
    move between processes or platforms.
    """
    if isinstance(label, (int, bool, np.integer)):
        return int(label)
    return int.from_bytes(hashlib.blake2b(str(label).encode(), digest_size=4).digest(), "big")


def stream(seed, *labels):
    """An independent generator for one coordinate under `seed`."""
    return np.random.default_rng(
        np.random.SeedSequence(int(seed), spawn_key=tuple(_coordinate(l) for l in labels)))


def stream_int(seed, *labels, bits=31):
    """A reproducible integer seed for a library that insists on one, such as scikit-learn."""
    return int(stream(seed, *labels).integers(1 << bits))
