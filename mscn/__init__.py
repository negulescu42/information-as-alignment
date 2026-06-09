"""MSCN -- a Multi-Scale Coherence Network toy model.

A small, self-contained, runnable implementation of the three-layer IBF
apparatus described in the pilot design:

* Layer 1 -- individual coherence-gradient learning   (:mod:`mscn.learner`)
* Layer 2 -- network cooperation and games            (:mod:`mscn.network`, :mod:`mscn.games`)
* Layer 3 -- meta-learning, self-knowledge, phase     (:mod:`mscn.hierarchy`,
             control                                    :mod:`mscn.selfmodel`, :mod:`mscn.phase`)

The integrated system lives in :mod:`mscn.mscn`. Run ``python -m mscn.demo`` for
the full guided tour, or ``python -m mscn.tests`` to check the formal guarantees
empirically.

Knowledge is stored as coherence modifications ``delta_R`` on configuration
space (a kernel-based associative memory), never as neural weights. Every
mechanism is tied to a theorem in the IBF formalisation; see the module
docstrings and ``mscn/README.md``.
"""

__version__ = "0.1.0"

__all__ = [
    "landscapes",
    "learner",
    "baselines",
    "network",
    "games",
    "hierarchy",
    "selfmodel",
    "phase",
    "mscn",
]
