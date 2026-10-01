# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-23  Created.
#   2026-10-01  add_noise decomposes operations on more than two wires into
#               one- and two-wire gates first (probes.gates.elementary), so a
#               StatePrep (A5 amplitude encoding) is charged as its Mottonen
#               gates. Circuits of one- and two-wire gates are unchanged.

"""Two-rate gate-noise models: depolarizing, bit flip and phase flip.

The model
---------
After every gate, a single-qubit channel is applied to every wire the gate
touched. The channel strength depends on the gate's arity only: ``p1`` after a
single-qubit gate, ``p2`` after a two-qubit gate (on each of its two wires).
This is the standard gate-level Pauli-noise abstraction of NISQ simulators.

Why two rates
-------------
G5 used one uniform per-gate rate, and the v10.1 review flagged the result: with
one rate, a circuit with many cheap rotations is charged like one with many
expensive entanglers. On hardware two-qubit gates are roughly ten times worse
than single-qubit gates, so the levels below keep ``p2 = 10 p1``.

Channels (PennyLane definitions)
--------------------------------
* ``depolarizing`` - ``DepolarizingChannel(p)``: X, Y, Z each with ``p/3``.
  Shrinks the whole Bloch vector by ``1 - 4p/3``.
* ``bitflip`` - ``BitFlip(p)``: X with probability ``p``. Shrinks the y and z
  Bloch components by ``1 - 2p``: it attacks populations, hence ``<Z>`` readout.
* ``phaseflip`` - ``PhaseFlip(p)``: Z with probability ``p``. Shrinks x and y by
  ``1 - 2p``: it attacks coherences and leaves ``<Z>`` of a diagonal state alone.

Levels
------
* ``small`` - ``p1 = 1e-4``, ``p2 = 1e-3``: best reported devices today
  (two-qubit error near 0.1%), the regime a near-term error-mitigated run
  targets.
* ``nisq`` - ``p1 = 1e-3``, ``p2 = 1e-2``: the textbook NISQ operating point
  (Preskill 2018) and the order of median two-qubit errors on widely available
  superconducting devices; it sits inside the band where G5 measured the
  critical per-gate noise (0.006 - 0.02).
* ``stress`` - ``p1 = 5e-3``, ``p2 = 5e-2``: five times NISQ, beyond the G5
  critical band. Not a hardware claim: a sensitivity probe that separates
  families whose decisions all survive the two realistic levels.

A two-qubit depolarizing error is modelled as a product of single-qubit
depolarizing channels on its two wires. That is an approximation (it omits
correlated two-qubit Paulis) and is declared here so any consumer inherits the
declaration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pennylane as qml

from ..probes.gates import elementary

CHANNELS: tuple[str, ...] = ("depolarizing", "bitflip", "phaseflip")
"""Supported channels, in reporting order."""

LEVELS: dict[str, tuple[float, float]] = {
    "small": (1e-4, 1e-3),
    "nisq": (1e-3, 1e-2),
    "stress": (5e-3, 5e-2),
}
"""Named ``(p1, p2)`` levels. See the module notes for the sources.

``small`` and ``nisq`` are the two reported levels; ``stress`` is an optional
sensitivity probe."""

REPORTED_LEVELS: tuple[str, ...] = ("small", "nisq")
"""The levels the task-noise section reports by default."""

_OPS = {"depolarizing": qml.DepolarizingChannel, "bitflip": qml.BitFlip,
        "phaseflip": qml.PhaseFlip}


@dataclass(frozen=True)
class NoiseModel:
    """A gate-level noise model: one channel, one rate per gate arity.

    Attributes:
        channel: One of :data:`CHANNELS`.
        p1: Channel probability after every single-qubit gate.
        p2: Channel probability, on each wire, after every two-qubit gate.
        level: Name of the preset it came from, or ``"custom"``.
    """

    channel: str
    p1: float
    p2: float
    level: str = "custom"

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {self.channel!r}")
        if not (0.0 <= self.p1 < 1.0 and 0.0 <= self.p2 < 1.0):
            raise ValueError("channel probabilities must lie in [0, 1)")

    @property
    def tag(self) -> str:
        """Short label, for example ``"depolarizing@nisq"``."""
        return f"{self.channel}@{self.level}"

    def ops_after(self, op) -> list:
        """Return the channel operations that follow one gate."""
        p = self.p1 if len(op.wires) == 1 else self.p2
        if p == 0.0:
            return []
        return [_OPS[self.channel](p, wires=w) for w in op.wires]

    def record(self) -> dict:
        """Return a JSON-ready description for result records."""
        return asdict(self)

    def __str__(self) -> str:
        return json.dumps(self.record(), sort_keys=True)


def preset(channel: str, level: str) -> NoiseModel:
    """Return the preset model for one channel at one named level.

    Args:
        channel: One of :data:`CHANNELS`.
        level: One of the keys of :data:`LEVELS`.

    Returns:
        The :class:`NoiseModel`.

    Raises:
        KeyError: If ``level`` is not a known level.
    """
    p1, p2 = LEVELS[level]
    return NoiseModel(channel=channel, p1=p1, p2=p2, level=level)


@qml.transform
def add_noise(tape, model: NoiseModel):
    """Insert ``model``'s channel after every gate of a circuit.

    A PennyLane transform, so it composes with any QNode on a mixed-state device
    (``default.mixed``), keeps parameter broadcasting and supports backprop.

    Args:
        tape: The circuit.
        model: The noise model.

    Returns:
        The transformed tape and a post-processing function, per the transform
        protocol.
    """
    ops = []
    for op in elementary(tape.operations):
        ops.append(op)
        ops += model.ops_after(op)
    return [tape.copy(operations=ops)], lambda results: results[0]
