# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-09-09  Created. QNodes and forward maps for the task axis.
#
#   Encoding is controlled HERE, not by the ansatz's own encoder, because the
#   descriptor package only ever fed a single fixed input and its RY-loop
#   encoders do not batch over a dataset. The circuits used are otherwise the
#   package circuits:
#     - A1, A2, A4 (feature route): AngleEmbedding(x) once, then the family's
#       variational block via ``Ansatz.apply(theta, n, d)`` (x=None).
#     - A3 (feature route): re-upload x before every layer, mirroring
#       ``ansatze/a3_reupload.apply``, so the re-uploading character is kept.
#     - A4 (compositional route): trainable word-box angles selected per
#       sentence from a word table, plus the shared IQP CRZ ring and unitary
#       cups of ``ansatze/a4_discocat``.
#   2026-09-22  A5 (Mari 2020 dressed head): build_dressed_model and the
#               TaskModel record.
#   2026-09-23  Fair protocol: build_fair_model (shared dressed interface for
#               every family) and the classical control C0.
#   2026-10-01  pure-v1. The dressed and fair builders, the classical control
#               and the Mari A5 are removed (archived with fair-v1 in
#               old/fair_v1_archive): they put ~1550 trainable classical
#               parameters around a 12-40 angle circuit. Added the pure model,
#               build_pure_model: fixed features in, the family's own encoding
#               and circuit, the local readout <Z_0>, and the two-parameter
#               head sigmoid(a <Z_0> + b), the only classical parameters. The
#               model registry MODELS covers A1-A7 and the two encoding-arm
#               variants (the A2 ansatz under amplitude and IQP encodings).
#               TaskModel, the legacy feature route (A1-A4) and the
#               compositional A4 route are unchanged.
#   2026-10-08  Documentation only. The MODELS note claimed A3's Euler block
#               and ring are A2's; the block is, the ring is not (A3: range 1,
#               between layers only; A2: growing range, after every layer).
#               Corrected, with the consequence for the encoding arm.

"""Quantum models for the task axis.

Two families of builders live here.

**pure-v1** (:func:`build_pure_model`, the protocol of :mod:`.pure`). The
circuit is the only trainable component. A fixed, unsupervised feature map
(standardised PCA of frozen encoder features) feeds the family's own encoding;
the circuit's local readout ``<Z_0>`` goes through the two-parameter head
``p = sigmoid(a <Z_0> + b)``. Nothing else is trained. :data:`MODELS` names
every model of the protocol and what it reads.

**Legacy v10 task axis** (A1-A4 only). The triple
``(forward, n_circuit_params, two_qubit_gates)``:

* ``forward(params, features)`` -> probabilities in ``(0, 1)``, batched. The
  last two entries of ``params`` are the classical head ``(a, b)``:
  ``p = sigmoid(a * <Z_0> + b)``.
* ``n_circuit_params`` - the number of circuit angles (P5, excluding the head).
* ``two_qubit_gates`` - the count of two-qubit gates in one circuit (P6).

Both use the local observable ``<Z_0>`` (G4: the global cost flattens faster).
For A7 that observable is the paper's ``Y`` readout, through the fixed basis
change at the end of its circuit (see :mod:`~vqc_scorecard.ansatze.a7_farhi_neven`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from numpy.typing import NDArray

from ..ansatze import REGISTRY
from ..ansatze import a5_circuit_centric, a6_iqp
from ..noise import NoiseModel, add_noise
from ..probes.gates import elementary

Model = tuple[Callable, int, int]

LEGACY_FAMILIES: tuple[str, ...] = ("A1", "A2", "A3", "A4")
"""Families of the legacy v10 feature route (angle embedding then the block)."""


@dataclass(frozen=True)
class TaskModel:
    """A trainable classifier for the task axis.

    Attributes:
        forward: ``(params, features) -> P(y = 1)``, batched over the rows of
            ``features``.
        p_circuit: Circuit angles, the P5 convention.
        p_classical: Classical parameters (heads, pre- and post-layers).
        two_qubit_gates: Two-qubit gates in one circuit, the P6 convention.
        init: ``rng -> flat initial parameter vector``, or ``None`` for the
            training loop's rule (uniform circuit angles, head ``(1, 0)``).
        init_rule: One line describing ``init``, written into every record.
        layout: Name -> ``(start, stop)`` of each block in the flat vector.
    """

    forward: Callable
    p_circuit: int
    p_classical: int
    two_qubit_gates: int
    init: Callable[[np.random.Generator], NDArray] | None = None
    init_rule: str = "circuit uniform on [0, 2pi); head (a, b) = (1, 0)"
    layout: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def p_total(self) -> int:
        """Length of the flat parameter vector."""
        return self.p_circuit + self.p_classical


def as_task_model(built: TaskModel | Model) -> TaskModel:
    """Return ``built`` as a :class:`TaskModel`, lifting a legacy triple.

    A legacy triple has a two-parameter sigmoid head after its circuit angles, so
    its classical count is 2 and its initialisation is the training loop's rule.

    Args:
        built: A :class:`TaskModel`, or ``(forward, n_circuit_params, two_qubit_gates)``.

    Returns:
        The equivalent :class:`TaskModel`.
    """
    if isinstance(built, TaskModel):
        return built
    forward, p, two_q = built
    return TaskModel(forward=forward, p_circuit=int(p), p_classical=2,
                     two_qubit_gates=int(two_q),
                     layout={"circuit": (0, int(p)), "head": (int(p), int(p) + 2)})


def _sigmoid(z):
    return 1.0 / (1.0 + pnp.exp(-pnp.clip(z, -30, 30)))


def _two_qubit_gates_feature(tag: str, n: int, d: int) -> int:
    """Count two-qubit gates in one feature-route circuit via the tape."""
    a = REGISTRY[tag]
    theta = np.zeros(a.n_params(n, d))
    x = np.zeros(n)
    with qml.tape.QuantumTape() as tape:
        if tag == "A3":
            th = theta.reshape(d, n, 3)
            for layer in range(d):
                qml.AngleEmbedding(x, wires=range(n), rotation="Y")
                for q in range(n):
                    qml.RZ(th[layer, q, 0], wires=q)
                    qml.RY(th[layer, q, 1], wires=q)
                    qml.RZ(th[layer, q, 2], wires=q)
                if layer < d - 1:
                    for q in range(n):
                        qml.CNOT(wires=[q, (q + 1) % n])
        else:
            qml.AngleEmbedding(x, wires=range(n), rotation="Y")
            a.apply(theta, n, d)
    return sum(1 for op in tape.operations if len(op.wires) == 2)


def build_feature_model(tag: str, n: int, d: int, *, shots: int | None = None) -> Model:
    """Build a feature-route model for one family.

    Args:
        tag: Family tag, one of :data:`LEGACY_FAMILIES`.
        n: Qubit count.
        d: Circuit depth.
        shots: ``None`` for analytic backprop training; an int for a
            finite-shot evaluation device (parameter-shift diff).

    Returns:
        The ``(forward, n_circuit_params, two_qubit_gates)`` model triple.
        ``forward(params, x)`` takes angle features ``x`` of shape
        ``(batch, n)``.
    """
    if tag not in LEGACY_FAMILIES:
        raise ValueError(f"{tag} is not a legacy family {LEGACY_FAMILIES}; use build_pure_model")
    a = REGISTRY[tag]
    p = a.n_params(n, d)
    dev = qml.device("default.qubit", wires=n, shots=shots)
    diff = "backprop" if shots is None else "parameter-shift"

    if tag == "A3":
        @qml.qnode(dev, interface="autograd", diff_method=diff)
        def circ(theta, x):
            th = theta.reshape(d, n, 3)
            for layer in range(d):
                qml.AngleEmbedding(x, wires=range(n), rotation="Y")
                for q in range(n):
                    qml.RZ(th[layer, q, 0], wires=q)
                    qml.RY(th[layer, q, 1], wires=q)
                    qml.RZ(th[layer, q, 2], wires=q)
                if layer < d - 1:
                    for q in range(n):
                        qml.CNOT(wires=[q, (q + 1) % n])
            return qml.expval(qml.PauliZ(0))
    else:
        @qml.qnode(dev, interface="autograd", diff_method=diff)
        def circ(theta, x):
            qml.AngleEmbedding(x, wires=range(n), rotation="Y")
            a.apply(theta, n, d)
            return qml.expval(qml.PauliZ(0))

    def forward(params, x):
        theta, head_a, head_b = params[:p], params[p], params[p + 1]
        return _sigmoid(head_a * circ(theta, x) + head_b)

    return forward, p, _two_qubit_gates_feature(tag, n, d)


def build_compositional_model(
    n: int, d: int, vocab_size: int, *, shots: int | None = None
) -> Model:
    """Build the compositional A4 model (parse-driven trainable word boxes).

    The circuit angles are a word table of length ``vocab_size`` (one RY angle
    per word, selected per sentence) plus ``n * d`` shared IQP-ring CRZ angles.

    Args:
        n: Qubit count (= words per sentence).
        d: Depth (IQP word-box repeats).
        vocab_size: Word-table length (includes the pad id at index 0).
        shots: ``None`` for analytic backprop; an int for finite-shot eval.

    Returns:
        The ``(forward, n_circuit_params, two_qubit_gates)`` model triple.
        ``forward(params, onehot)`` takes a one-hot tensor
        ``(batch, n, vocab_size)``.
    """
    p2 = n * d
    p = vocab_size + p2
    dev = qml.device("default.qubit", wires=n, shots=shots)
    diff = "backprop" if shots is None else "parameter-shift"

    @qml.qnode(dev, interface="autograd", diff_method=diff)
    def circ(w, onehot):
        table = w[:vocab_size]
        crz = w[vocab_size:vocab_size + p2]
        word_ang = qml.math.tensordot(onehot, table, axes=([2], [0]))  # (batch, n)
        for q in range(n):
            qml.RY(word_ang[..., q], wires=q)
        i = 0
        for _ in range(d):
            for q in range(n):
                qml.Hadamard(wires=q)
            for q in range(n):
                qml.CRZ(crz[i], wires=[q, (q + 1) % n])
                i += 1
        for q in range(0, n - 1, 2):
            qml.CNOT(wires=[q, q + 1])
            qml.Hadamard(wires=q)
        return qml.expval(qml.PauliZ(0))

    # two-qubit gate count: CRZ ring (n per layer) + cups (n//2)
    two_q = n * d + (n // 2)

    def forward(params, onehot):
        w, head_a, head_b = params[:p], params[p], params[p + 1]
        return _sigmoid(head_a * circ(w, onehot) + head_b)

    return forward, p, two_q


# --------------------------------------------------------------------------- #
# pure-v1: the circuit is the only trainable component                         #
# --------------------------------------------------------------------------- #

ENCODINGS: tuple[str, ...] = ("angle", "reupload", "amplitude", "iqp", "readout-angle")
"""Data encodings of the pure protocol.

* ``angle`` - one ``RY(x_q)`` per qubit, once (A1, A2, A4).
* ``reupload`` - the same ``RY(x_q)`` before every layer (A3).
* ``amplitude`` - ``2**n`` features as the amplitudes of the state (A5).
* ``iqp`` - the second-order ZZ feature map, applied twice (A6).
* ``readout-angle`` - readout in ``|1>``, ``RY(x_j)`` on the ``n - 1`` data
  wires (A7).
"""


@dataclass(frozen=True)
class ModelSpec:
    """One model of the pure protocol: an ansatz and the encoding that feeds it.

    Attributes:
        tag: Model tag written into every record (``"A5"``, ``"A2:amp"``).
        ansatz: Registry tag of the variational circuit.
        encoding: One of :data:`ENCODINGS`.
        arm: ``"family"`` (each family as published) or ``"encoding"`` (the A2
            ansatz under a different encoding, the Class II arm).
    """

    tag: str
    ansatz: str
    encoding: str
    arm: str = "family"

    def n_features(self, n: int) -> int:
        """Return the feature length the encoding reads at width ``n``."""
        if self.encoding == "amplitude":
            return 2 ** n
        if self.encoding == "readout-angle":
            return n - 1
        return n

    @property
    def feature_view(self) -> str:
        """Return the feature view the encoding reads: ``"angles"`` or ``"pca_std"``.

        Rotation encodings read angles on ``[0, pi]``; the amplitude encoding
        reads standardised PCA scores, which it normalises to a unit vector.
        """
        return "pca_std" if self.encoding == "amplitude" else "angles"


UNBROADCAST_ON_MIXED: tuple[str, ...] = ("amplitude", "iqp")
"""Encodings whose minibatch is split into single circuits on the noisy device."""

OWN_ENCODING: dict[str, str] = {"A5": "amplitude", "A6": "iqp", "A7": "readout-angle"}
"""Families whose ``apply`` loads the data itself, with the encoding it uses."""

MODELS: dict[str, ModelSpec] = {s.tag: s for s in (
    ModelSpec("A1", "A1", "angle"),
    ModelSpec("A2", "A2", "angle"),
    ModelSpec("A3", "A3", "reupload"),
    ModelSpec("A4", "A4", "angle"),
    ModelSpec("A5", "A5", "amplitude"),
    ModelSpec("A6", "A6", "iqp"),
    ModelSpec("A7", "A7", "readout-angle"),
    ModelSpec("A2:amp", "A2", "amplitude", arm="encoding"),
    ModelSpec("A2:iqp", "A2", "iqp", arm="encoding"),
)}
"""Every model of the pure protocol, by tag.

The encoding arm (Class II) holds the SEL ansatz of A2 fixed and swaps the
encoding: ``A2`` (angle), ``A3`` (re-upload), ``A2:amp`` and ``A2:iqp``. The
A3 and A2 records serve both arms.

``A2:amp`` and ``A2:iqp`` run the A2 circuit exactly. ``A3`` does not: it shares
A2's Euler block (``RZ RY RZ``) but entangles with a range-1 CNOT ring placed
only between layers, where A2 rings after every layer with a range that grows
with the layer (see :mod:`~vqc_scorecard.ansatze.a2_sel` and
:mod:`~vqc_scorecard.ansatze.a3_reupload`). The angle against re-upload
contrast therefore changes the encoding and the entangler together."""


def _reupload_body(theta, x, n: int, d: int) -> None:
    th = qml.math.reshape(theta, (d, n, 3))
    for layer in range(d):
        for q in range(n):
            qml.RY(x[..., q], wires=q)
        for q in range(n):
            qml.RZ(th[layer, q, 0], wires=q)
            qml.RY(th[layer, q, 1], wires=q)
            qml.RZ(th[layer, q, 2], wires=q)
        if layer < d - 1:
            for q in range(n):
                qml.CNOT(wires=[q, (q + 1) % n])


def circuit_body(tag: str, n: int, d: int) -> Callable:
    """Return ``body(theta, x)``: one model's encoding and variational circuit.

    ``x`` may be a batch ``(batch, n_features)``; every encoding indexes the
    last axis, so a whole minibatch runs in one circuit call.

    Args:
        tag: A key of :data:`MODELS`.
        n: Qubit count.
        d: Depth, in the unit of the ansatz.

    Returns:
        A function that queues the gates on the active tape.

    Raises:
        KeyError: If ``tag`` is not a pure-protocol model.
    """
    spec = MODELS[tag]
    a = REGISTRY[spec.ansatz]
    if spec.encoding == "reupload":
        return lambda theta, x: _reupload_body(theta, x, n, d)

    if OWN_ENCODING.get(spec.ansatz) == spec.encoding:
        # The family loads its own data (A5 amplitude, A6 IQP, A7 readout and
        # data wires): use its apply with x, one definition for the
        # descriptors and the task alike. Every encoder indexes x[..., k], so
        # a minibatch broadcasts.
        return lambda theta, x: a.apply(theta, n, d, x)

    def encode(x):
        if spec.encoding == "angle":
            for q in range(n):
                qml.RY(x[..., q], wires=q)
        elif spec.encoding == "amplitude":
            a5_circuit_centric.embed(x, n)
        elif spec.encoding == "iqp":
            a6_iqp.embed(x, n)
        else:
            raise ValueError(f"encoding {spec.encoding!r} cannot feed {spec.ansatz}")

    def body(theta, x):
        encode(x)
        a.apply(theta, n, d)

    return body


def body_two_qubit_count(tag: str, n: int, d: int) -> int:
    """Two-qubit gates of one model's body, encoding included (P6 convention).

    Operations on more than two wires (the amplitude encoding) are decomposed
    first, so the count is the hardware count.
    """
    spec = MODELS[tag]
    theta = np.zeros(REGISTRY[spec.ansatz].n_params(n, d))
    x = np.linspace(0.1, 1.0, spec.n_features(n))
    script = qml.tape.make_qscript(lambda: circuit_body(tag, n, d)(theta, x))()
    return sum(1 for op in elementary(script.operations) if len(op.wires) == 2)


def build_pure_model(
    tag: str,
    n: int,
    d: int,
    *,
    noise: NoiseModel | None = None,
    shots: int | None = None,
) -> TaskModel:
    """Build one pure-protocol model: fixed features, the circuit, a two-parameter head.

    ``p(y = 1 | x) = sigmoid(a <Z_0>_theta(x) + b)``. The circuit angles
    ``theta`` and the head ``(a, b)`` are the whole parameter vector, in that
    order; ``(a, b)`` are the only classical parameters (the readout scale and
    bias the paper of A5 also trains).

    The initialisation is the training loop's rule (``init=None``): circuit
    angles uniform on ``[0, 2 pi)``, head ``(1, 0)``. That is the distribution
    under which G1 measures gradient variance, so the trainability descriptor
    and the training run start from the same ensemble.

    Args:
        tag: A key of :data:`MODELS`.
        n: Qubit count.
        d: Depth, in the unit of the ansatz.
        noise: Optional gate-noise model; the circuit then runs on
            ``default.mixed`` with the model's channel after every gate
            (operations on more than two wires are decomposed first).
        shots: ``None`` for analytic evaluation (backprop training); an int for
            a finite-shot evaluation device (parameter-shift).

    Returns:
        The :class:`TaskModel`. ``forward(params, x)`` takes ``x`` of shape
        ``(batch, MODELS[tag].n_features(n))``.

    Raises:
        KeyError: If ``tag`` is not a pure-protocol model.
        ValueError: If the ansatz does not permit width ``n`` (A4 needs even n).
    """
    spec = MODELS[tag]
    if spec.ansatz == "A4" and n % 2:
        raise ValueError("A4 needs an even qubit count")
    p_circ = REGISTRY[spec.ansatz].n_params(n, d)
    body = circuit_body(tag, n, d)
    device = "default.mixed" if noise is not None else "default.qubit"
    dev = qml.device(device, wires=n, shots=shots)
    diff = "backprop" if shots is None else "parameter-shift"

    def qfunc(theta, x):
        body(theta, x)
        return qml.expval(qml.PauliZ(0))

    circ = qml.QNode(qfunc, dev, interface="autograd", diff_method=diff)
    if noise is not None:
        if spec.encoding in UNBROADCAST_ON_MIXED:
            # Split the batch into single circuits first: the Mottonen
            # decomposition of a state preparation does not broadcast, and
            # default.mixed (PennyLane 0.42) mis-shapes batched diagonal gates
            # (the RZ / IsingZZ phases of the IQP map).
            circ = qml.transforms.broadcast_expand(circ)
        circ = add_noise(circ, noise)

    def forward(params, x):
        x = np.asarray(x, dtype=float)
        theta, head_a, head_b = params[:p_circ], params[p_circ], params[p_circ + 1]
        return _sigmoid(head_a * circ(theta, x) + head_b)

    return TaskModel(
        forward=forward,
        p_circuit=p_circ,
        p_classical=2,
        two_qubit_gates=body_two_qubit_count(tag, n, d),
        init=None,
        init_rule="circuit uniform on [0, 2pi); head (a, b) = (1, 0)",
        layout={"circuit": (0, p_circ), "head": (p_circ, p_circ + 2)},
    )
