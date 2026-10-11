"""Causal, bounded repair for OpenPose/Kinetics skeleton sequences.

The input layout is (3, time, joints, people): centered x, flipped y, and
confidence. Coordinates are already in the model's coordinate system. This
module does not normalize them or treat zero coordinates as missing.
Detector confidence is a nonnegative score, not a guaranteed probability:
some official OpenPose values exceed one and are preserved unchanged.

Only earlier observations of the *same* joint and person are used. A small
alpha-beta tracker estimates position and velocity; reliable observations are
passed through unchanged. During short gaps, the tracker extrapolates with
explicit speed/displacement limits and decaying confidence. Long gaps and
joints without a previous observation remain invalid. Person identity should
be consistent along the input's person axis. An optional OpenPose torso-jump
guard conservatively resets a slot when its visible torso suddenly changes.
It does not track identities, reorder people, or guarantee detection of swaps
when fewer than two anchors are visible or people overlap.

Defaults are initial engineering choices, not parameters fitted on test data.
Choose any tuning parameters on a separate calibration split and freeze them
before comparing held-out baseline and repaired predictions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = True
    min_confidence: float = 0.2
    max_gap_frames: int = 5
    alpha: float = 0.85
    beta: float = 0.2
    confidence_decay: float = 0.75
    max_speed: float = 0.08
    max_displacement: float = 0.2
    identity_jump_threshold: float | None = None
    min_repair_confidence: float = 0.0

    def __post_init__(self):
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean")
        if (
            isinstance(self.max_gap_frames, (bool, np.bool_))
            or not isinstance(self.max_gap_frames, (int, np.integer))
            or self.max_gap_frames < 0
        ):
            raise ValueError("max_gap_frames must be a nonnegative integer")
        for name in ("min_confidence", "alpha", "beta", "confidence_decay"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.alpha == 0:
            raise ValueError("alpha must be greater than zero")
        if self.confidence_decay == 0:
            raise ValueError("confidence_decay must be greater than zero")
        for name in ("max_speed", "max_displacement"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.identity_jump_threshold is not None and (
            not np.isfinite(self.identity_jump_threshold)
            or self.identity_jump_threshold <= 0
        ):
            raise ValueError(
                "identity_jump_threshold must be None or finite and positive"
            )
        if (
            not np.isfinite(self.min_repair_confidence)
            or self.min_repair_confidence < 0
        ):
            raise ValueError("min_repair_confidence must be finite and nonnegative")


@dataclass
class RecoveryResult:
    repaired: np.ndarray
    original_confidence: np.ndarray
    observed_mask: np.ndarray
    imputed_mask: np.ndarray
    # Number of person-slot resets in this result's frame or sequence.
    identity_resets: int = 0

    @property
    def unresolved_mask(self) -> np.ndarray:
        """Joints that were neither observed reliably nor repaired."""
        return ~(self.observed_mask | self.imputed_mask)


def validate_sequence(sequence: np.ndarray) -> np.ndarray:
    """Validate the canonical layout without copying or renormalizing it."""
    array = np.asarray(sequence)
    if array.ndim != 4 or array.shape[0] != 3:
        raise ValueError("Expected pose layout (3, time, joints, people)")
    if array.shape[2] < 1 or array.shape[3] < 1:
        raise ValueError("At least one joint and one person slot are required")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Pose coordinates and confidence must use a floating dtype")
    if not np.isfinite(array).all():
        raise ValueError(
            "Pose input must be finite; encode missing joints with confidence 0"
        )
    if np.any(array[2] < 0):
        raise ValueError("Pose confidence scores must be nonnegative")
    return array


def _bounded_vector(vector: np.ndarray, bound: float) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=0)
    scale = np.minimum(1.0, bound / np.maximum(norm, np.finfo(np.float64).tiny))
    return vector * scale


class CausalPoseRecovery:
    """Stateful repair for frames shaped (3, joints, people).

    Create a new instance, or call reset(), at every independent clip or person
    identity reset. Frame count is the time unit: max_gap_frames and max_speed
    must be interpreted at the skeleton stream's sampling rate. Bounds are in
    input coordinate units; no anatomical/bone-length constraint is claimed.
    min_repair_confidence gates repair using the last real observation's score;
    it does not change which real observations pass through to the output.
    """

    def __init__(self, config: RecoveryConfig | None = None):
        self.config = config or RecoveryConfig()
        self.reset()

    def reset(self):
        self._shape = None
        self._position = None
        self._velocity = None
        self._last_observation = None
        self._last_confidence = None
        self._gap = None
        self._active = None
        self._previous_coordinates = None
        self._previous_observed = None
        self.identity_resets = 0

    def step(self, frame: np.ndarray) -> RecoveryResult:
        """Repair one frame; result masks/confidence have shape (joints, people)."""
        array = np.asarray(frame)
        if array.ndim != 3:
            raise ValueError("Expected frame layout (3, joints, people)")
        validate_sequence(array[:, None])
        shape = array.shape[1:]
        if self.config.identity_jump_threshold is not None and shape[0] != 18:
            raise ValueError("The torso identity guard requires OpenPose's 18 joints")
        if self._shape is None:
            self._shape = shape
            self._position = np.zeros((2, *shape), dtype=np.float64)
            self._velocity = np.zeros((2, *shape), dtype=np.float64)
            self._last_observation = np.zeros((2, *shape), dtype=np.float64)
            self._last_confidence = np.zeros(shape, dtype=np.float64)
            self._gap = np.zeros(shape, dtype=np.int64)
            self._active = np.zeros(shape, dtype=bool)
            self._previous_coordinates = np.zeros((2, *shape), dtype=np.float64)
            self._previous_observed = np.zeros(shape, dtype=bool)
        elif shape != self._shape:
            raise ValueError("Joint/person dimensions changed; reset the tracker first")

        config = self.config
        original_confidence = array[2].copy()
        observed = (original_confidence > 0) & (
            original_confidence >= config.min_confidence
        )
        repaired = array.copy()
        imputed = np.zeros(shape, dtype=bool)
        if not config.enabled:
            return RecoveryResult(repaired, original_confidence, observed, imputed)

        resets_this_frame = 0
        if config.identity_jump_threshold is not None:
            anchors = [1, 2, 5, 8, 11]  # Neck, shoulders, hips in OpenPose COCO-18.
            common = observed[anchors] & self._previous_observed[anchors]
            displacement = np.linalg.norm(
                array[:2, anchors] - self._previous_coordinates[:, anchors], axis=0
            )
            for person in range(shape[1]):
                if common[:, person].sum() < 2:
                    continue
                median_jump = np.median(displacement[common[:, person], person])
                if median_jump > config.identity_jump_threshold:
                    self._position[:, :, person] = 0
                    self._velocity[:, :, person] = 0
                    self._last_observation[:, :, person] = 0
                    self._last_confidence[:, person] = 0
                    self._gap[:, person] = 0
                    self._active[:, person] = False
                    resets_this_frame += 1
            self.identity_resets += resets_this_frame
        # Compare only adjacent observed frames. When anchors disappear, the
        # guard cannot establish identity; short-gap bounds remain the fallback.
        self._previous_coordinates[:] = array[:2]
        self._previous_observed[:] = observed

        continuing = observed & self._active
        starting = observed & ~self._active
        predicted = self._position + self._velocity
        residual = array[:2] - predicted
        self._position[:, continuing] = (predicted + config.alpha * residual)[
            :, continuing
        ]
        updated_velocity = _bounded_vector(
            self._velocity + config.beta * residual, config.max_speed
        )
        self._velocity[:, continuing] = updated_velocity[:, continuing]
        self._position[:, starting] = array[:2, starting]
        self._velocity[:, starting] = 0
        self._last_observation[:, observed] = array[:2, observed]
        self._last_confidence[observed] = original_confidence[observed]
        self._gap[observed] = 0
        self._active[observed] = True

        missing = ~observed
        self._gap[missing & self._active] += 1
        eligible_gap = missing & self._active & (self._gap <= config.max_gap_frames)
        imputed = eligible_gap & (self._last_confidence >= config.min_repair_confidence)
        # Anchor extrapolation to the last actual measurement. Anchoring it to
        # the smoothed state could create a large first-gap jump when the filter
        # trails a rapidly moving joint, despite the velocity limit.
        displacement = _bounded_vector(
            self._velocity * self._gap[None], config.max_displacement
        )
        bounded_prediction = self._last_observation + displacement
        # The repair-score gate controls output only. Advancing the same hidden
        # state keeps later reliable observations unaffected by this policy.
        self._position[:, eligible_gap] = bounded_prediction[:, eligible_gap]

        # Invalid observations do not contribute coordinates or tracker updates.
        repaired[:, missing] = 0
        repaired[:2, imputed] = bounded_prediction[:, imputed]
        repaired[2, imputed] = (
            self._last_confidence[imputed]
            * config.confidence_decay ** self._gap[imputed]
        )

        expired = missing & self._active & ~eligible_gap
        self._active[expired] = False
        self._velocity[:, expired] = 0
        # Original confidence is kept separately so repair cannot inflate the
        # number of genuinely observed joints in coverage/quality calculations.
        return RecoveryResult(
            repaired, original_confidence, observed, imputed, resets_this_frame
        )


def recover_pose_sequence(
    sequence: np.ndarray, config: RecoveryConfig | None = None
) -> RecoveryResult:
    """Repair an independent clip without mutating it or using later frames."""
    array = validate_sequence(sequence)
    tracker = CausalPoseRecovery(config)
    repaired = array.copy()
    observed = np.zeros(array.shape[1:], dtype=bool)
    imputed = np.zeros(array.shape[1:], dtype=bool)
    for index in range(array.shape[1]):
        result = tracker.step(array[:, index])
        repaired[:, index] = result.repaired
        observed[index] = result.observed_mask
        imputed[index] = result.imputed_mask
    return RecoveryResult(
        repaired, array[2].copy(), observed, imputed, tracker.identity_resets
    )
