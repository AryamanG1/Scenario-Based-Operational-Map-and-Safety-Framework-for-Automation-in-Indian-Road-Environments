"""The shared per-tick contract implemented by both CarlaTickSource and
LocalKinematicTickSource.

This project never uses abc.ABC or class inheritance anywhere in src/
(every "class" here is a @dataclass, a torch.nn.Module subclass, or a
plain stateful utility like odd_classifier.FeatureScaler) -- so this uses
typing.Protocol (structural typing, zero runtime cost, no forced base
class) rather than introducing inheritance machinery for something that
only ever has two implementations.
"""

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np


@dataclass
class TickResult:
    """One simulation tick's observation, in the shape closed_loop_runner.py needs.

    Attributes:
        tick: The tick index (0-based).
        frame: A BGR image array, shaped to match
            data_pipeline.IMAGE_SIZE, ready for SegNet/YOLO inference.
        v_ego: Ego vehicle longitudinal speed (m/s).
        v_front: Front (leading) vehicle longitudinal speed (m/s) -- see
            each TickSource implementation's docstring for how this is
            obtained (a real measurement for CarlaTickSource, a documented
            synthetic profile for LocalKinematicTickSource).
        yaw_rate: Ego yaw rate (rad/s).
        slip_angle: Ego vehicle sideslip angle estimate (rad).
        position: (x, y) position in an arbitrary local ground-plane frame
            (meters), used only to derive `progress`.
        progress: Fraction in [0.0, 1.0] of a notional route completed,
            used by the dashboard replay to place a marker along a road
            polyline.
    """

    tick: int
    frame: np.ndarray
    v_ego: float
    v_front: float
    yaw_rate: float
    slip_angle: float
    position: Tuple[float, float]
    progress: float


class TickSource(Protocol):
    """Structural interface shared by CarlaTickSource and LocalKinematicTickSource."""

    def tick(self) -> TickResult:
        """Advances the simulation by one timestep and returns the observation."""
        ...

    def apply_mode(self, mode: str) -> None:
        """Feeds the current Stage 7 decision back into the simulation.

        CarlaTickSource sets real vehicle physics/weather for `mode`;
        LocalKinematicTickSource adjusts its kinematic stepper's target
        speed (and injects the documented slip-angle disturbance under
        Degraded/Takeover).

        Args:
            mode: One of "Normal", "Degraded", "Takeover".
        """
        ...
