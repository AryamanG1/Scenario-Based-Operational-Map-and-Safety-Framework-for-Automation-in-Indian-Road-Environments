"""The no-CARLA fallback tick source: real IDD-Lite frames + a local kinematic stepper.

Used whenever CarlaConfig.use_carla is False, carla_bridge.is_carla_available()
returns False, or force_fallback is passed to run_closed_loop_simulation() --
i.e. exactly the "if carla is not attached I can run the rest of the
subsystem" requirement. Requires no GPU, no external server, and no
`carla` package; pure numpy plus the same dataset loader the rest of this
project already uses.

Frame source: cycles through data_pipeline.load_and_clean_dataset()'s
train-split images, exactly like perception_monitor.py's existing
precedent for "no live simulator -- reuse real static images instead of
fabricating camera frames." There is no validation-split loader anywhere
in this codebase (see docs/FUTURE_STEPS.md for that as a real gap, not
invented here to avoid scope creep).

Kinematics: a first-order (single time-constant) speed-convergence model
targeting the current mode's PhysicsProfile.target_speed_mps, moving in a
straight line (yaw_rate is always 0 here -- there is no route/steering
model in the fallback, only in CarlaTickSource, which follows CARLA's
actual road network). Under Degraded/Takeover, a fixed nonzero slip angle
is injected deliberately, so rss_safety.is_sideslip_stable()/
stability_margin() have something real to react to -- this is a chosen
test scenario, not simulated noise, exactly parallel to how
perturbation_engine.py's perturbations are deliberately chosen, not random
noise for its own sake.

v_front: IDD-Lite has no velocity or object-tracking signal of any kind
(single static images), so there is no real measurement to fall back on
here (contrast with CarlaTickSource, which reads an actual lead-vehicle
actor's velocity). v_front is therefore an explicitly synthetic, labeled
profile: v_front = clamp(v_ego - fallback_closing_rate_mps, 0, v_ego).
"""

import math

import numpy as np

from src.simulation.carla_config import CarlaConfig
from src.simulation.tick_source import TickResult

_SPEED_TIME_CONSTANT_S = 2.0  # seconds to close ~63% of the gap to target speed
_SLIP_ANGLE_BY_MODE = {
    "Normal": 0.0,
    "Degraded": 0.03,
    "Takeover": 0.08,
}


class LocalKinematicTickSource:
    """A GPU-free, CARLA-free TickSource: real frames + a simple bicycle-model stepper.

    Structurally satisfies simulation.tick_source.TickSource (tick(),
    apply_mode()) with no shared base class, per this project's
    no-inheritance style.
    """

    def __init__(self, config: CarlaConfig, images: np.ndarray, route_length_m: float = 200.0) -> None:
        """Initializes the fallback tick source.

        Args:
            config: Full simulation configuration.
            images: uint8 image array, shape (M, H, W, 3) BGR, as returned
                by data_pipeline.load_and_clean_dataset(). Must be
                non-empty.
            route_length_m: Notional total route length (m) used only to
                compute TickResult.progress for the dashboard replay.

        Raises:
            ValueError: If `images` is empty.
        """
        if len(images) == 0:
            raise ValueError("LocalKinematicTickSource requires at least one image.")
        self.config = config
        self.images = images
        self._route_length_m = route_length_m

        self._cursor = 0
        self._tick_count = 0
        self.v_ego = 0.0
        self.heading = 0.0
        self.x = 0.0
        self.y = 0.0
        self._distance_travelled_m = 0.0
        self._current_mode = "Normal"

    def tick(self) -> TickResult:
        """Advances the kinematic stepper by one tick_dt and returns the observation.

        Returns:
            A TickResult with a real (cycled) IDD-Lite frame and simulated
            ego kinematics.
        """
        frame = self.images[self._cursor % len(self.images)]
        self._cursor += 1

        dt = self.config.tick_dt
        target_speed = self.config.physics_profiles[self._current_mode].target_speed_mps
        alpha = min(1.0, dt / _SPEED_TIME_CONSTANT_S)
        self.v_ego += (target_speed - self.v_ego) * alpha

        self.x += self.v_ego * math.cos(self.heading) * dt
        self.y += self.v_ego * math.sin(self.heading) * dt
        self._distance_travelled_m += self.v_ego * dt

        slip_angle = _SLIP_ANGLE_BY_MODE[self._current_mode]
        v_front = max(0.0, min(self.v_ego, self.v_ego - self.config.fallback_closing_rate_mps))

        progress = min(1.0, self._distance_travelled_m / self._route_length_m)

        result = TickResult(
            tick=self._tick_count,
            frame=frame,
            v_ego=self.v_ego,
            v_front=v_front,
            yaw_rate=0.0,
            slip_angle=slip_angle,
            position=(self.x, self.y),
            progress=progress,
        )
        self._tick_count += 1
        return result

    def apply_mode(self, mode: str) -> None:
        """Records the current mode; the next tick() converges toward its target speed.

        Args:
            mode: One of "Normal", "Degraded", "Takeover".
        """
        self._current_mode = mode
