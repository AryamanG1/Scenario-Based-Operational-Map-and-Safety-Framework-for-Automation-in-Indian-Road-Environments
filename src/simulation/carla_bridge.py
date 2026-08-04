"""Real CARLA client integration (CARLA 0.9.15 Python API).

IMPORTANT -- environment limitation, stated plainly: this module is
correct against the documented CARLA 0.9.15 Python API (verified by
reading CARLA's own API reference), but it is **reviewed-by-reading only,
not execution-verified**, in this project's development environment. This
machine has no discrete GPU (CARLA's server requires one to render, even
headless) and the `carla` PyPI package has no wheel for this project's
Python 3.12 venv (CARLA 0.9.13-0.9.15 wheels only support CPython
3.7-3.10) -- confirmed directly: `import carla` raises ModuleNotFoundError
here. See docs/FUTURE_STEPS.md for exactly what a capable machine needs to
actually exercise this path, and rss_safety.py's own module docstring for
this project's precedent of "implemented faithfully, verified in
isolation" for formulas this sandbox categorically cannot run end-to-end.

Every formula this module feeds (RSS distance, sideslip stability) lives
in src/odd/rss_safety.py and is called from closed_loop_runner.py, not
reimplemented here -- this module's only job is to turn real CARLA actor
state into a TickResult and to actuate CARLA physics/weather from a
CarlaConfig's per-mode profiles.
"""

import queue

import numpy as np

from src.simulation.carla_config import CarlaConfig
from src.simulation.tick_source import TickResult


def is_carla_available(config: CarlaConfig) -> bool:
    """Probes whether a real CARLA server is reachable, without raising.

    This is the master fallback trigger: closed_loop_runner.py calls this
    once at startup (when config.use_carla is True) and silently uses
    LocalKinematicTickSource instead if it returns False -- satisfying
    "if carla is not attached I can run the rest of the subsystem."

    Args:
        config: Connection parameters (host, port, timeout_s).

    Returns:
        True iff the `carla` package is importable AND a server answers
        at config.host:config.port within config.timeout_s. False on any
        failure (ImportError, connection refused, version mismatch,
        timeout) -- never raises.
    """
    try:
        import carla
    except ImportError:
        return False

    try:
        client = carla.Client(config.host, config.port)
        client.set_timeout(config.timeout_s)
        client.get_server_version()
        return True
    except Exception:
        return False


class CarlaTickSource:
    """Wraps a live CARLA client/world/ego-vehicle/camera as a TickSource.

    Structurally satisfies simulation.tick_source.TickSource (tick(),
    apply_mode()) with no shared base class, per this project's
    no-inheritance style.
    """

    def __init__(self, config: CarlaConfig) -> None:
        """Connects to CARLA and spawns the ego vehicle, camera, and lead vehicle.

        Args:
            config: Full simulation configuration.

        Raises:
            RuntimeError: If connection, world load, or actor spawn fails.
                closed_loop_runner.py catches this at startup (falls back
                to LocalKinematicTickSource) or mid-run (per
                config.fallback_on_error).
        """
        import carla

        self._carla = carla
        self.config = config
        self._current_mode = "Normal"
        self._tick_count = 0
        self._route_length_m = 200.0  # notional route length used only for `progress`
        self._distance_travelled_m = 0.0

        self.client = carla.Client(config.host, config.port)
        self.client.set_timeout(config.timeout_s)
        self.world = self.client.load_world(config.town)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config.tick_dt
        self.world.apply_settings(settings)

        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"CARLA town '{config.town}' has no spawn points.")
        ego_spawn = spawn_points[0]

        vehicle_bp = blueprint_library.find(config.vehicle_blueprint)
        self.vehicle = self.world.spawn_actor(vehicle_bp, ego_spawn)

        lead_bp = blueprint_library.find(config.lead_vehicle_blueprint)
        forward = ego_spawn.get_forward_vector()
        lead_location = carla.Location(
            x=ego_spawn.location.x + forward.x * config.lead_vehicle_spawn_gap_m,
            y=ego_spawn.location.y + forward.y * config.lead_vehicle_spawn_gap_m,
            z=ego_spawn.location.z,
        )
        lead_spawn = carla.Transform(lead_location, ego_spawn.rotation)
        self.lead_vehicle = self.world.spawn_actor(lead_bp, lead_spawn)
        self.lead_vehicle.set_autopilot(True)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(config.camera_width))
        camera_bp.set_attribute("image_size_y", str(config.camera_height))
        camera_transform = carla.Transform(carla.Location(x=config.camera_x, z=config.camera_z))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        self._image_queue: "queue.Queue" = queue.Queue()
        self.camera.listen(self._image_queue.put)

    def _carla_image_to_bgr(self, image) -> np.ndarray:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        return array[:, :, :3].copy()

    def tick(self) -> TickResult:
        """Advances CARLA by one synchronous tick and reads back ego/lead state.

        Returns:
            A TickResult with a real camera frame and real ego/lead kinematics.
        """
        self.world.tick()
        image = self._image_queue.get()
        frame = self._carla_image_to_bgr(image)

        velocity = self.vehicle.get_velocity()
        v_ego = float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))

        lead_velocity = self.lead_vehicle.get_velocity()
        v_front = float(np.linalg.norm([lead_velocity.x, lead_velocity.y, lead_velocity.z]))

        angular_velocity = self.vehicle.get_angular_velocity()  # deg/s
        yaw_rate = float(np.deg2rad(angular_velocity.z))

        transform = self.vehicle.get_transform()
        heading_rad = float(np.deg2rad(transform.rotation.yaw))
        velocity_heading_rad = float(np.arctan2(velocity.y, velocity.x)) if v_ego > 1e-3 else heading_rad
        slip_angle = float(np.arctan2(np.sin(velocity_heading_rad - heading_rad), np.cos(velocity_heading_rad - heading_rad)))

        self._distance_travelled_m += v_ego * self.config.tick_dt
        progress = min(1.0, self._distance_travelled_m / self._route_length_m)

        result = TickResult(
            tick=self._tick_count,
            frame=frame,
            v_ego=v_ego,
            v_front=v_front,
            yaw_rate=yaw_rate,
            slip_angle=slip_angle,
            position=(transform.location.x, transform.location.y),
            progress=progress,
        )
        self._tick_count += 1
        return result

    def apply_mode(self, mode: str) -> None:
        """Sets CARLA vehicle physics + weather for `mode`, and steers toward its target speed.

        Args:
            mode: One of "Normal", "Degraded", "Takeover".
        """
        carla = self._carla
        if mode != self._current_mode:
            physics_profile = self.config.physics_profiles[mode]
            physics_control = self.vehicle.get_physics_control()
            wheels = physics_control.wheels
            for wheel in wheels:
                wheel.tire_friction = physics_profile.tire_friction
            physics_control.wheels = wheels
            physics_control.damping_rate_full_throttle = physics_profile.damping_rate_full_throttle
            physics_control.max_rpm = physics_profile.max_rpm
            self.vehicle.apply_physics_control(physics_control)

            weather_profile = self.config.weather_profiles[mode]
            self.world.set_weather(
                carla.WeatherParameters(
                    cloudiness=weather_profile.cloudiness,
                    precipitation=weather_profile.precipitation,
                    fog_density=weather_profile.fog_density,
                    wetness=weather_profile.wetness,
                )
            )
            self._current_mode = mode

        target_speed = self.config.physics_profiles[mode].target_speed_mps
        velocity = self.vehicle.get_velocity()
        current_speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
        speed_error = target_speed - current_speed
        throttle = max(0.0, min(1.0, speed_error * 0.5))
        brake = max(0.0, min(1.0, -speed_error * 0.5))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))

    def close(self) -> None:
        """Destroys spawned actors and restores asynchronous world settings."""
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        for actor in (getattr(self, "camera", None), getattr(self, "lead_vehicle", None), getattr(self, "vehicle", None)):
            if actor is not None:
                actor.destroy()
