"""Tests for carla_bridge.py.

CarlaTickSource's live-server path (connecting to and driving a real
CARLA simulator) gets NO automated test here -- that is an intentional,
permanent limitation of this development environment, not a gap to chase:
this machine has no discrete GPU (CARLA's server needs one even headless)
and the `carla` PyPI package has no wheel for this project's Python 3.12
venv. See docs/modules/simulation/carla_bridge.md and
docs/FUTURE_STEPS.md. What IS tested here, directly, is the exact
fallback-trigger condition closed_loop_runner.py depends on:
is_carla_available() must return False cleanly (never raise) whenever
CARLA isn't reachable -- which is unconditionally true in this sandbox
right now.
"""

import pytest

from src.simulation.carla_bridge import is_carla_available
from src.simulation.carla_config import CarlaConfig


def test_carla_package_is_not_importable_in_this_environment():
    """Documents the real, current state of this sandbox (see module docstring)."""
    try:
        import carla  # noqa: F401
    except ImportError:
        pass


def test_is_carla_available_returns_false_without_a_reachable_server():
    config = CarlaConfig(host="localhost", port=2000, timeout_s=0.5)
    assert is_carla_available(config) is False


def test_is_carla_available_never_raises_on_an_unreachable_host():
    config = CarlaConfig(host="203.0.113.1", port=2000, timeout_s=0.5)  # TEST-NET-3, RFC 5737
    assert is_carla_available(config) is False


def test_is_carla_available_never_raises_on_an_invalid_port():
    config = CarlaConfig(host="localhost", port=1, timeout_s=0.5)
    assert is_carla_available(config) is False
