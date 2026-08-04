"""Stage 5 (ODD Mapping): standalone vehicle-safety-envelope formulas.

Implements two verified formulas as pure, unit-testable functions -- NOT
wired into any simulator or control loop, per this project's no-CARLA/
no-closed-loop-simulation scope decision:

1. The RSS (Responsibility-Sensitive Safety) minimum safe longitudinal
   distance (Salvi et al. 2022, "Fuzzy Interpretation of Operational
   Design Domains in Autonomous Driving", IEEE IV, Eq. 1), with the
   paper's own per-uODD parameter table (their Table IV).
2. The vehicle sideslip self-stabilizing-region stability criterion
   (Jiang et al. 2024, "Enhancing Autonomous Vehicle Safety Based on
   Operational Design Domain Definition, Monitoring, and Functional
   Degradation", IEEE TIV, Eqs. 7-9 and 20), including the simplified
   magic-tire-formula lateral force model (Eq. 8) with the paper's fitted
   constants B=0.442, D=0.897.

Both take a single kinematic/tire-state snapshot and return a scalar
margin -- there is no time integration here.
"""

import math
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------
# RSS minimum safe distance (Salvi et al. 2022, Eq. 1 + Table IV)
# --------------------------------------------------------------------------


@dataclass
class RSSParameters:
    """Per-uODD RSS parameters (Salvi et al. 2022, Table IV).

    Attributes:
        a_max: Maximum ego acceleration during the response time rho (m/s^2).
        b_max: Maximum possible braking of the front (leading) vehicle (m/s^2).
        b_min: Minimum ego deceleration after the response time (m/s^2).
        rho: Ego response time (s).
    """

    a_max: float
    b_max: float
    b_min: float
    rho: float


# Exact values from Salvi et al. 2022, Table IV.
RSS_PARAMETERS = {
    "NoRain": RSSParameters(a_max=2.0, b_max=7.5, b_min=4.0, rho=0.2),
    "LowRain": RSSParameters(a_max=2.0, b_max=5.5, b_min=3.0, rho=0.2),
    "HeavyRain": RSSParameters(a_max=2.0, b_max=2.0, b_min=1.0, rho=0.2),
}


def rss_minimum_safe_distance(
    v_ego: float, v_front: float, params: RSSParameters
) -> float:
    """Computes the RSS minimum safe following distance (Salvi et al. Eq. 1).

    d_min = [v_ego*rho + 0.5*a_max*rho^2 + (v_ego + rho*a_max)^2/(2*b_min)
             - v_front^2/(2*b_max)]_+

    Args:
        v_ego: Ego vehicle longitudinal speed (m/s).
        v_front: Front (leading) vehicle longitudinal speed (m/s).
        params: The RSSParameters for the current micro-ODD (uODD).

    Returns:
        The minimum safe distance in meters (never negative).
    """
    response_phase = v_ego * params.rho + 0.5 * params.a_max * params.rho**2
    ego_braking_phase = (v_ego + params.rho * params.a_max) ** 2 / (2 * params.b_min)
    front_braking_phase = v_front**2 / (2 * params.b_max)

    d_min = response_phase + ego_braking_phase - front_braking_phase
    return max(d_min, 0.0)


def rss_parameters_for_mu_odd(mu_odd_label: str) -> RSSParameters:
    """Looks up the RSS parameter set for a fuzzy_odd.py micro-ODD label.

    Args:
        mu_odd_label: One of "NoRain", "LowRain", "HeavyRain" (see fuzzy_odd.py).

    Returns:
        The corresponding RSSParameters.

    Raises:
        KeyError: If mu_odd_label is not a recognized micro-ODD state.
    """
    return RSS_PARAMETERS[mu_odd_label]


def unsafe_distance_membership(d: float, d_unsafe: float, d_safe: float) -> float:
    """Fuzzy 'unsafe situation' membership degree (Salvi et al. Def. 3.3, Eq. 6).

    mu_A(d) = 1               for 0 < d <= d_unsafe
            = 0               for d > d_safe
            = (d_safe - d) / (d_safe - d_unsafe)   otherwise

    Args:
        d: The actual inter-vehicle distance (m).
        d_unsafe: Distance at or below which the situation is fully unsafe.
        d_safe: Distance at or above which the situation is fully safe.

    Returns:
        Membership degree in [0.0, 1.0].
    """
    if d <= d_unsafe:
        return 1.0
    if d > d_safe:
        return 0.0
    return (d_safe - d) / (d_safe - d_unsafe)


# --------------------------------------------------------------------------
# Sideslip stability (Jiang et al. 2024, Eqs. 7-9, 20)
# --------------------------------------------------------------------------

# Magic-tire-formula fitting constants, exact values from Jiang et al. 2024.
TIRE_B = 0.442
TIRE_D = 0.897


def magic_tire_lateral_force(mu: float, f_z: float, alpha: float) -> float:
    """Simplified magic tire formula for lateral tire force (Jiang et al. Eq. 8).

    F_y = mu * F_z * sin(B * arctan(D * alpha))

    Args:
        mu: Road adhesion coefficient.
        f_z: Vertical load on the tire (N).
        alpha: Tire slip angle (rad).

    Returns:
        Lateral tire force (N).
    """
    return mu * f_z * math.sin(TIRE_B * math.atan(TIRE_D * alpha))


@dataclass
class SideslipState:
    """A single-timestep vehicle sideslip state snapshot.

    Attributes:
        beta: Vehicle sideslip angle (rad).
        omega_r: Yaw rate (rad/s).
        v_x: Longitudinal speed (m/s).
        mass: Vehicle mass (kg).
        i_z: Yaw moment of inertia (kg*m^2).
        l_f: Distance from center of gravity to front axle (m).
        l_r: Distance from center of gravity to rear axle (m).
        delta: Front steering angle (rad).
        mu: Road adhesion coefficient.
        f_zf: Vertical load on the front axle (N).
        f_zr: Vertical load on the rear axle (N).
        alpha_f: Front tire slip angle (rad).
        alpha_r: Rear tire slip angle (rad).
    """

    beta: float
    omega_r: float
    v_x: float
    mass: float
    i_z: float
    l_f: float
    l_r: float
    delta: float
    mu: float
    f_zf: float
    f_zr: float
    alpha_f: float
    alpha_r: float


def sideslip_dynamics(state: SideslipState) -> Tuple[float, float]:
    """Computes the instantaneous sideslip/yaw-rate derivatives (Jiang et al. Eq. 7).

    beta_dot   = (cos(beta)/(m*vx))*(Fyf*cos(delta) + Fyr)
                 - (sin(beta)/(m*vx))*Fyf*sin(delta) - omega_r
    omega_dot_r = (Fyf*cos(delta)*Lf - Fyr*Lr) / Iz

    Tire forces Fyf/Fyr are computed via the magic tire formula (Eq. 8).

    Args:
        state: The current SideslipState.

    Returns:
        A tuple (beta_dot, omega_r_dot).

    Raises:
        ValueError: If v_x is zero (division by zero in the model).
    """
    if state.v_x == 0:
        raise ValueError("sideslip_dynamics is undefined at v_x = 0.")

    f_yf = magic_tire_lateral_force(state.mu, state.f_zf, state.alpha_f)
    f_yr = magic_tire_lateral_force(state.mu, state.f_zr, state.alpha_r)

    beta_dot = (
        (math.cos(state.beta) / (state.mass * state.v_x))
        * (f_yf * math.cos(state.delta) + f_yr)
        - (math.sin(state.beta) / (state.mass * state.v_x)) * f_yf * math.sin(state.delta)
        - state.omega_r
    )
    omega_r_dot = (f_yf * math.cos(state.delta) * state.l_f - f_yr * state.l_r) / state.i_z

    return beta_dot, omega_r_dot


@dataclass
class StabilityBoundary:
    """Calibrated self-stabilizing-region boundary parameters (Jiang et al. Eq. 20).

    Attributes:
        b1: Sideslip-rate weighting coefficient.
        b2: Lower sideslip-angle bound.
        b3: Upper sideslip-angle bound.
    """

    b1: float
    b2: float
    b3: float


def compute_stability_boundary(v_x: float, mu: float, delta: float) -> StabilityBoundary:
    """Computes the calibrated self-stabilizing-region boundary (Jiang et al. Eq. 20).

    B1 = 0.0074*vx^3 - 0.005*vx^2 + 0.011*vx + 0.129
    B2 = -(0.61*mu^3 - 0.76*mu^2 + 0.36*mu + 0.02) * (1 - 0.7*delta)
    B3 =  (0.61*mu^3 - 0.76*mu^2 + 0.36*mu + 0.02) * (1 + 0.7*delta)

    Args:
        v_x: Longitudinal speed (m/s).
        mu: Road adhesion coefficient.
        delta: Front steering angle (rad).

    Returns:
        A StabilityBoundary(b1, b2, b3).
    """
    b1 = 0.0074 * v_x**3 - 0.005 * v_x**2 + 0.011 * v_x + 0.129
    envelope = 0.61 * mu**3 - 0.76 * mu**2 + 0.36 * mu + 0.02
    b2 = -envelope * (1 - 0.7 * delta)
    b3 = envelope * (1 + 0.7 * delta)
    return StabilityBoundary(b1=b1, b2=b2, b3=b3)


def is_sideslip_stable(beta: float, beta_dot: float, boundary: StabilityBoundary) -> bool:
    """Checks the self-stabilizing-region stability criterion (Jiang et al. Eq. 9/20).

    lat_stable = 1 iff B2 <= beta + B1*beta_dot <= B3, else 0.

    Args:
        beta: Vehicle sideslip angle (rad).
        beta_dot: Sideslip angle rate (rad/s), as from sideslip_dynamics().
        boundary: The StabilityBoundary for the current speed/road/steering.

    Returns:
        True if the vehicle is within its self-stabilizing region.
    """
    phase_plane_value = beta + boundary.b1 * beta_dot
    return boundary.b2 <= phase_plane_value <= boundary.b3


def stability_margin(beta: float, beta_dot: float, boundary: StabilityBoundary) -> float:
    """Signed distance from the sideslip phase-plane value to the nearer boundary.

    Positive values indicate the state is inside the stable region (larger
    is safer); non-positive values indicate instability.

    Args:
        beta: Vehicle sideslip angle (rad).
        beta_dot: Sideslip angle rate (rad/s).
        boundary: The StabilityBoundary for the current speed/road/steering.

    Returns:
        The signed margin (same units as beta).
    """
    phase_plane_value = beta + boundary.b1 * beta_dot
    return min(phase_plane_value - boundary.b2, boundary.b3 - phase_plane_value)


if __name__ == "__main__":
    for label, params in RSS_PARAMETERS.items():
        d_min = rss_minimum_safe_distance(v_ego=16.7, v_front=13.9, params=params)
        print(f"{label}: RSS min safe distance at 60/50 km/h = {d_min:.2f} m ({params})")

    state = SideslipState(
        beta=0.05,
        omega_r=0.1,
        v_x=20.0,
        mass=1500.0,
        i_z=2500.0,
        l_f=1.2,
        l_r=1.6,
        delta=0.05,
        mu=0.8,
        f_zf=7000.0,
        f_zr=6500.0,
        alpha_f=0.03,
        alpha_r=0.02,
    )
    beta_dot, omega_r_dot = sideslip_dynamics(state)
    boundary = compute_stability_boundary(state.v_x, state.mu, state.delta)
    stable = is_sideslip_stable(state.beta, beta_dot, boundary)
    margin = stability_margin(state.beta, beta_dot, boundary)
    print(f"beta_dot={beta_dot:.4f}, omega_r_dot={omega_r_dot:.4f}")
    print(f"boundary={boundary}, stable={stable}, margin={margin:.4f}")
