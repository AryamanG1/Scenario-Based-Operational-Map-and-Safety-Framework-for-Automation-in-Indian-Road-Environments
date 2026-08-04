# `rss_safety.py`

**Stage:** 5
**Package:** `src.odd.rss_safety`

## Purpose

This module implements two vehicle-safety-envelope formulas used within Stage 5, "ODD Mapping," as standalone, pure, unit-testable functions: the RSS (Responsibility-Sensitive Safety) minimum safe longitudinal following distance, and the vehicle sideslip self-stabilizing-region stability criterion (including the magic-tire-formula lateral force model that underlies it). Both take a single kinematic/tire-state snapshot and return a scalar margin or distance — there is no time integration or control-loop wiring here.

This module exists specifically to satisfy the project's documented no-CARLA / no-closed-loop-simulation scope decision (see the README's "Known scope limitations"): rather than skip these formulas entirely because there is no simulator to run them in, they are implemented faithfully and exercised with unit tests, so the safety-relevant math itself is verified even though it is not wired into a live control loop.

## Paper / Formula Provenance

**1. RSS minimum safe distance** — Salvi, Weiss, Trapp, Oboril, Buerkle, "Fuzzy Interpretation of Operational Design Domains in Autonomous Driving," IEEE IV 2022, Eq. 1, with the paper's own per-uODD parameter table (their Table IV):

```
d_min = [ v_ego*rho + 0.5*a_max*rho^2
          + (v_ego + rho*a_max)^2 / (2*b_min)
          - v_front^2 / (2*b_max) ]_+
```
(`[...]_+` denotes clamped to be never negative.)

Table IV parameters, reproduced exactly:

| uODD | a_max (m/s²) | b_max (m/s²) | b_min (m/s²) | rho (s) |
|---|---|---|---|---|
| NoRain | 2.0 | 7.5 | 4.0 | 0.2 |
| LowRain | 2.0 | 5.5 | 3.0 | 0.2 |
| HeavyRain | 2.0 | 2.0 | 1.0 | 0.2 |

Also from the same paper, **Def. 3.3 / Eq. 6**, the fuzzy "unsafe situation" membership:

```
mu_A(d) = 1                              for 0 < d <= d_unsafe
        = 0                              for d > d_safe
        = (d_safe - d) / (d_safe - d_unsafe)   otherwise
```

**2. Sideslip stability** — Jiang, Pan, Liu, Han, Pan, Li, Pan, "Enhancing Autonomous Vehicle Safety Based on Operational Design Domain Definition, Monitoring, and Functional Degradation," IEEE TIV, Oct 2024, Eqs. 7-9 and 20.

Simplified magic tire formula for lateral force (Eq. 8), with the paper's fitted constants `B=0.442`, `D=0.897`:

```
F_y = mu * F_z * sin(B * arctan(D * alpha))
```

Sideslip/yaw-rate dynamics (Eq. 7), with tire forces `Fyf`/`Fyr` from the magic tire formula above:

```
beta_dot    = (cos(beta)/(m*vx)) * (Fyf*cos(delta) + Fyr)
              - (sin(beta)/(m*vx)) * Fyf*sin(delta) - omega_r
omega_dot_r = (Fyf*cos(delta)*Lf - Fyr*Lr) / Iz
```

Self-stabilizing-region boundary (Eq. 20):

```
B1 = 0.0074*vx^3 - 0.005*vx^2 + 0.011*vx + 0.129
B2 = -(0.61*mu^3 - 0.76*mu^2 + 0.36*mu + 0.02) * (1 - 0.7*delta)
B3 =  (0.61*mu^3 - 0.76*mu^2 + 0.36*mu + 0.02) * (1 + 0.7*delta)
```

Stability criterion (Eq. 9/20):

```
lat_stable = 1  iff  B2 <= beta + B1*beta_dot <= B3,  else 0
```

## Public API

### RSS minimum safe distance

```python
@dataclass
class RSSParameters:
    a_max: float
    b_max: float
    b_min: float
    rho: float
```
Per-uODD RSS parameters (Salvi et al. 2022, Table IV): `a_max` = max ego acceleration during the response time `rho`; `b_max` = max possible braking of the leading vehicle; `b_min` = min ego deceleration after the response time; `rho` = ego response time.

```python
RSS_PARAMETERS = {
    "NoRain": RSSParameters(a_max=2.0, b_max=7.5, b_min=4.0, rho=0.2),
    "LowRain": RSSParameters(a_max=2.0, b_max=5.5, b_min=3.0, rho=0.2),
    "HeavyRain": RSSParameters(a_max=2.0, b_max=2.0, b_min=1.0, rho=0.2),
}
```

```python
def rss_minimum_safe_distance(
    v_ego: float, v_front: float, params: RSSParameters
) -> float
```
Computes the RSS minimum safe following distance (Eq. 1 above). `v_ego`/`v_front` are longitudinal speeds in m/s. Returns the minimum safe distance in meters, clamped to never be negative (`max(d_min, 0.0)`).

```python
def rss_parameters_for_mu_odd(mu_odd_label: str) -> RSSParameters
```
Looks up the RSS parameter set for a `fuzzy_odd.py` micro-ODD label (one of `"NoRain"`, `"LowRain"`, `"HeavyRain"`). Raises `KeyError` if `mu_odd_label` is not a recognized micro-ODD state.

```python
def unsafe_distance_membership(d: float, d_unsafe: float, d_safe: float) -> float
```
Fuzzy "unsafe situation" membership degree (Salvi et al. Def. 3.3, Eq. 6), as defined above. Returns a value in `[0.0, 1.0]`.

### Sideslip stability

```python
TIRE_B = 0.442
TIRE_D = 0.897
```
Magic-tire-formula fitting constants, exact values from Jiang et al. 2024.

```python
def magic_tire_lateral_force(mu: float, f_z: float, alpha: float) -> float
```
Simplified magic tire formula for lateral tire force (Eq. 8): `mu * f_z * sin(TIRE_B * atan(TIRE_D * alpha))`. `mu` = road adhesion coefficient, `f_z` = vertical tire load (N), `alpha` = tire slip angle (rad). Returns lateral tire force in N.

```python
@dataclass
class SideslipState:
    beta: float       # sideslip angle (rad)
    omega_r: float    # yaw rate (rad/s)
    v_x: float        # longitudinal speed (m/s)
    mass: float       # kg
    i_z: float        # yaw moment of inertia (kg*m^2)
    l_f: float        # CG-to-front-axle distance (m)
    l_r: float        # CG-to-rear-axle distance (m)
    delta: float      # front steering angle (rad)
    mu: float         # road adhesion coefficient
    f_zf: float       # front axle vertical load (N)
    f_zr: float       # rear axle vertical load (N)
    alpha_f: float    # front tire slip angle (rad)
    alpha_r: float    # rear tire slip angle (rad)
```
A single-timestep vehicle sideslip state snapshot.

```python
def sideslip_dynamics(state: SideslipState) -> Tuple[float, float]
```
Computes the instantaneous sideslip/yaw-rate derivatives (Eq. 7), with tire forces computed via `magic_tire_lateral_force` (Eq. 8). Returns `(beta_dot, omega_r_dot)`. **Raises `ValueError` if `state.v_x == 0`** (division by zero in the model — the dynamics equations are undefined at zero longitudinal speed).

```python
@dataclass
class StabilityBoundary:
    b1: float   # sideslip-rate weighting coefficient
    b2: float   # lower sideslip-angle bound
    b3: float   # upper sideslip-angle bound
```
Calibrated self-stabilizing-region boundary parameters (Eq. 20).

```python
def compute_stability_boundary(v_x: float, mu: float, delta: float) -> StabilityBoundary
```
Computes `B1`/`B2`/`B3` per Eq. 20 above from longitudinal speed, road adhesion coefficient, and front steering angle. Returns a `StabilityBoundary`.

```python
def is_sideslip_stable(beta: float, beta_dot: float, boundary: StabilityBoundary) -> bool
```
Checks the self-stabilizing-region stability criterion (Eq. 9/20): `True` iff `boundary.b2 <= beta + boundary.b1*beta_dot <= boundary.b3`.

```python
def stability_margin(beta: float, beta_dot: float, boundary: StabilityBoundary) -> float
```
Signed distance from the sideslip phase-plane value (`beta + b1*beta_dot`) to the *nearer* boundary: `min(phase_plane_value - b2, b3 - phase_plane_value)`. Positive values indicate the state is inside the stable region (larger = safer); non-positive values indicate instability.

## Key Design Decisions & Edge Cases

- **Standalone, unit-tested functions — not wired into a simulator or control loop.** This is a deliberate, documented scope decision: this machine has no discrete GPU and CARLA is not installed, so these formulas are verified in isolation rather than in closed-loop simulation.
- **Single-snapshot only, no time integration.** `sideslip_dynamics` returns instantaneous derivatives (`beta_dot`, `omega_r_dot`); nothing in this module integrates them forward in time to produce a trajectory.
- **RSS distance is clamped to zero, never negative** (`max(d_min, 0.0)`) — e.g. when the ego vehicle is stationary and the front vehicle is fast, the formula's raw value can be negative, which correctly means "no minimum distance is required," not a nonsensical negative distance.
- **`sideslip_dynamics` raises `ValueError` at `v_x = 0`** rather than returning `inf`/`NaN`, since both `beta_dot` terms divide by `v_x`.
- **`stability_margin` is signed and takes the minimum of both boundary distances**, so it degrades gracefully near either edge of the stable region rather than only checking one side.
- **`rss_parameters_for_mu_odd` is a plain dict lookup with no fuzzy blending** — even though the upstream `fuzzy_odd.py` module produces continuous (fuzzy) membership degrees across all three uODD states, this function expects one crisp label and raises `KeyError` on anything else.
- **This module has zero internal project dependencies** (no import of `fuzzy_odd.py` even though `rss_parameters_for_mu_odd`'s docstring references its labels by name/convention) — the coupling is purely a documented naming convention, not a code import.

## Dependencies

- Standard library: `math`, `dataclasses`, `typing`
- External: none
- Internal: none — this module is fully standalone (no `src.common.paths` import even in `__main__`, unlike the other five ODD modules)

## Usage Example

```python
from src.odd.rss_safety import (
    RSS_PARAMETERS, rss_minimum_safe_distance, rss_parameters_for_mu_odd,
    SideslipState, sideslip_dynamics, compute_stability_boundary,
    is_sideslip_stable, stability_margin,
)

params = rss_parameters_for_mu_odd("HeavyRain")
d_min = rss_minimum_safe_distance(v_ego=16.7, v_front=13.9, params=params)  # m/s ~= 60/50 km/h

state = SideslipState(
    beta=0.05, omega_r=0.1, v_x=20.0, mass=1500.0, i_z=2500.0,
    l_f=1.2, l_r=1.6, delta=0.05, mu=0.8, f_zf=7000.0, f_zr=6500.0,
    alpha_f=0.03, alpha_r=0.02,
)
beta_dot, omega_r_dot = sideslip_dynamics(state)
boundary = compute_stability_boundary(state.v_x, state.mu, state.delta)
stable = is_sideslip_stable(state.beta, beta_dot, boundary)
margin = stability_margin(state.beta, beta_dot, boundary)
```

## Running Standalone

```bash
python -m src.odd.rss_safety
```
Prints the RSS minimum safe distance at 60/50 km/h (`v_ego=16.7`, `v_front=13.9` m/s) for each of the three `RSS_PARAMETERS` uODD entries (NoRain/LowRain/HeavyRain), then builds one sample `SideslipState`, computes `beta_dot`/`omega_r_dot`, the stability boundary, the stability verdict, and the stability margin, and prints them.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/odd/test_rss_safety.py` covers: RSS distance clamping to zero for a stationary ego vehicle, an exact hand-computation match against the Eq. 1 formula, monotonically increasing RSS distance as weather worsens (NoRain < LowRain < HeavyRain), the unsafe-distance membership function's boundary and midpoint values, the magic tire formula being zero at zero slip angle and saturating for large slip angles, `sideslip_dynamics` raising `ValueError` at zero speed, the stability boundary being symmetric (`b2 == -b3`) at zero steering, stability holding at the zero state, and instability at an extreme phase-plane state.
