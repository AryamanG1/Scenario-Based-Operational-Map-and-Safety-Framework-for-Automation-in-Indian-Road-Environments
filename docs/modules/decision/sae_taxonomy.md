# `sae_taxonomy.py`

**Stage:** Cross-cutting (not numbered 1-7 in the README's module table; used by Stage 7's decision framing)
**Package:** `src.decision.sae_taxonomy`

## Purpose

This module implements the SAE J3016 driving-automation taxonomy as literal, testable code: a Level 0-5 classifier driven by a system's Dynamic Driving Task (DDT) capability profile, and a deterministic state machine modeling how an Automated Driving System (ADS) transitions between engagement states (engaged, ODD-exit warning, request to intervene, driver fallback, minimal risk condition). It exists to formally pin down and defend this project's SAE automation-level claim, and to give the rest of the pipeline (particularly `feasibility_map.py` and the Stage 7 decision system) a principled way to talk about "Normal/Degraded/Takeover" in SAE's own vocabulary.

Concretely, this module also documents and fixes a scope correction: an earlier draft of this project (seeded from a one-shot LLM scaffold) incorrectly targeted SAE Level 3. This module fixes that at the source — `classify_sae_level()` applied to this project's actual DDT configuration (`THIS_PROJECT_DDT`) returns 2, not 3, matching the capstone proposal's explicit SAE Level 2 commitment.

## Paper / Formula Provenance

Implements **SAE J3016 (Rev. APR2021), "Taxonomy and Definitions for Terms Related to Driving Automation Systems for On-Road Motor Vehicles"**, per the README's citation table and the module docstring:

- The **Level 0-5 classification decision tree** (SAE's Table 1 / Figure 10), driven by 5 binary criteria: lateral control, longitudinal control, both simultaneously, OEDR (Object and Event Detection and Response), DDT fallback, and whether the ODD is unlimited.
- The **ADS engagement state machine** (SAE's Figures 3-8, 13-14): ADS Engaged → ODD-exit warning → Request to Intervene → DDT Fallback → Minimal Risk Condition, implemented as a deterministic, testable `ADSStateMachine`.

This is a literal, direct implementation of the standard's decision tree and state diagrams, not an adaptation — unlike `ecofusion_gate.py` or `perception_monitor.py`, there is no simulated-data workaround here.

## Public API

```python
@dataclass
class DynamicDrivingTask:
    lateral_control: bool = False
    longitudinal_control: bool = False
    oedr: bool = False
    ddt_fallback: bool = False
    odd_unlimited: bool = False
```
The Dynamic Driving Task (DDT) capability profile of a system (SAE J3016 Sec. 3). `oedr`: whether the *system* performs Object and Event Detection and Response (if `False`, the human driver performs OEDR). `ddt_fallback`: whether the system automatically achieves a Minimal Risk Condition without driver intervention. `odd_unlimited`: whether the system's ODD is unlimited versus limited.

```python
LEVEL_NAMES = {
    0: "No Driving Automation",
    1: "Driver Assistance",
    2: "Partial Driving Automation",
    3: "Conditional Driving Automation",
    4: "High Driving Automation",
    5: "Full Driving Automation",
}
```

```python
def classify_sae_level(ddt: DynamicDrivingTask) -> int
```
Classifies a system's SAE level from its DDT profile (Table 1 decision tree):
- Neither lateral nor longitudinal control → **0**.
- Either axis controlled, but not OEDR → **2** if both axes are controlled simultaneously, else **1**.
- OEDR performed by the system, but not DDT fallback → **3**.
- Both OEDR and DDT fallback performed by the system → **5** if ODD is unlimited, else **4**.

Returns an integer in `[0, 5]`.

```python
THIS_PROJECT_DDT = DynamicDrivingTask(
    lateral_control=True,
    longitudinal_control=True,
    oedr=False,
    ddt_fallback=False,
    odd_unlimited=False,
)
```
This project's actual DDT capability profile: the ODD classification pipeline sustains both lateral and longitudinal control *decisions* (speed/following-distance advisories via `rss_safety.py`) within a strictly limited ODD (IDD-Lite Indian road scenes only), but OEDR and DDT fallback both remain the driver's responsibility. `classify_sae_level(THIS_PROJECT_DDT)` evaluates to **2**, matching the capstone proposal.

```python
class ADSState(Enum):
    ENGAGED = "ADS Engaged"
    ODD_EXIT_WARNING = "ODD Exit Warning"
    REQUEST_TO_INTERVENE = "Request to Intervene"
    DRIVER_ENGAGED = "Driver Performing DDT"
    FALLBACK = "DDT Fallback"
    MINIMAL_RISK_CONDITION = "Minimal Risk Condition"
```

```python
@dataclass
class ADSStateMachine:
    state: ADSState = ADSState.ENGAGED
    sae_level: int = 2

    def step(self, within_odd: bool, driver_responsive: bool = True) -> ADSState: ...
    def reset(self) -> None: ...
```
A deterministic ADS engagement state machine (SAE J3016 Figs. 3-8, 13-14).

`step(within_odd, driver_responsive=True) -> ADSState` advances the machine by one observation:
- `ENGAGED`: if `not within_odd`, transitions to `ODD_EXIT_WARNING`; otherwise stays.
- `ODD_EXIT_WARNING`: back to `ENGAGED` if `within_odd`, else to `REQUEST_TO_INTERVENE`.
- `REQUEST_TO_INTERVENE`: to `DRIVER_ENGAGED` if `driver_responsive`, else to `FALLBACK`.
- `FALLBACK`: always advances to `MINIMAL_RISK_CONDITION`.
- `MINIMAL_RISK_CONDITION`: terminal — no further automatic transition ("Terminal until externally reset (e.g. vehicle stopped, driver resumes)").
- `DRIVER_ENGAGED`: returns to `ENGAGED` if `within_odd`, else stays.

Returns the resulting `ADSState` (also stored in `self.state`).

`reset() -> None` resets the state machine back to `ENGAGED` (e.g. after a driver takeover completes).

```python
def map_odd_mode_to_sae_state(odd_mode: str) -> ADSState
```
Maps this project's `"Normal"`/`"Degraded"`/`"Takeover"` mode (from `odd_classifier.py`) onto the corresponding `ADSState`, via the internal lookup `_ODD_MODE_TO_ADS_STATE = {"Normal": ENGAGED, "Degraded": ODD_EXIT_WARNING, "Takeover": REQUEST_TO_INTERVENE}`. Raises `KeyError` for any other string.

## Key Design Decisions & Edge Cases

- **The Level 3 → Level 2 correction is the central documented decision in this module.** The docstring explicitly states an earlier one-shot-scaffold draft incorrectly targeted Level 3, and that this module "fixes that error at the source rather than only in documentation" — `classify_sae_level(THIS_PROJECT_DDT)` is asserted (in the standalone block) to equal 2.
- **`MINIMAL_RISK_CONDITION` is a true terminal state** within `step()` — no `within_odd`/`driver_responsive` combination will move the machine out of it automatically; only `reset()` does, modeling that resuming from a minimal-risk-condition stop is an external/driver action, not something the monitoring loop itself decides.
- **`DRIVER_ENGAGED` only returns to `ENGAGED`, never advances further on its own** while `within_odd` is `False` — the driver stays in control until the ODD condition that triggered the takeover clears.
- **The Level 1/2 split hinges on simultaneous dual-axis control**, not on OEDR — per Table 1, a system controlling only one axis (lateral OR longitudinal) without OEDR is Level 1, while a system controlling both axes without OEDR is Level 2. This is directly testable and exercised by the test suite's parametrized decision-tree cases.
- **This module is purely a taxonomy/state-machine — it doesn't consume live pipeline data.** `map_odd_mode_to_sae_state()` is the only bridge from the rest of the pipeline's actual Normal/Degraded/Takeover outputs into SAE vocabulary; nothing here computes ODD status itself.

## Dependencies

**Internal:** None — this module has no imports from other project modules; it's a self-contained taxonomy/state-machine implementation. (It is *referenced conceptually* by `odd_classifier.py`'s Normal/Degraded/Takeover modes via `map_odd_mode_to_sae_state`, but does not itself import from `odd_classifier.py`.)

**External:** `dataclasses.dataclass`, `enum.Enum` (both standard library only — no third-party dependencies).

## Usage Example

```python
from src.decision.sae_taxonomy import (
    classify_sae_level, THIS_PROJECT_DDT, ADSState, ADSStateMachine, map_odd_mode_to_sae_state,
)

print(classify_sae_level(THIS_PROJECT_DDT))   # 2

machine = ADSStateMachine()
for odd_mode in ["Normal", "Degraded", "Takeover", "Takeover", "Normal"]:
    target_state = map_odd_mode_to_sae_state(odd_mode)
    within_odd = target_state == ADSState.ENGAGED
    new_state = machine.step(within_odd=within_odd, driver_responsive=True)
    print(odd_mode, "->", new_state.value)
```

## Running Standalone

```bash
python -m src.decision.sae_taxonomy
```

Prints this project's classified SAE level and name, asserts it equals 2, then prints a full sanity check of Table 1's decision tree at each level boundary (0, 1, 3, 4, 5), and finally walks an `ADSStateMachine` through the sequence `["Normal", "Degraded", "Takeover", "Takeover", "Normal"]`, printing the resulting state after each step.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/decision/test_sae_taxonomy.py` covers: `classify_sae_level` against a parametrized set of `DynamicDrivingTask` configurations spanning Table 1's full decision tree (Levels 0, 1 via each single axis, 2, 3, 4, 5); `THIS_PROJECT_DDT` classifies as Level 2; the state machine's `ENGAGED → ODD_EXIT_WARNING` transition; a full takeover sequence from `ENGAGED` through `ODD_EXIT_WARNING → REQUEST_TO_INTERVENE → FALLBACK → MINIMAL_RISK_CONDITION` (with `driver_responsive=False`); recovery back to `ENGAGED` when ODD returns; `reset()` behavior; and `map_odd_mode_to_sae_state` for all three `"Normal"`/`"Degraded"`/`"Takeover"` inputs.
