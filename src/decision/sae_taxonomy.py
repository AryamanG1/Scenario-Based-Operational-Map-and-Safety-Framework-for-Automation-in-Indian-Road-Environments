"""Cross-cutting: SAE J3016 driving-automation taxonomy and state machine.

Implements SAE J3016 (Rev. APR2021, "Taxonomy and Definitions for Terms
Related to Driving Automation Systems for On-Road Motor Vehicles") as
literal code:

- The Level 0-5 classification decision tree (their Table 1 / Figure 10),
  driven by 5 binary criteria: does the system perform lateral or
  longitudinal control, both simultaneously, OEDR (Object and Event
  Detection and Response), DDT fallback, and whether its ODD is limited.
- The ADS engagement state machine (their Figures 3-8, 13-14: ADS Engaged
  -> ODD-exit warning -> Request to Intervene -> DDT Fallback -> Minimal
  Risk Condition), as a deterministic, testable ADSStateMachine.

This project targets SAE Level 2 (Partial Driving Automation) per the
capstone proposal's explicit commitment -- an earlier draft (the one-shot
prompt this project started from) incorrectly targeted Level 3. At Level
2, the driver performs OEDR and DDT fallback; the system sustains both
lateral and longitudinal control within a limited ODD. This module fixes
that error at the source rather than only in documentation: classify_sae_level()
applied to this project's actual DynamicDrivingTask configuration (see
THIS_PROJECT_DDT below) returns 2, not 3.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class DynamicDrivingTask:
    """The Dynamic Driving Task (DDT) capability profile of a system (SAE J3016 Sec. 3).

    Attributes:
        lateral_control: System sustains lateral vehicle motion control.
        longitudinal_control: System sustains longitudinal vehicle motion control.
        oedr: System performs Object and Event Detection and Response
            (monitoring the environment and responding to it) -- if False,
            the human driver performs OEDR.
        ddt_fallback: System automatically performs DDT fallback (achieving
            a Minimal Risk Condition) without driver intervention.
        odd_unlimited: The system's Operational Design Domain is unlimited
            (can operate under all road/environmental/traffic conditions a
            human driver could manage), as opposed to a limited ODD.
    """

    lateral_control: bool = False
    longitudinal_control: bool = False
    oedr: bool = False
    ddt_fallback: bool = False
    odd_unlimited: bool = False


LEVEL_NAMES = {
    0: "No Driving Automation",
    1: "Driver Assistance",
    2: "Partial Driving Automation",
    3: "Conditional Driving Automation",
    4: "High Driving Automation",
    5: "Full Driving Automation",
}


def classify_sae_level(ddt: DynamicDrivingTask) -> int:
    """Classifies a system's SAE level from its DDT capability profile (Table 1).

    Decision tree:
        Neither lateral nor longitudinal control -> Level 0.
        Either lateral or longitudinal, but not OEDR -> Level 1 or 2
            (2 if both axes are controlled simultaneously, else 1).
        OEDR is performed by the system, but not DDT fallback -> Level 3.
        OEDR and DDT fallback both performed by the system -> Level 4 or 5
            (5 if the ODD is unlimited, else 4).

    Args:
        ddt: The system's DynamicDrivingTask capability profile.

    Returns:
        An integer SAE level in [0, 5].
    """
    if not (ddt.lateral_control or ddt.longitudinal_control):
        return 0

    if not ddt.oedr:
        return 2 if (ddt.lateral_control and ddt.longitudinal_control) else 1

    if not ddt.ddt_fallback:
        return 3

    return 5 if ddt.odd_unlimited else 4


# This project's actual DDT capability profile: the ODD classification
# pipeline sustains both lateral and longitudinal control decisions
# (speed/following-distance advisories via rss_safety.py) within a
# strictly limited ODD (IDD-Lite Indian road scenes only), but OEDR and
# DDT fallback both remain the driver's responsibility -- per
# classify_sae_level(), this is Level 2, matching the capstone proposal.
THIS_PROJECT_DDT = DynamicDrivingTask(
    lateral_control=True,
    longitudinal_control=True,
    oedr=False,
    ddt_fallback=False,
    odd_unlimited=False,
)


class ADSState(Enum):
    """ADS engagement states (SAE J3016 Figs. 3-8, 13-14)."""

    ENGAGED = "ADS Engaged"
    ODD_EXIT_WARNING = "ODD Exit Warning"
    REQUEST_TO_INTERVENE = "Request to Intervene"
    DRIVER_ENGAGED = "Driver Performing DDT"
    FALLBACK = "DDT Fallback"
    MINIMAL_RISK_CONDITION = "Minimal Risk Condition"


@dataclass
class ADSStateMachine:
    """A deterministic ADS engagement state machine (SAE J3016 Figs. 3-8, 13-14).

    Attributes:
        state: The current ADSState.
        sae_level: The SAE level this state machine represents (informational).
    """

    state: ADSState = ADSState.ENGAGED
    sae_level: int = 2

    def step(self, within_odd: bool, driver_responsive: bool = True) -> ADSState:
        """Advances the state machine by one observation and returns the new state.

        Args:
            within_odd: Whether the vehicle is currently within its ODD.
            driver_responsive: Whether the driver responds to a Request to
                Intervene in time (only consulted from REQUEST_TO_INTERVENE).

        Returns:
            The resulting ADSState.
        """
        if self.state == ADSState.ENGAGED:
            if not within_odd:
                self.state = ADSState.ODD_EXIT_WARNING

        elif self.state == ADSState.ODD_EXIT_WARNING:
            self.state = ADSState.ENGAGED if within_odd else ADSState.REQUEST_TO_INTERVENE

        elif self.state == ADSState.REQUEST_TO_INTERVENE:
            self.state = ADSState.DRIVER_ENGAGED if driver_responsive else ADSState.FALLBACK

        elif self.state == ADSState.FALLBACK:
            self.state = ADSState.MINIMAL_RISK_CONDITION

        elif self.state == ADSState.MINIMAL_RISK_CONDITION:
            pass  # Terminal until externally reset (e.g. vehicle stopped, driver resumes).

        elif self.state == ADSState.DRIVER_ENGAGED:
            if within_odd:
                self.state = ADSState.ENGAGED

        return self.state

    def reset(self) -> None:
        """Resets the state machine to ENGAGED (e.g. after a driver takeover completes)."""
        self.state = ADSState.ENGAGED


_ODD_MODE_TO_ADS_STATE = {
    "Normal": ADSState.ENGAGED,
    "Degraded": ADSState.ODD_EXIT_WARNING,
    "Takeover": ADSState.REQUEST_TO_INTERVENE,
}


def map_odd_mode_to_sae_state(odd_mode: str) -> ADSState:
    """Maps this project's Normal/Degraded/Takeover mode onto an SAE ADSState.

    Args:
        odd_mode: One of "Normal", "Degraded", "Takeover" (odd_classifier.py).

    Returns:
        The corresponding ADSState.
    """
    return _ODD_MODE_TO_ADS_STATE[odd_mode]


if __name__ == "__main__":
    level = classify_sae_level(THIS_PROJECT_DDT)
    print(f"This project's SAE level: {level} ({LEVEL_NAMES[level]})")
    assert level == 2, "Expected this capstone to classify as SAE Level 2 per the proposal."

    print("\nFull Table 1 decision tree sanity check:")
    print(f"  Level 0 (no control): {classify_sae_level(DynamicDrivingTask())}")
    print(
        f"  Level 1 (one axis, driver OEDR): "
        f"{classify_sae_level(DynamicDrivingTask(lateral_control=True))}"
    )
    print(
        f"  Level 3 (ADS OEDR, no auto-fallback): "
        f"{classify_sae_level(DynamicDrivingTask(lateral_control=True, longitudinal_control=True, oedr=True))}"
    )
    print(
        f"  Level 4 (ADS OEDR+fallback, limited ODD): "
        f"{classify_sae_level(DynamicDrivingTask(True, True, True, True, odd_unlimited=False))}"
    )
    print(
        f"  Level 5 (full DDT, unlimited ODD): "
        f"{classify_sae_level(DynamicDrivingTask(True, True, True, True, odd_unlimited=True))}"
    )

    print("\nADS state machine walkthrough:")
    machine = ADSStateMachine()
    for odd_mode in ["Normal", "Degraded", "Takeover", "Takeover", "Normal"]:
        target_state = map_odd_mode_to_sae_state(odd_mode)
        within_odd = target_state == ADSState.ENGAGED
        new_state = machine.step(within_odd=within_odd, driver_responsive=True)
        print(f"  odd_mode={odd_mode} -> state={new_state.value}")
