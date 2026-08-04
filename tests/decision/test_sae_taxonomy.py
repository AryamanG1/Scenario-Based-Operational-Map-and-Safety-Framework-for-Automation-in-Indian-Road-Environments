"""Unit tests for sae_taxonomy.py's SAE J3016 level classifier and state machine."""

import pytest

from src.decision.sae_taxonomy import (
    THIS_PROJECT_DDT,
    ADSState,
    ADSStateMachine,
    DynamicDrivingTask,
    classify_sae_level,
    map_odd_mode_to_sae_state,
)


@pytest.mark.parametrize(
    "ddt,expected_level",
    [
        (DynamicDrivingTask(), 0),
        (DynamicDrivingTask(lateral_control=True), 1),
        (DynamicDrivingTask(longitudinal_control=True), 1),
        (DynamicDrivingTask(lateral_control=True, longitudinal_control=True), 2),
        (
            DynamicDrivingTask(lateral_control=True, longitudinal_control=True, oedr=True),
            3,
        ),
        (
            DynamicDrivingTask(
                lateral_control=True, longitudinal_control=True, oedr=True,
                ddt_fallback=True, odd_unlimited=False,
            ),
            4,
        ),
        (
            DynamicDrivingTask(
                lateral_control=True, longitudinal_control=True, oedr=True,
                ddt_fallback=True, odd_unlimited=True,
            ),
            5,
        ),
    ],
)
def test_classify_sae_level_table_1(ddt, expected_level):
    assert classify_sae_level(ddt) == expected_level


def test_this_project_is_level_2():
    # The capstone proposal explicitly commits to SAE Level 2 -- a prior
    # draft (the one-shot prompt) incorrectly targeted Level 3.
    assert classify_sae_level(THIS_PROJECT_DDT) == 2


def test_state_machine_engaged_to_odd_exit_warning():
    machine = ADSStateMachine()
    assert machine.state == ADSState.ENGAGED
    new_state = machine.step(within_odd=False)
    assert new_state == ADSState.ODD_EXIT_WARNING


def test_state_machine_full_takeover_sequence_to_fallback():
    machine = ADSStateMachine()
    machine.step(within_odd=False)  # ENGAGED -> ODD_EXIT_WARNING
    machine.step(within_odd=False)  # -> REQUEST_TO_INTERVENE
    final_state = machine.step(within_odd=False, driver_responsive=False)  # -> FALLBACK
    assert final_state == ADSState.FALLBACK
    mrc_state = machine.step(within_odd=False)  # -> MINIMAL_RISK_CONDITION
    assert mrc_state == ADSState.MINIMAL_RISK_CONDITION


def test_state_machine_recovers_to_engaged_when_odd_returns():
    machine = ADSStateMachine()
    machine.step(within_odd=False)  # -> ODD_EXIT_WARNING
    recovered = machine.step(within_odd=True)  # -> ENGAGED
    assert recovered == ADSState.ENGAGED


def test_state_machine_reset():
    machine = ADSStateMachine()
    machine.step(within_odd=False)
    machine.reset()
    assert machine.state == ADSState.ENGAGED


def test_map_odd_mode_to_sae_state():
    assert map_odd_mode_to_sae_state("Normal") == ADSState.ENGAGED
    assert map_odd_mode_to_sae_state("Degraded") == ADSState.ODD_EXIT_WARNING
    assert map_odd_mode_to_sae_state("Takeover") == ADSState.REQUEST_TO_INTERVENE
