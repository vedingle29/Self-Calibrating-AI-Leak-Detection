import numpy as np
import pytest

from leak_detection import Diagnoser, Scenario, observe, train_detector


@pytest.mark.parametrize(
    "scenario, expected",
    [
        (Scenario(demand=10), "Normal usage"),
        (Scenario(demand=12), "Normal usage"),
        (Scenario(demand=10, leak=4), "Leak"),
        (Scenario(demand=10, sensor_bias=4), "Flow sensor drift"),
        (Scenario(demand=10, sensor_bias=-4), "Flow sensor drift"),
    ],
)
def test_active_diagnosis_without_noise(scenario, expected):
    diagnoser = Diagnoser()
    history = [(1.0, observe(scenario, noise=False))]

    for _ in range(2):
        selected = diagnoser.choose_test(history)
        assert selected is not None
        valve = selected["valve"]
        history.append((valve, observe(scenario, valve, noise=False)))

    result = diagnoser.diagnose(history)

    assert result["diagnosis"] == expected
    assert sum(result["relative_fit"].values()) == pytest.approx(1.0)
    assert result["standardized_rmse"] == pytest.approx(0.0)


def test_passive_leak_and_usage_can_look_identical():
    leak = observe(Scenario(demand=10, leak=4), noise=False)
    usage = observe(Scenario(demand=14), noise=False)
    np.testing.assert_allclose(leak, usage)

    result = Diagnoser().diagnose([(1.0, leak)])
    assert result["diagnosis"].startswith("Inconclusive")


def test_no_test_when_pressure_bounds_cannot_be_met():
    diagnoser = Diagnoser()
    history = [(1.0, observe(Scenario(), noise=False))]
    assert diagnoser.choose_test(
        history, pressure_bounds=(100.0, 110.0)
    ) is None


def test_detector_interface():
    detector = train_detector()
    reading = observe(Scenario(), noise=False).reshape(1, -1)
    assert detector.predict(reading)[0] in (-1, 1)


def test_reject_invalid_inputs():
    with pytest.raises(ValueError):
        Scenario(leak=-1)

    with pytest.raises(ValueError):
        observe(Scenario(), valve=0)

    with pytest.raises(ValueError):
        Diagnoser().diagnose([])

    with pytest.raises(ValueError):
        Diagnoser().diagnose([(1.0, [np.nan, 50.0])])
