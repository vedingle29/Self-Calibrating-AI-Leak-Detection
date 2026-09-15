"""Synthetic leak diagnosis. Never connect this module to real valves."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


LABELS = ("Normal usage", "Leak", "Flow sensor drift")
SENSOR_SCALE = np.array([0.05, 0.10])


@dataclass(frozen=True)
class Scenario:
    demand: float = 10.0
    leak: float = 0.0
    sensor_bias: float = 0.0

    def __post_init__(self):
        values = (self.demand, self.leak, self.sensor_bias)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Scenario values must be finite.")
        if self.demand < 0 or self.leak < 0:
            raise ValueError("Demand and leak must be nonnegative.")


def predict(demand, leak, bias, valve):
    """Toy response model, not a hydraulic network solver.

    Normal consumption scales linearly with valve setting.
    Leakage scales with its square root.
    Flow sensor bias does not change physical pressure.
    """
    if not np.isfinite(valve) or not 0 < valve <= 1:
        raise ValueError("Valve setting must be in (0, 1].")

    physical_flow = (
        np.asarray(demand) * valve
        + np.asarray(leak) * np.sqrt(valve)
    )
    measured_flow = physical_flow + np.asarray(bias)
    pressure = 60.0 - 0.8 * physical_flow
    return np.stack((measured_flow, pressure), axis=-1)


def observe(scenario, valve=1.0, noise=True, rng=None):
    reading = predict(
        scenario.demand,
        scenario.leak,
        scenario.sensor_bias,
        valve,
    )
    if noise:
        rng = np.random.default_rng() if rng is None else rng
        reading = reading + rng.normal(0, SENSOR_SCALE)
    return reading


def train_detector(seed=42):
    """Learn passive normal-operation readings."""
    rng = np.random.default_rng(seed)
    demands = rng.uniform(8.0, 12.0, size=600)
    readings = predict(
        demands,
        np.zeros_like(demands),
        np.zeros_like(demands),
        1.0,
    )
    readings += rng.normal(0, SENSOR_SCALE, size=readings.shape)

    detector = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=seed,
    )
    detector.fit(readings)
    return detector


def candidate_models():
    """Finite hypothesis grid: one fault at a time."""
    candidates = []

    for demand in np.arange(4.0, 20.01, 0.5):
        candidates.append((0, demand, 0.0, 0.0))

        for leak in np.arange(0.5, 6.01, 0.5):
            candidates.append((1, demand, leak, 0.0))

        for bias in np.arange(-6.0, 6.01, 0.5):
            if abs(bias) > 0.01:
                candidates.append((2, demand, 0.0, bias))

    return np.asarray(candidates, dtype=float)


class Diagnoser:
    def __init__(self):
        self.models = candidate_models()

    def predictions(self, valve):
        return predict(
            self.models[:, 1],
            self.models[:, 2],
            self.models[:, 3],
            valve,
        )

    def errors(self, history):
        if not history:
            raise ValueError("At least one observation is required.")

        errors = np.zeros(len(self.models))
        for valve, reading in history:
            reading = np.asarray(reading, dtype=float)
            if reading.shape != (2,) or not np.all(np.isfinite(reading)):
                raise ValueError("Reading must contain finite flow and pressure.")
            residual = (self.predictions(valve) - reading) / SENSOR_SCALE
            errors += np.sum(residual**2, axis=1)

        return errors

    def best_per_label(self, errors):
        indices = []
        for label in range(len(LABELS)):
            members = np.flatnonzero(self.models[:, 0] == label)
            indices.append(int(members[np.argmin(errors[members])]))
        return indices

    def choose_test(
        self,
        history,
        valves=(0.70, 0.85, 0.95),
        pressure_bounds=(35.0, 59.0),
    ):
        """Choose the strongest predicted separation between hypotheses.

        Pressure bounds are checked against every grid model. This is a
        conservative simulation filter, not a hardware safety guarantee.
        """
        low, high = pressure_bounds
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("Pressure bounds must be finite and increasing.")

        errors = self.errors(history)
        representatives = self.best_per_label(errors)
        used = [valve for valve, _ in history]
        best = None

        for valve in valves:
            if any(np.isclose(valve, previous) for previous in used):
                continue

            predictions = self.predictions(valve)
            pressures = predictions[:, 1]
            if np.any(pressures < low) or np.any(pressures > high):
                continue

            normalized = predictions[representatives] / SENSOR_SCALE
            separation = float(np.var(normalized, axis=0).sum())

            if best is None or separation > best["separation"]:
                best = {
                    "valve": float(valve),
                    "separation": separation,
                }

        return best

    def diagnose(self, history):
        errors = self.errors(history)
        representatives = self.best_per_label(errors)
        class_errors = errors[representatives]
        winner = int(np.argmin(class_errors))
        model = self.models[representatives[winner]]

        # Relative fit weights, not calibrated probabilities.
        differences = class_errors - class_errors.min()
        weights = np.exp(-0.5 * np.minimum(differences, 1400))
        weights /= weights.sum()

        standardized_rmse = float(
            np.sqrt(class_errors[winner] / (2 * len(history)))
        )
        adequate = standardized_rmse <= 3.0
        ambiguous = float(weights.max()) < 0.80

        if not adequate:
            diagnosis = "Inconclusive: outside model assumptions"
        elif ambiguous:
            diagnosis = "Inconclusive: competing explanations"
        else:
            diagnosis = LABELS[winner]

        return {
            "diagnosis": diagnosis,
            "best_fit": LABELS[winner],
            "relative_fit": {
                label: float(weight)
                for label, weight in zip(LABELS, weights)
            },
            "standardized_rmse": standardized_rmse,
            "parameters": {
                "demand": float(model[1]),
                "leak": float(model[2]),
                "sensor_bias": float(model[3]),
            },
        }
