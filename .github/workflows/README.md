# AI Leak Lab

A Python research prototype that compares passive anomaly detection
with active simulated diagnosis of water-system anomalies.

**Simulation only. Do not connect this software to physical valves.**

## Features

- Synthetic flow and pressure readings.
- Isolation Forest trained on normal operating data.
- Competing explanations: normal usage, leak, or flow sensor drift.
- Simulated valve-test selection based on predicted response separation.
- Pressure-bound filtering across the candidate model grid.
- Streamlit dashboard and automated tests.
- Inconclusive results when explanations remain ambiguous or fit poorly.

## Requirements

Python 3.11 or 3.12.

## Setup

```bash
python -m venv .venv
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run app.py
```

Open the local address printed by Streamlit.

## Test

```bash
python -m pytest -q
```

GitHub Actions also runs the tests on pushes and pull requests.

## How it works

1. Generate a passive reading with the simulated valve fully open.
2. Use Isolation Forest to flag unusual passive readings.
3. Fit candidate normal-usage, leak, and sensor-drift explanations.
4. Select a simulated valve setting that separates their predicted
   responses while meeting the configured simulation pressure bounds.
5. Collect a synthetic response and update the diagnosis.
6. Repeat for one additional diagnostic test.

Active tests run independently of the anomaly flag for demonstration.

### Toy equations

For valve setting v, demand d, leakage magnitude L, and sensor bias b:

    physical_flow = d * v + L * sqrt(v)
    measured_flow = physical_flow + b
    pressure = 60 - 0.8 * physical_flow

Flow and pressure units are illustrative.

The intentionally different valve responses let active measurements
separate leaks from normal consumption that looks identical passively.
These equations are not a validated hydraulic model.

## Limitations

- The simulator and diagnoser use the same equations, making this an
  optimistic demonstration rather than independent validation.
- Only one fault type is modeled at a time.
- Demand is assumed constant during diagnostic tests.
- Pressure sensor faults and transient pipe dynamics are not modeled.
- Diagnosis uses a finite grid of candidate parameter values.
- Relative fit weights are not calibrated probabilities.
- Test selection compares the best candidate from each explanation;
  it is a heuristic, not a proven optimal information-gain policy.
- Pressure filtering does not establish physical safety.
- Included tests check deterministic behavior, not real-world accuracy.
- Dependency ranges are not a reproducible lockfile.

## Research evaluation

Before making performance claims, evaluate on independently generated
or measured data, including unseen demand patterns, varying noise,
simultaneous faults, and model mismatch.

Compare passive-only and active diagnosis using:

- Per-class precision and recall.
- False-alarm rate.
- Frequency of inconclusive diagnoses.
- Number of diagnostic tests.
- Simulated control disturbance.

## Patent and disclosure notice

This repository does not establish novelty or patentability. Active
diagnosis and leak detection have existing prior art.

A possible research direction is a new uncertainty-aware diagnostic
test-selection method with demonstrated technical benefits.

If patent protection matters, seek qualified advice before making the
repository public. Public disclosure can affect patent rights.
