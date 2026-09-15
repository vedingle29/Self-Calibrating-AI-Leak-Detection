import numpy as np
import streamlit as st

from leak_detection import (
    Diagnoser,
    Scenario,
    observe,
    train_detector,
)


st.set_page_config(page_title="AI Leak Lab", page_icon="💧", layout="wide")

st.title("💧 AI Leak Lab")
st.write(
    "Compare passive anomaly detection with active, simulated diagnosis."
)
st.warning(
    "Simulation only. No real valve control. "
    "Results depend on simplified model assumptions."
)


@st.cache_resource
def get_detector():
    return train_detector()


with st.sidebar:
    st.header("Synthetic scenario")
    kind = st.selectbox(
        "Scenario",
        ["Normal usage", "Leak", "Flow sensor drift"],
    )
    demand = st.slider("Underlying demand", 8.0, 12.0, 10.0, 0.5)

    leak = 0.0
    bias = 0.0

    if kind == "Leak":
        leak = st.slider("Leak magnitude", 0.5, 6.0, 4.0, 0.5)
    elif kind == "Flow sensor drift":
        bias = st.select_slider(
            "Flow sensor offset",
            options=[
                value / 2
                for value in range(-12, 13)
                if value != 0
            ],
            value=4.0,
        )

    noise = st.checkbox("Add sensor noise", value=True)
    seed = st.number_input(
        "Random seed", min_value=0, max_value=1_000_000, value=42
    )
    run = st.button("Run simulated diagnosis", type="primary")

st.caption(
    "Units are illustrative: flow in L/min and pressure in kPa. "
    "Valve setting is a dimensionless toy-model control."
)

if run:
    rng = np.random.default_rng(int(seed))
    scenario = Scenario(demand=demand, leak=leak, sensor_bias=bias)
    diagnoser = Diagnoser()

    initial = observe(scenario, valve=1.0, noise=noise, rng=rng)
    history = [(1.0, initial)]

    detector = get_detector()
    anomalous = detector.predict(initial.reshape(1, -1))[0] == -1

    st.subheader("Passive observation")
    left, middle, right = st.columns(3)
    left.metric("Measured flow", f"{initial[0]:.2f} L/min")
    middle.metric("Measured pressure", f"{initial[1]:.2f} kPa")
    right.metric("Anomaly detector", "Flagged" if anomalous else "Not flagged")

    passive = diagnoser.diagnose(history)
    st.write(f"Passive diagnosis: **{passive['diagnosis']}**")

    st.subheader("Active simulated tests")
    st.caption(
        "Tests run even if the anomaly detector does not flag the reading, "
        "so you can compare passive and active diagnosis."
    )

    for step in range(2):
        selected = diagnoser.choose_test(history)
        if selected is None:
            st.info("No unused test satisfies the simulation pressure bounds.")
            break

        valve = selected["valve"]
        reading = observe(scenario, valve=valve, noise=noise, rng=rng)
        history.append((valve, reading))

        st.write(
            f"Test {step + 1}: valve setting **{valve:.0%}**, "
            f"predicted separation score **{selected['separation']:.1f}**"
        )

    result = diagnoser.diagnose(history)

    st.subheader("Result")
    st.write(f"**{result['diagnosis']}**")
    st.caption(
        "Relative fit weights compare the three modeled explanations. "
        "They are not calibrated confidence probabilities."
    )

    st.dataframe(
        [
            {"Explanation": label, "Relative fit weight": weight}
            for label, weight in result["relative_fit"].items()
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Observation history")
    st.dataframe(
        [
            {
                "Valve setting": valve,
                "Flow (L/min)": float(reading[0]),
                "Pressure (kPa)": float(reading[1]),
            }
            for valve, reading in history
        ],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Model diagnostics"):
        st.json(result)
        st.write("Synthetic ground truth:")
        st.json(
            {
                "scenario": kind,
                "demand": demand,
                "leak": leak,
                "sensor_bias": bias,
            }
        )
else:
    st.info("Choose a scenario and click Run simulated diagnosis.")
