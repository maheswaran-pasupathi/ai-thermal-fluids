"""Stage 4 - interactive surrogate decision dashboard.

Run from this project's code directory with:
    streamlit run stage4_dashboard.py

The dashboard wraps the Stage 2 surrogate and Stage 3 bounded grid-search
optimizer into an engineering decision-support interface. It is intentionally
transparent about extrapolation: inputs outside the surrogate's trained range
are flagged rather than presented as trustworthy predictions.

Credit: ECO-Qube EU project (CORDIS 956059) and contributing partners - see
README.md for full citation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "merged_surrogate_dataset.csv"

FEATURES = [
    "airIRG2GroupStatusAirFlowMetric[L/s]",
    "airIRG2GroupStatusCoolOutput[KWh]",
    "airIRG2RDT2StatusEvaporatorFanSpeed[%]",
    "airIRG2RDT2StatusReturnAirTempMetric[C]",
    "airIRG2RDT2StatusSupplyAirTempMetric[C]",
]

TARGET = "max_rack_temp"

DISPLAY_NAMES = {
    "airIRG2GroupStatusAirFlowMetric[L/s]": "Air flow [L/s]",
    "airIRG2GroupStatusCoolOutput[KWh]": "Cooling output [kWh]",
    "airIRG2RDT2StatusEvaporatorFanSpeed[%]": "Evaporator fan speed [%]",
    "airIRG2RDT2StatusReturnAirTempMetric[C]": "Return air temperature [°C]",
    "airIRG2RDT2StatusSupplyAirTempMetric[C]": "Supply air temperature [°C]",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(data: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=0,
    )
    model.fit(data[FEATURES], data[TARGET])
    return model


def feature_ranges(data: pd.DataFrame) -> dict:
    return {
        feature: (
            float(data[feature].quantile(0.02)),
            float(data[feature].quantile(0.98)),
        )
        for feature in FEATURES
    }


def predict_hotspot(
    model: RandomForestRegressor,
    ranges: dict,
    settings: dict,
    safe_limit_c: float = 33.8,
) -> dict:
    """Predict the maximum rack temperature and explicitly flag extrapolation."""
    row = pd.DataFrame([settings])[FEATURES]
    out_of_range = [
        feature
        for feature in FEATURES
        if not (ranges[feature][0] <= settings[feature] <= ranges[feature][1])
    ]

    prediction = float(model.predict(row)[0])

    return {
        "predicted_max_temp_C": round(prediction, 2),
        "within_safe_limit": bool(prediction <= safe_limit_c),
        "out_of_trained_range": out_of_range,
    }


def recommend_setting(
    model: RandomForestRegressor,
    ranges: dict,
    safe_limit_c: float = 33.8,
    n_levels: int = 8,
) -> dict:
    """Find the minimum cooling-power-proxy setting within the trained range."""
    grid = {
        feature: np.linspace(ranges[feature][0], ranges[feature][1], n_levels)
        for feature in FEATURES
    }

    combinations = pd.MultiIndex.from_product(
        grid.values(), names=FEATURES
    ).to_frame(index=False)

    combinations["predicted_max_temp"] = model.predict(combinations[FEATURES])
    combinations["cooling_power_proxy"] = (
        combinations["airIRG2GroupStatusCoolOutput[KWh]"]
        + combinations["airIRG2RDT2StatusEvaporatorFanSpeed[%]"] / 100.0
    )

    feasible = combinations[
        combinations["predicted_max_temp"] <= safe_limit_c
    ].copy()

    if feasible.empty:
        return {"feasible": False}

    best = feasible.loc[feasible["cooling_power_proxy"].idxmin()]
    return {"feasible": True, **best.to_dict()}


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Data Center Cooling Surrogate",
    page_icon=None,
    layout="wide",
)

st.title("Data Center Cooling Surrogate")
st.caption(
    "Random-Forest surrogate + bounded optimization for hotspot temperature "
    "prediction and cooling-setting recommendation."
)

st.info(
    "Engineering rule used in this dashboard: predictions are only considered "
    "credible inside the surrogate's observed training range. Out-of-range "
    "inputs are explicitly flagged."
)


data = load_data()
model = train_model(data)
ranges = feature_ranges(data)

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------

st.sidebar.header("Safety constraint")
safe_limit = st.sidebar.number_input(
    "Maximum allowed rack temperature [°C]",
    min_value=30.0,
    max_value=40.0,
    value=33.8,
    step=0.1,
)

st.sidebar.header("Surrogate validity")
st.sidebar.write(
    "Slider limits correspond to the 2nd-98th percentile range used as the "
    "surrogate's bounded operating envelope."
)

# -----------------------------------------------------------------------------
# Main tabs
# -----------------------------------------------------------------------------

predict_tab, optimize_tab, model_tab = st.tabs(
    [
        "Predict hotspot",
        "Recommend cooling setting",
        "Model / data context",
    ]
)


with predict_tab:
    st.subheader("1. Predict hotspot temperature from cooling settings")
    st.write(
        "Adjust the operating conditions below. The surrogate predicts the "
        "maximum rack temperature and checks it against the safety constraint."
    )

    input_columns = st.columns(2)
    user_settings = {}

    for i, feature in enumerate(FEATURES):
        low, high = ranges[feature]
        median = float(data[feature].median())
        span = max(high - low, 1e-6)
        step = span / 100.0

        with input_columns[i % 2]:
            user_settings[feature] = st.slider(
                DISPLAY_NAMES[feature],
                min_value=float(low),
                max_value=float(high),
                value=float(np.clip(median, low, high)),
                step=float(step),
                format="%.2f",
            )

    result = predict_hotspot(
        model=model,
        ranges=ranges,
        settings=user_settings,
        safe_limit_c=safe_limit,
    )

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric(
        "Predicted max rack temperature",
        f"{result['predicted_max_temp_C']:.2f} °C",
    )
    metric_b.metric(
        "Safety limit",
        f"{safe_limit:.2f} °C",
    )
    margin = safe_limit - result["predicted_max_temp_C"]
    metric_c.metric(
        "Temperature margin",
        f"{margin:+.2f} °C",
    )

    if result["within_safe_limit"]:
        st.success("Prediction is within the selected temperature safety limit.")
    else:
        st.error("Prediction exceeds the selected temperature safety limit.")

    if result["out_of_trained_range"]:
        names = [DISPLAY_NAMES[f] for f in result["out_of_trained_range"]]
        st.warning(
            "Out-of-range input detected. Treat this prediction as extrapolation: "
            + ", ".join(names)
        )
    else:
        st.caption("All selected inputs are inside the bounded surrogate range.")

    comparison = pd.DataFrame(
        {
            "Temperature [°C]": [
                result["predicted_max_temp_C"],
                safe_limit,
            ]
        },
        index=["Predicted hotspot", "Safety limit"],
    )
    st.bar_chart(comparison)


with optimize_tab:
    st.subheader("2. Recommend a lower-cooling feasible operating point")
    st.write(
        "The optimizer searches only within the surrogate's trained range and "
        "selects the feasible point with the lowest cooling-power proxy."
    )

    n_levels = st.slider(
        "Grid-search resolution per feature",
        min_value=4,
        max_value=10,
        value=8,
        step=1,
        help="Higher values test more combinations but require more computation.",
    )

    if st.button("Find recommended setting", type="primary"):
        recommendation = recommend_setting(
            model=model,
            ranges=ranges,
            safe_limit_c=safe_limit,
            n_levels=n_levels,
        )

        if not recommendation["feasible"]:
            st.error(
                "No feasible point was found inside the bounded surrogate range "
                "for this safety constraint."
            )
        else:
            st.success(
                "Feasible recommendation found inside the surrogate's trained range."
            )

            kpi_1, kpi_2 = st.columns(2)
            kpi_1.metric(
                "Predicted max rack temperature",
                f"{recommendation['predicted_max_temp']:.2f} °C",
            )
            kpi_2.metric(
                "Cooling-power proxy",
                f"{recommendation['cooling_power_proxy']:.3f}",
            )

            recommendation_table = pd.DataFrame(
                {
                    "Variable": [DISPLAY_NAMES[f] for f in FEATURES],
                    "Recommended value": [recommendation[f] for f in FEATURES],
                    "Training-range minimum": [ranges[f][0] for f in FEATURES],
                    "Training-range maximum": [ranges[f][1] for f in FEATURES],
                }
            )

            st.dataframe(
                recommendation_table,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Cooling-power proxy = cooling output + fan speed / 100. "
                "It is a portfolio exercise metric, not a validated energy-cost model."
            )


with model_tab:
    st.subheader("3. Model and data context")

    left, right = st.columns(2)

    with left:
        st.markdown("**Surrogate**")
        st.write("RandomForestRegressor")
        st.write("200 trees, max depth 6")
        st.write(f"Training rows: {len(data):,}")
        st.write(f"Target: {TARGET}")

    with right:
        st.markdown("**Engineering interpretation**")
        st.write(
            "This model is intended as a fast thermal KPI surrogate for exploring "
            "cooling settings, not as a replacement for CFD or experimental validation."
        )
        st.write(
            "The available logged target range is narrow, so confidence should be "
            "limited to nearby operating conditions represented in the dataset."
        )

    ranges_table = pd.DataFrame(
        {
            "Feature": [DISPLAY_NAMES[f] for f in FEATURES],
            "2nd percentile": [ranges[f][0] for f in FEATURES],
            "Median": [float(data[f].median()) for f in FEATURES],
            "98th percentile": [ranges[f][1] for f in FEATURES],
        }
    )
    st.dataframe(ranges_table, use_container_width=True, hide_index=True)

    st.markdown("**Target distribution**")
    st.line_chart(data[TARGET].reset_index(drop=True))


st.divider()
st.caption(
    "Independent learning exercise using public ECO-Qube data. "
    "No proprietary or employer data is used."
)
