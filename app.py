from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from maintenance_scheduler import schedule_maintenance_tasks


st.set_page_config(page_title="Railway Asset Risk Predictor", page_icon="🚆")

MODEL_PATH = Path(__file__).parent / "models" / "random_forest.joblib"
ASSET_TYPES = [
    "Track and rail", "Turnout / points", "Signalling equipment",
    "Level crossing", "Overhead line / electrification",
    "Traction power substation", "Locomotive", "Passenger coach",
    "Freight wagon", "Bridge", "Tunnel", "Platform or station equipment",
    "Telecommunications equipment",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🚆 Railway Asset Risk Predictor")
st.write("Estimate maintenance risk using a synthetic railway-asset demonstration dataset.")
st.warning("Demo only: this model uses artificial data and must not support real safety decisions.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run the command below once from the project folder.")
    st.code("python train_model.py")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error("The model could not be loaded. Re-run: python train_model.py")
    st.exception(error)
    st.stop()

with st.form("prediction_form"):
    asset_type = st.selectbox("Railway asset type", ASSET_TYPES)
    left, right = st.columns(2)
    with left:
        asset_age = st.number_input("Asset age (years)", min_value=0, value=14, step=1)
        previous_failures = st.number_input("Previous failures", min_value=0, value=3, step=1)
        days_since_maintenance = st.number_input(
            "Days since maintenance", min_value=0, value=160, step=1
        )
    with right:
        usage = st.number_input("Usage (%)", min_value=0, max_value=100, value=85, step=1)
        condition = st.number_input(
            "Condition score (0 = poor, 100 = excellent)",
            min_value=0, max_value=100, value=40, step=1,
        )
        criticality = st.number_input(
            "Criticality (1 = lowest, 5 = highest)",
            min_value=1, max_value=5, value=5, step=1,
        )
    submitted = st.form_submit_button("Predict risk", type="primary")

if submitted:
    asset = pd.DataFrame(
        [[asset_type, asset_age, previous_failures, days_since_maintenance, usage, condition, criticality]],
        columns=[
            "asset_type", "asset_age", "previous_failures", "days_since_maintenance",
            "usage", "condition", "criticality",
        ],
    )
    prediction = model.predict(asset)[0]
    probabilities = model.predict_proba(asset)[0]
    risk_probability = probabilities[list(model.classes_).index(1)]

    st.subheader("Prediction")
    if prediction == 1:
        st.error("HIGH RISK — prioritize inspection or maintenance.")
    else:
        st.success("LOW RISK — normal monitoring is appropriate.")
    st.metric("Estimated high-risk probability", f"{risk_probability:.1%}")
    st.progress(float(risk_probability))
    with st.expander("Submitted asset data"):
        st.dataframe(asset, use_container_width=True, hide_index=True)


st.divider()
st.header("🗓️ Maintenance Task Planner")
st.write(
    "Add multiple maintenance tasks, then use OR-Tools to appoint each task to a "
    "team and create a priority-aware schedule. Days are relative to the plan start."
)

default_tasks = pd.DataFrame(
    [
        {
            "task_name": "Track inspection", "asset_type": "Track and rail",
            "duration_days": 2, "priority": 5, "due_day": 3,
        },
        {
            "task_name": "Signal calibration", "asset_type": "Signalling equipment",
            "duration_days": 1, "priority": 4, "due_day": 2,
        },
        {
            "task_name": "Coach service", "asset_type": "Passenger coach",
            "duration_days": 3, "priority": 3, "due_day": 6,
        },
    ]
)
tasks_editor = st.data_editor(
    default_tasks,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "task_name": st.column_config.TextColumn("Task name", required=True),
        "asset_type": st.column_config.SelectboxColumn("Asset type", options=ASSET_TYPES, required=True),
        "duration_days": st.column_config.NumberColumn("Duration (days)", min_value=1, step=1, required=True),
        "priority": st.column_config.NumberColumn("Priority (1-5)", min_value=1, max_value=5, step=1, required=True),
        "due_day": st.column_config.NumberColumn("Due by day", min_value=1, step=1, required=True),
    },
    key="maintenance_tasks",
)

team_count = st.number_input("Available maintenance teams", min_value=1, max_value=20, value=2, step=1)
if st.button("Optimize task appointments and schedule", type="primary"):
    tasks = tasks_editor.dropna().to_dict("records")
    if not tasks:
        st.error("Add at least one complete maintenance task.")
    else:
        try:
            schedule = schedule_maintenance_tasks(tasks, int(team_count))
            schedule_df = pd.DataFrame(schedule)
            st.success("Schedule created. Each task has been appointed to a maintenance team.")
            st.dataframe(
                schedule_df[
                    ["task_name", "asset_type", "assigned_team", "start_day", "end_day", "late_days", "priority"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download schedule as CSV",
                data=schedule_df.to_csv(index=False),
                file_name="maintenance_schedule.csv",
                mime="text/csv",
            )
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
