import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, date, time

# ---------- Load model ----------
model = joblib.load("delay_classifier.pkl")

# ---------- Class labels ----------
CLASS_LABELS = {
    0: "On-time (0–2 min)",
    1: "Slight delay (2–5 min)",
    2: "Major delay (5+ min)"
}

# ---------- Representative delay (minutes) ----------
REP_DELAY_MIN = {
    0: 1.0,
    1: 3.5,
    2: 8.0
}

# ---------- Train Metadata ----------
TRAIN_CONFIG = {
    "Nightjet": {"company": "NS Int", "numbers": [420, 402, 421, 403]},
    "Intercity": {"company": "NS", "numbers": [1410, 1409, 1414, 1413, 1714, 1716, 1718, 2217]},
    "Sprinter": {"company": "NS", "numbers": [5108, 4117, 4308, 5808, 5612, 4019, 6116, 4610]},
    "Stoptrein": {"company": "Arriva", "numbers": [37812, 32210, 37807, 37607, 20352, 8020, 20350, 30906]},
    "Intercity direct": {"company": "NS", "numbers": [1817, 1819, 1821, 1814, 1812, 2415, 1816, 1823]},
    "ICE": {"company": "DB", "numbers": [222, 121, 220, 225, 128, 123]},
    "EuroCity": {"company": "EuroCity", "numbers": [9211, 9215, 9212, 9219, 9216, 9223]},
    "Eurocity Direct": {"company": "NS", "numbers": [9512, 9523, 9516, 9520, 9527, 9524]},
    "Eurostar": {"company": "Eurostar", "numbers": [9301, 9310, 9106, 9303, 9115, 9316]},
    "European Sleeper": {"company": "European Sleeper", "numbers": [452, 453]},
    "Nachttrein": {"company": "NS Int", "numbers": [32748, 32786, 32749, 32787]}
}

# ---------- Station metadata ----------
STATIONS = {
    "AMS": "Amsterdam Centraal",
    "UT":  "Utrecht Centraal",
    "RDM": "Rotterdam Centraal",
    "DH":  "Den Haag Centraal",
    "EHV": "Eindhoven",
    "HLM": "Haarlem",
}

PLATFORMS = ["1", "2", "3", "4", "5", "6", "7", "8"]

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Train Delay Prediction",
    page_icon="🚆",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .main { background: #020617; }
    .hero {
        padding: 22px 26px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1d2440);
        border: 1px solid #1f2937;
        box-shadow: 0 18px 40px rgba(15,23,42,0.9);
        margin-bottom: 16px;
    }
    .hero-title { font-size: 34px; font-weight: 800; }
    .hero-sub { font-size: 14px; color: #9ca3af; }
    .card {
        background-color: #020617;
        border-radius: 16px;
        padding: 18px 22px;
        border: 1px solid #1f2937;
        box-shadow: 0 10px 30px rgba(15,23,42,0.8);
    }
    .metric-label { font-size: 13px; color: #9ca3af; }
    .metric-value { font-size: 26px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HERO SECTION ----------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🚆 Train Delay Prediction</div>
        <div class="hero-sub">Estimate the expected arrival delay in minutes.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- SESSION STATE ----------
if "selected_train_name" not in st.session_state:
    st.session_state.selected_train_name = None

def update_train_name():
    st.session_state.selected_train_name = st.session_state.train_name
    st.session_state.train_number = TRAIN_CONFIG[st.session_state.train_name]["numbers"][0]

# ---------- LAYOUT ----------
col_inputs, col_outputs = st.columns([1.1, 1.1])

# ===================== INPUTS =====================
with col_inputs:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎛 Input configuration")

    train_name = st.selectbox(
        "Train name",
        list(TRAIN_CONFIG.keys()),
        key="train_name",
        on_change=update_train_name
    )

    train_number = st.selectbox(
        "Train number",
        TRAIN_CONFIG[train_name]["numbers"],
        key="train_number"
    )

    with st.form("input_form"):

        c3, c4 = st.columns(2)

        with c3:
            station_display = [f"{code} – {name}" for code, name in STATIONS.items()]
            station_choice = st.selectbox("Station", station_display)
            station_code = station_choice.split(" – ")[0]

        with c4:
            platform = st.selectbox("Platform", PLATFORMS, index=4)

        st.markdown("#### Arrival details")

        c5, c6 = st.columns(2)
        with c5:
            arrival_date = st.date_input("Planned arrival date", value=date.today())
        with c6:
            arrival_time = st.time_input("Planned arrival time", value=time(8, 0))

        submitted = st.form_submit_button("Predict delay")

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== OUTPUT =====================
with col_outputs:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Prediction summary")

    if submitted:

        arr_dt = datetime.combine(arrival_date, arrival_time)
        arr_hour = arr_dt.hour
        arr_dayofweek = arr_dt.weekday()
        arr_month = arr_dt.month
        is_weekend = 1 if arr_dayofweek >= 5 else 0

        input_df = pd.DataFrame([{
            "service_train_number": int(train_number),
            "service_maximum_delay": 0.0,
            "arr_hour": arr_hour,
            "arr_dayofweek": arr_dayofweek,
            "arr_month": arr_month,
            "is_weekend": is_weekend,
            "service_type": train_name,
            "service_company": TRAIN_CONFIG[train_name]["company"],
            "stop_station_code": station_code,
            "stop_platform_change": False,
            "stop_planned_platform": platform,
            "stop_actual_platform": platform,
            "stop_order": 3,
            "prev_stop_delay": 0.0,
        }])

        proba = model.predict_proba(input_df)[0]

        # ---------- Expected delay ----------
        expected_delay = (
            proba[0] * REP_DELAY_MIN[0] +
            proba[1] * REP_DELAY_MIN[1] +
            proba[2] * REP_DELAY_MIN[2]
        )

        # 🔥 ---------- FIXED CATEGORY LOGIC ----------
        if expected_delay <= 2:
            class_label = "On-time (0–2 min)"
        elif expected_delay <= 5:
            class_label = "Slight delay (2–5 min)"
        else:
            class_label = "Major delay (5+ min)"
        # 🔥 ------------------------------------------

        # ---------- Display ----------
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("<div class='metric-label'>Estimated delay</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-value'>{expected_delay:.2f} min</div>",
                unsafe_allow_html=True
            )

        with m2:
            st.markdown("<div class='metric-label'>Predicted category</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-value'>{class_label}</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown("#### Delay category probabilities")
        prob_df = pd.DataFrame({
            "Delay category": [
                "On-time (0–2 min)",
                "Slight delay (2–5 min)",
                "Major delay (5+ min)"
            ],
            "Probability": [
                round(float(proba[0]), 3),
                round(float(proba[1]), 3),
                round(float(proba[2]), 3)
            ]
        })

        st.table(prob_df)

        st.caption(
            "Probabilities show model confidence for each delay range."
        )

    else:
        st.info("Fill inputs on the left and click **Predict delay**.")

    st.markdown("</div>", unsafe_allow_html=True)
