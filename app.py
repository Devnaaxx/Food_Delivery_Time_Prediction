
import streamlit as st
import pandas as pd
import pickle

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Food Delivery Predictor",
    page_icon="🍔",
    layout="wide"
)

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #101827;
    color: white;
}

h1 {
    color: #FFB347;
    text-align: center;
}

h2, h3 {

    color: #FFFFFF !important;
}
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] div,
.stNumberInput label,
.stSelectbox label,
.stSlider label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}
div.stButton > button {
    background-color: #FF8C42;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    background-color: #FF6B1A;
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(255, 140, 66, 0.45);
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #111827 !important;
}

</style>
""", unsafe_allow_html=True)
# ==========================================
# 3. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    with open("food_delivery_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    saved_data = load_model()
    model = saved_data["model"]
    categories = saved_data["categories"]

except FileNotFoundError:
    st.error("Model file not found. Train the model first.")
    st.stop()

# ==========================================
# 4. MAIN TITLE
# ==========================================

st.title("🍔 FOOD DELIVERY TIME PREDICTOR")

st.markdown(
    "<p style='text-align:center;'>"
    "Predict your food delivery time using Machine Learning"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

st.success(
    "Active Model: " + saved_data["model_name"]
)

# ==========================================
# 5. USER INPUT
# ==========================================

st.subheader("📋 Enter Delivery Details")

col1, col2 = st.columns(2)

with col1:

    st.subheader("👨‍🍳 Delivery Information")

    age = st.number_input(
        "Delivery Person Age",
        18, 60, 25
    )

    rating = st.number_input(
        "Delivery Person Rating",
        1.0, 5.0, 4.5, 0.1
    )

    vehicle = st.selectbox(
        "Vehicle Condition",
        [0, 1, 2, 3]
    )

    deliveries = st.selectbox(
        "Multiple Deliveries",
        [1, 2, 3, 4]
    )

    hour = st.slider(
        "Order Time",
        0, 23, 13
    )

    distance = st.number_input(
        "Distance (KM)",
        0.1, 100.0, 5.0, 0.5
    )

with col2:

    st.subheader("🌦️ Environment Information")

    def get_options(column):

        options = categories.get(column, [])

        if not options:
            return ["Unknown"]

        return options

    weather = st.selectbox(
        "Weather Conditions",
        get_options("Weatherconditions")
    )

    traffic = st.selectbox(
        "Road Traffic Density",
        get_options("Road_traffic_density")
    )

    order_type = st.selectbox(
        "Type of Order",
        get_options("Type_of_order")
    )

    vehicle_type = st.selectbox(
        "Type of Vehicle",
        get_options("Type_of_vehicle")
    )

    city = st.selectbox(
        "City",
        get_options("City")
    )

# ==========================================
# 6. PREDICTION
# ==========================================

st.divider()

if st.button("🚀 PREDICT DELIVERY TIME"):

    input_data = pd.DataFrame([{

        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Vehicle_condition": vehicle,
        "multiple_deliveries": deliveries,
        "order_hour": hour,
        "distance_km": distance,

        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "City": city

    }])

    input_data = input_data[saved_data["features"]]

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    minutes = round(prediction)

    # ======================================
    # 7. DISPLAY RESULT
    # ======================================


    st.divider()
    st.subheader("🎉 YOUR PREDICTION RESULT")

    col_res1, col_res2 = st.columns([2, 1])

    with col_res1:
        st.metric(
            "Estimated Delivery Time",
            f"{minutes} Minutes"
        )

    with col_res2:
        if minutes <= 25:
            st.metric("Delivery Vibe", "🚀 Rocket Speed")
        elif minutes <= 40:
            st.metric("Delivery Vibe", "🛵 Smooth Cruise")
        else:
            st.metric("Delivery Vibe", "🐢 Scenic Route")

    # Visual ETA Progress Bar
    progress_val = min(minutes / 60.0, 1.0)
    st.progress(progress_val, text=f"Kitchen-to-Doorstep Meter: {minutes} / 60 mins")

    # Playful conditional reactions
    if minutes <= 20:
        st.balloons()
        st.success(f"⚡ {minutes} mins?! The chef probably started cooking before you even clicked order!")
    elif minutes <= 30:
        st.balloons()
        st.success(f"🍕 Arriving in {minutes} minutes! Just enough time to pick a movie and grab plates.")
    elif minutes <= 45:
        st.info(f"🚴 Expected in {minutes} minutes. Your rider is dodging traffic like a pro!")
    else:
        st.snow()
        st.warning(f"⏳ {minutes} minutes! Traffic or weather is playing hard mode—maybe grab a small snack while you wait.")

    # Collapsible Order Receipt
    with st.expander("🧾 View Order & Route Receipt"):
        st.write(f"**📍 Distance:** {distance} km | **🏙️ City:** {city}")
        st.write(f"**🌦️ Weather:** {weather} | **🚦 Traffic:** {traffic}")
        st.write(f"**🛵 Rider Rating:** {rating} ⭐ ({vehicle_type})")
        st.write(f"**📦 Order Type:** {order_type} (Batch size: {deliveries})")

    st.caption(
        "This is an estimated time generated by a machine learning model."
    )
# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<p style='text-align:center;'>"
    "Made with ❤️ using Python, Scikit-learn and Streamlit"
    "</p>",
    unsafe_allow_html=True
)
