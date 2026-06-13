import streamlit as st
import joblib
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🫀",
    layout="wide"
)

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("heart_model.pkl")

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff5f5 0%, #ffe3e3 35%, #f8fbff 100%);
}

/* Main card */
.main-card {
    background: rgba(255, 255, 255, 0.92);
    padding: 35px;
    border-radius: 25px;
    box-shadow: 0px 10px 30px rgba(180, 0, 0, 0.15);
    margin-bottom: 25px;
}

/* Title */
.title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    color: #b00020;
}

.subtitle {
    text-align: center;
    font-size: 19px;
    color: #555;
    margin-bottom: 25px;
}

/* Hover effect */
div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
    transition: 0.3s;
}

div[data-testid="stNumberInput"]:hover, div[data-testid="stSelectbox"]:hover {
    transform: scale(1.02);
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #b00020, #ff4b5c);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 15px;
    padding: 12px 30px;
    border: none;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ff4b5c, #b00020);
    transform: scale(1.03);
}

/* Result box */
.result-box {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="main-card">
    <div class="title">🫀 AI Healthcare Risk Prediction Chatbot</div>
    <div class="subtitle">
        Advanced Machine Learning Based Heart Disease Risk Prediction System
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("🩺 About Project")
st.sidebar.info("""
This AI Healthcare Chatbot predicts heart disease risk using patient health parameters.

**Technologies Used:**
- Python
- Streamlit
- Machine Learning
- Scikit-learn
- Joblib
- Heart Disease Dataset
""")

st.sidebar.warning("""
Note: This system is for educational/project purpose only.
It does not replace medical advice.
""")

# ---------------------------
# Input Section
# ---------------------------
st.markdown("### 🧾 Enter Patient Medical Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40,
        help="Enter patient's age in years"
    )

    sex_label = st.selectbox(
        "Sex",
        ["Female", "Male"],
        help="Female = 0, Male = 1"
    )
    sex = 1 if sex_label == "Male" else 0

    cp_label = st.selectbox(
        "Chest Pain Type",
        [
            "0 - Typical Angina",
            "1 - Atypical Angina",
            "2 - Non-anginal Pain",
            "3 - Asymptomatic"
        ],
        help="Select type of chest pain"
    )
    cp = int(cp_label[0])

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120,
        help="Resting blood pressure in mm Hg"
    )

with col2:
    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=200,
        help="Serum cholesterol in mg/dl"
    )

    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"],
        help="No = 0, Yes = 1"
    )
    fbs = 1 if fbs_label == "Yes" else 0

    restecg_label = st.selectbox(
        "Resting ECG Result",
        [
            "0 - Normal",
            "1 - ST-T Wave Abnormality",
            "2 - Left Ventricular Hypertrophy"
        ],
        help="ECG result category"
    )
    restecg = int(restecg_label[0])

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150,
        help="Maximum heart rate achieved"
    )

with col3:
    exang_label = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"],
        help="No = 0, Yes = 1"
    )
    exang = 1 if exang_label == "Yes" else 0

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="ST depression induced by exercise"
    )

    slope_label = st.selectbox(
        "Slope",
        [
            "0 - Upsloping",
            "1 - Flat",
            "2 - Downsloping"
        ],
        help="Slope of peak exercise ST segment"
    )
    slope = int(slope_label[0])

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4],
        help="Number of major vessels colored by fluoroscopy"
    )

    thal_label = st.selectbox(
        "Thalassemia",
        [
            "0 - Unknown",
            "1 - Normal",
            "2 - Fixed Defect",
            "3 - Reversible Defect"
        ],
        help="Thalassemia category"
    )
    thal = int(thal_label[0])

# ---------------------------
# Prediction Button
# ---------------------------
st.write("")

if st.button("🔍 Predict Heart Disease Risk"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_percent = probability * 100

    st.markdown("### 📊 Prediction Result")
    st.progress(int(risk_percent))

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box">
            <h2 style="color:#b00020;">⚠ High Risk of Heart Disease</h2>
            <h3>Risk Probability: {risk_percent:.2f}%</h3>
            <p>Please consult a cardiologist for proper medical checkup.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-box">
            <h2 style="color:#008000;">✅ Low Risk of Heart Disease</h2>
            <h3>Risk Probability: {risk_percent:.2f}%</h3>
            <p>Maintain a healthy lifestyle, regular exercise, and balanced diet.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
<br>
<hr>
<center>
    <p style="color:gray;">
    Developed for Final Year Project | AI Healthcare Assistant
    </p>
</center>
""", unsafe_allow_html=True)