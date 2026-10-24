import streamlit as st
import joblib
import numpy as np
from pathlib import Path

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
# (model file is loaded from the SAME folder as this .py file,
#  so the app works no matter which folder you run it from)
# ---------------------------
MODEL_PATH = Path(__file__).parent / "heart_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "heart_model.pkl file was not found next to this app file. "
        "Please keep heart_model.pkl and app_chatbot.py in the same folder."
    )
    st.stop()

# ---------------------------
# Custom CSS (same style as the original app)
# ---------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff5f5 0%, #ffe3e3 35%, #f8fbff 100%);
}

.main-card {
    background: rgba(255, 255, 255, 0.92);
    padding: 35px;
    border-radius: 25px;
    box-shadow: 0px 10px 30px rgba(180, 0, 0, 0.15);
    margin-bottom: 25px;
}

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

.stButton > button {
    background: linear-gradient(90deg, #b00020, #ff4b5c);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 15px;
    padding: 10px 28px;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ff4b5c, #b00020);
    transform: scale(1.03);
}

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
# Questions
# Same 13 inputs, in the SAME order, with the SAME values
# as the original app / the trained model expects.
# kind = "number" (whole number), "decimal" (number with .),
#        "choice" (pick one option)
# ---------------------------
QUESTIONS = [
    {
        "key": "age",
        "question": "Hello! I will ask you 13 questions, one by one. "
                    "First — what is the patient's **age** (in years)?",
        "kind": "number", "min": 1, "max": 120, "default": 40,
    },
    {
        "key": "sex",
        "question": "What is the patient's **sex**?",
        "kind": "choice",
        "options": [("Female", 0), ("Male", 1)],
    },
    {
        "key": "cp",
        "question": "What type of **chest pain** does the patient have?",
        "kind": "choice",
        "options": [
            ("0 - Typical Angina", 0),
            ("1 - Atypical Angina", 1),
            ("2 - Non-anginal Pain", 2),
            ("3 - Asymptomatic", 3),
        ],
    },
    {
        "key": "trestbps",
        "question": "What is the patient's **resting blood pressure** "
                    "(in mm Hg)?",
        "kind": "number", "min": 80, "max": 250, "default": 120,
    },
    {
        "key": "chol",
        "question": "What is the patient's **cholesterol** level "
                    "(in mg/dl)?",
        "kind": "number", "min": 100, "max": 600, "default": 200,
    },
    {
        "key": "fbs",
        "question": "Is the patient's **fasting blood sugar > 120 mg/dl**?",
        "kind": "choice",
        "options": [("No", 0), ("Yes", 1)],
    },
    {
        "key": "restecg",
        "question": "What is the patient's **resting ECG result**?",
        "kind": "choice",
        "options": [
            ("0 - Normal", 0),
            ("1 - ST-T Wave Abnormality", 1),
            ("2 - Left Ventricular Hypertrophy", 2),
        ],
    },
    {
        "key": "thalach",
        "question": "What is the **maximum heart rate** the patient "
                    "has achieved?",
        "kind": "number", "min": 60, "max": 250, "default": 150,
    },
    {
        "key": "exang",
        "question": "Does the patient get **angina (chest pain) "
                    "during exercise**?",
        "kind": "choice",
        "options": [("No", 0), ("Yes", 1)],
    },
    {
        "key": "oldpeak",
        "question": "What is the patient's **Oldpeak** value? "
                    "(ST depression induced by exercise, e.g. 1.0)",
        "kind": "decimal", "min": 0.0, "max": 10.0, "default": 1.0,
        "step": 0.1,
    },
    {
        "key": "slope",
        "question": "What is the **slope** of the peak exercise "
                    "ST segment?",
        "kind": "choice",
        "options": [
            ("0 - Upsloping", 0),
            ("1 - Flat", 1),
            ("2 - Downsloping", 2),
        ],
    },
    {
        "key": "ca",
        "question": "How many **major vessels** are colored by "
                    "fluoroscopy? (0 to 4)",
        "kind": "choice",
        "options": [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4)],
    },
    {
        "key": "thal",
        "question": "Last question — what is the patient's "
                    "**Thalassemia** type?",
        "kind": "choice",
        "options": [
            ("0 - Unknown", 0),
            ("1 - Normal", 1),
            ("2 - Fixed Defect", 2),
            ("3 - Reversible Defect", 3),
        ],
    },
]

TOTAL_QUESTIONS = len(QUESTIONS)

# Order in which the model's input row must be built.
# This MUST stay the same as the training data columns.
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ---------------------------
# Chat State (session)
# ---------------------------
GREETING = (
    "👋 Hello! I am your **AI Healthcare Assistant**. "
    "I will ask you a few questions about the patient, one by one, "
    "and then predict the heart disease risk."
)


def start_chat():
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.result = None
    st.session_state.messages = [
        ("assistant", GREETING),
        ("assistant", QUESTIONS[0]["question"]),
    ]


if "step" not in st.session_state:
    start_chat()

# ---------------------------
# Show the conversation so far
# ---------------------------
for role, text in st.session_state.messages:
    avatar = "🫀" if role == "assistant" else "🙂"
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

step = st.session_state.step

# ---------------------------
# Ask the current question
# ---------------------------
if step < TOTAL_QUESTIONS:
    q = QUESTIONS[step]

    st.progress(step / TOTAL_QUESTIONS)
    st.caption(f"Question {step + 1} of {TOTAL_QUESTIONS}")

    if q["kind"] == "number":
        value = st.number_input(
            "Your answer:",
            min_value=q["min"],
            max_value=q["max"],
            value=q["default"],
            key=f"input_{step}",
        )
        answer_value = int(value)
        answer_text = str(answer_value)

    elif q["kind"] == "decimal":
        value = st.number_input(
            "Your answer:",
            min_value=q["min"],
            max_value=q["max"],
            value=q["default"],
            step=q["step"],
            key=f"input_{step}",
        )
        answer_value = float(value)
        answer_text = f"{answer_value:.1f}"

    else:  # choice
        labels = [label for label, _ in q["options"]]
        chosen = st.selectbox(
            "Select one option:",
            labels,
            key=f"input_{step}",
        )
        answer_value = dict(q["options"])[chosen]
        answer_text = chosen

    if st.button("Send Answer ➤", key=f"send_{step}"):
        st.session_state.answers[q["key"]] = answer_value
        st.session_state.messages.append(("user", answer_text))
        st.session_state.step = step + 1

        if st.session_state.step < TOTAL_QUESTIONS:
            next_q = QUESTIONS[st.session_state.step]
            st.session_state.messages.append(
                ("assistant", next_q["question"])
            )
        else:
            st.session_state.messages.append(
                ("assistant",
                 "✅ Thank you! You have answered all 13 questions. "
                 "Here is your prediction result 👇")
            )
        st.rerun()

# ---------------------------
# All answers collected -> Prediction
# ---------------------------
else:
    if st.session_state.result is None:
        input_data = np.array(
            [[st.session_state.answers[key] for key in FEATURE_ORDER]]
        )
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        st.session_state.result = (int(prediction), probability * 100)

    prediction, risk_percent = st.session_state.result

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

    st.write("")
    if st.button("🔄 Start Again"):
        start_chat()
        st.rerun()

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
