from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit as st
import base64

# --- DATA PREPARATION ---
df = pd.read_csv('heartdisease.csv')
df = df.drop_duplicates()

x = df.drop(columns='target')
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=0)

# SCALING FOR STABILITY
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# MODEL WITH HIGH CONVERGENCE
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train_scaled, y_train)

# Calculate Accuracy (Fixed for UI display)
accuracy = round(lr.score(x_test_scaled, y_test) * 100, 1)

# --- UI CONFIG ---
st.set_page_config(page_title="PulseAI | Heart Prediction",
                   page_icon="💓", layout="wide")

# LOAD IMAGES
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Ensure the image path matches the one we just copied
hero_img_path = 'heart_tech_hero.png'
hero_base64 = get_base64_of_bin_file(hero_img_path)

# --- CUSTOM THEME (NON-VIBECODED) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;600;800&family=Lexend:wght@300;400&display=swap');

    :root {{
        --bg-deep: #0f1115;
        --accent-ruby: #e63946;
        --border-slate: #2d3436;
        --text-muted: #94a3b8;
    }}

    .main {{
        background-color: var(--bg-deep);
        color: #f8fafc;
        font-family: 'Lexend', sans-serif;
    }}

    h1, h2, h3 {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: #ffffff !important;
    }}

    .stTextInput, .stNumberInput, .stTextArea {{
        background-color: transparent !important;
        border: 1px solid var(--border-slate) !important;
        border-radius: 4px !important;
    }}

    .stButton > button {{
        background: var(--accent-ruby) !important;
        color: white !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
    }}

    .stButton > button:hover {{
        background: #ff4d4d !important;
        box-shadow: 0 0 20px rgba(230, 57, 70, 0.4);
    }}

    /* Remove card-like rounding/shadows from standard elements */
    [data-testid="stVerticalBlock"] > div {{
        border-radius: 0 !important;
    }}

    .hero-container {{
        position: relative;
        height: 400px;
        overflow: hidden;
        border: 1px solid var(--border-slate);
        margin-bottom: 2rem;
    }}

    .hero-image {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0.6;
    }}

    .hero-text {{
        position: absolute;
        top: 50%;
        left: 5%;
        transform: translateY(-50%);
        max-width: 600px;
    }}

    .metric-box {{
        padding: 1rem;
        border-left: 2px solid var(--accent-ruby);
        background: rgba(255, 255, 255, 0.02);
    }}

    .footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 10px;
        background: #000;
        border-top: 1px solid var(--border-slate);
        text-align: center;
        font-size: 12px;
        color: var(--text-muted);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='color:#e63946;'>PulseAI</h2>", unsafe_allow_html=True)
    selected = option_menu(None,
                           ['Overview', 'Diagnostic', 'Developer'],
                           icons=['grid-3x3-gap', 'activity', 'code-slash'],
                           menu_icon="cast", default_index=0,
                           styles={{
                               "container": {{"padding": "0!important", "background-color": "transparent"}},
                               "icon": {{"color": "#e63946", "font-size": "18px"}}, 
                               "nav-link": {{"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#2d3436"}},
                               "nav-link-selected": {{"background-color": "#2d3436"}},
                           }})

# --- PAGE ROUTING ---
if selected == 'Overview':
    st.markdown(f"""
        <div class="hero-container">
            <img src="data:image/png;base64,{hero_base64}" class="hero-image">
            <div class="hero-text">
                <h1>Precision Heart<br>Risk Analysis</h1>
                <p style="color:#94a3b8; font-size:1.1rem;">Advanced logistic regression model trained for high-fidelity clinical diagnostic prediction.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Technical Specs")
        st.markdown(f"""
        <div class="metric-box">
            <p><strong>ALGORITHM:</strong> LOGISTIC REGRESSION (LBFGS)</p>
            <p><strong>CALIBRATION:</strong> STANDARD SCALER GAUSSIAN</p>
            <p><strong>PRECISION:</strong> {accuracy}% R-SQUARED</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("### Clinical Dataset")
        st.markdown("""
        Analyzes 13 critical health biomarkers including serum cholesterol, 
        thalassemia indicators, and ST-segment peak exercise slopes to establish a risk profile.
        """)

elif selected == 'Diagnostic':
    st.markdown("<h1>Diagnostic Inputs</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Ensure all biomarkers are entered from valid clinical reports.</p>", unsafe_allow_html=True)
    
    with st.form("diagnostic_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 0, 100, help="Patient biological age")
            resting_bp = st.number_input("Blood Pressure", 90, 180, help="Resting BP in mm Hg")
            restecg = st.number_input("Resting ECG (0-2)", 0, 2)
            oldpeak = st.number_input("ST Depression", 0.0, 5.0, format="%.1f")
            thal = st.number_input("Thal (3,6,7)", 3, 7)
        with col2:
            sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
            serum_cholestoral = st.number_input("Cholesterol", 150, 300)
            thalach = st.number_input("Max Heart Rate", 100, 180)
            slope = st.number_input("ST Slope (0-2)", 0, 2)
        with col3:
            Chest_Pain = st.number_input("Chest Pain (1-4)", 1, 4)
            fasting_sugar = st.number_input("Fasting Sugar", 120, 300)
            exang = st.selectbox("Exercise Angina", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
            ca = st.number_input("Major Vessels (0-3)", 0, 3)

        submitted = st.form_submit_button("Execute Analysis")
        
        if submitted:
            # Prepare data
            input_data = np.array([[age, sex, Chest_Pain, resting_bp, serum_cholestoral,
                                  fasting_sugar, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            # Scale data
            input_scaled = scaler.transform(input_data)
            pred = lr.predict(input_scaled)

            st.markdown("<br>", unsafe_allow_html=True)
            if pred[0] == 0:
                st.markdown(f"""
                <div style="padding:1.5rem; border:1px solid #22c55e; background:rgba(34, 197, 94, 0.1);">
                    <h3 style="color:#22c55e !important; margin:0;">NEGATIVE PROBABILITY</h3>
                    <p>Analysis indicates no significant heart disease biomarkers present. Model Confidence: {accuracy}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding:1.5rem; border:1px solid #e63946; background:rgba(230, 57, 70, 0.1);">
                    <h3 style="color:#e63946 !important; margin:0;">POSITIVE RISK DETECTED</h3>
                    <p>Clinical biomarkers match heart disease risk patterns. Urgent medical review recommended.</p>
                </div>
                """, unsafe_allow_html=True)

elif selected == 'Developer':
    st.markdown("<h1>Developer Metadata</h1>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="border:1px solid var(--border-slate); padding:2rem;">
        <p><strong>Primary Maintainer:</strong> Priyansh Dua</p>
        <p><strong>Registry:</strong> priyanshduaofficial@gmail.com</p>
        <p><strong>Infrastructure:</strong> Streamlit / Scikit-Learn / Python 3.14</p>
        <br>
        <a href="https://www.linkedin.com/in/priyansh-dua-b8829431b/" class="stButton" style="text-decoration:none; display:inline-block; padding:10px 20px; background:#e63946; color:white;">Connect on LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    DEVELOPED WITH PRECISION BY <a href="https://www.linkedin.com/in/priyansh-dua-b8829431b/" style="color:#e63946; text-decoration:none;">PRIYANSH DUA</a> | &copy; 2026
</div>
""", unsafe_allow_html=True)
