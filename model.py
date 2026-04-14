import plotly.express as px
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import io
import pandas as pd
import numpy as np
import streamlit as st
import base64
import joblib
import json
from streamlit_option_menu import option_menu

# --- MODEL LOADING ---
scaler = joblib.load('scaler.joblib')
lr = joblib.load('model.joblib')

with open('metrics.json', 'r') as f:
    metrics = json.load(f)
accuracy_str = f"{metrics.get('accuracy', 'N/A')}%"

# --- UTILS ---
def generate_pdf_report(patient_data, probability, insights):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_fill_color(15, 17, 21)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(230, 57, 70)
    pdf.set_xy(10, 10)
    # Using modern XPos/YPos to avoid DeprecationWarnings and TypeErrors
    pdf.cell(0, 10, "PulseAI | Clinical Report", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 10, f"Generated for Priyansh Dua Diagnostic Suite | 2026", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Body
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 16)
    pdf.ln(15)
    pdf.cell(0, 10, "1. Patient Profile Snapshot", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 12)
    
    for label, val in patient_data.items():
        if label != "Target":
            pdf.cell(80, 8, f"{label}:", border=0, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(0, 8, f"{val}", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
    # Result
    pdf.ln(10)
    pdf.set_fill_color(230, 57, 70)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, f"  DIAGNOSTIC RISK LIKELIHOOD: {probability}%", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L', fill=True)
    
    # Insights
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Clinical Evaluation Highlights", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    
    if insights:
        for title, rec in insights:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, f"- {title}:", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 10)
            # Alignment must be a string 'L', 'C', 'R', or 'J' in fpdf2
            pdf.multi_cell(0, 5, f"{rec}", border=0, align='L')
            pdf.ln(2)
    else:
        pdf.cell(0, 10, "No critical clinical flags detected.", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5, "Disclaimer: PulseAI is a machine learning tool. This report does not substitute for a professional medical consultation. Always consult with a cardiologist for clinical interpretation.", border=0, align='C')
    
    return bytes(pdf.output())

# --- UI CONFIG ---
st.set_page_config(page_title="PulseAI | Heart Prediction",
                   page_icon="💓", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Lexend:wght@300;400;500&display=swap');
    :root { --ruby: #e63946; --slate: #1e293b; --bg: #0f1115; --border-slate: #334155; }
    .stApp { background-color: var(--bg); color: #f8fafc; font-family: 'Lexend', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; font-weight: 800; text-transform: uppercase; letter-spacing: -0.02em; }
    .stButton>button { background: var(--ruby) !important; color: white !important; font-family: 'Outfit'; border-radius: 0px !important; border: none !important; padding: 0.5rem 2rem !important; transition: all 0.2s ease; }
    .stButton>button:hover { background: #d00000 !important; transform: translateY(-2px); box-shadow: 0 10px 20px -10px var(--ruby); }
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input { background-color: transparent !important; border: 1px solid var(--border-slate) !important; border-radius: 0px !important; color: white !important; }
    .footer { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #475569; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f'<h2 style="color:var(--ruby); margin-bottom:0;">PULSEAI</h2><p style="font-size:10px; margin-top:0; color:#475569;">PRECISION CARDIAC DIAGNOSTICS</p>', unsafe_allow_html=True)
    selected = option_menu(None, ["Overview", "Diagnostic", "Developer"], 
        icons=['activity', 'heart-pulse', 'code-slash'], 
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#64748b", "font-size": "18px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#1e293b", "font-family": "Outfit", "color":"#94a3b8", "text-transform":"uppercase", "letter-spacing":"1px"},
            "nav-link-selected": {"background-color": "#1e293b", "color": "#e63946", "border-left": "3px solid #e63946"},
        })

readable_features = ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol', 'Fasting Sugar', 'Rest ECG', 'Max Heart Rate', 'Exercise Angina', 'ST Depression', 'ST Slope', 'Major Vessels', 'Thalassemia']

if selected == 'Overview':
    st.markdown("<h1>System Overview</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("### Global Model Fingerprint")
        st.markdown("<p style='color:#94a3b8;'>Visualizing the statistical significance of biomarkers across our training set.</p>", unsafe_allow_html=True)
        global_impact = lr.coef_[0]
        impact_df = pd.DataFrame({'Biomarker': readable_features, 'Impact': global_impact}).sort_values(by='Impact', ascending=True)
        fig_global = px.bar(impact_df, x='Impact', y='Biomarker', orientation='h', color='Impact', color_continuous_scale='RdBu_r', template='plotly_dark')
        fig_global.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_global, width='stretch')

    with col2:
        st.markdown("### Clinical Standards")
        st.info("PulseAI aligns with ACC/AHA guidelines for hypertension (Threshold: 130/80) and Hyperlipidemia (Threshold: 200 mg/dL).")
        st.warning(f"Logistic Regression Accuracy: {accuracy_str} | Clinical Grade Diagnostics enabled.")

elif selected == 'Diagnostic':
    st.markdown("<h1>Clinical Intake Form</h1>", unsafe_allow_html=True)
    with st.form("diagnostic_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Patient Age", 1, 120, 45)
            sex = st.selectbox("Sex at Birth", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
            cp_label = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
            cp_map = {"Typical Angina": 3, "Atypical Angina": 2, "Non-Anginal Pain": 1, "Asymptomatic": 0}
            Chest_Pain = cp_map[cp_label]
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 50, 250, 120)
        with col2:
            serum_cholestoral = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
            fasting_blood_sugar = st.number_input("Fasting Sugar (mg/dl)", 50, 500, 100)
            fbs = 1 if fasting_blood_sugar > 120 else 0
            restecg = st.selectbox("Resting ECG Results", options=[(0, "Normal"), (1, "ST-T Wave Abnormality"), (2, "Left Ventricular Hypertrophy")], format_func=lambda x: x[1])[0]
            thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
        with col3:
            oldpeak = st.number_input("ST Depression (Relative to Rest)", 0.0, 10.0, 0.0)
            slope = st.selectbox("ST Slope", options=[(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")], format_func=lambda x: x[1])[0]
            thal_opt = st.selectbox("Thalassemia Pattern", options=[(1, "Normal"), (2, "Fixed Defect"), (3, "Reversible Defect")])
            thal = thal_opt[0]
            exang = st.selectbox("Exercise Induced Angina", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
            ca = st.number_input("Major Vessels (0-4)", 0, 4, 0)
        submitted = st.form_submit_button("Execute Analysis")
    
    if submitted:
        input_data = np.array([[age, sex, Chest_Pain, resting_bp, serum_cholestoral, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        # Use .values to ensure consistency and suppress scikit-learn feature name warnings
        input_scaled = scaler.transform(input_data)
        pred = lr.predict(input_scaled); prob = lr.predict_proba(input_scaled)[0][1]; risk_pct = round(prob * 100, 1)
        
        # --- DYNAMIC INSIGHT ENGINE ---
        local_impact = input_scaled[0] * lr.coef_[0]
        local_df = pd.DataFrame({'Feature': readable_features, 'Drive': local_impact})
        
        flags = []
        # 1. Clinical Threshold Flags
        if resting_bp > 140: flags.append(("Hypertension", f"Resting BP ({resting_bp} mmHg) exceeds safety thresholds."))
        if serum_cholestoral > 240: flags.append(("Hyperlipidemia", f"Serum Cholesterol ({serum_cholestoral} mg/dl) is high."))
        if fbs == 1: flags.append(("Blood Sugar", f"Fasting sugar indicates glycemic risk."))
        if oldpeak > 1.0: flags.append(("ST Depression", f"Significant ischemic marker ({oldpeak}) detected."))
        if Chest_Pain > 0: flags.append(("Angina Pattern", f"Profile: {cp_label}. Consistent with clinical cardiac risk."))
        if ca > 0: flags.append(("Vessel Blockage", f"Major vessels show coloration ({ca}) indicating potential obstruction."))

        # 2. Dynamic Driver Detection
        if risk_pct > 50:
            top_drivers = local_df.sort_values(by='Drive', ascending=False).head(2)
            for _, row in top_drivers.iterrows():
                feat, drive = row['Feature'], row['Drive']
                if not any(feat in f[0] for f in flags):
                    flags.append((f"Risk Driver: {feat}", f"This marker shows strong statistical weight in your profile."))
        
        if risk_pct > 80 and len(flags) < 3:
            flags.append(("Systemic Correlation", "Risk is driven by a complex interplay of biomarkers patterns."))

        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 1.2])
        with res_col1:
            color = "#e63946" if pred[0] == 1 else "#22c55e"
            st.markdown(f"""<div style="padding:1.5rem; border:1px solid {color}; background:rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1);"><h3 style="color:{color} !important; margin:0;">{'POSITIVE RISK' if pred[0]==1 else 'NEGATIVE RISK'}</h3><p>Likelihood: <strong>{risk_pct}%</strong></p><div style="width:100%; background:#2d3436; height:4px; margin-top:10px;"><div style="width:{risk_pct}%; background:{color}; height:100%;"></div></div></div>""", unsafe_allow_html=True)
            
            patient_data = {"Age": age, "Sex": "Male" if sex==1 else "Female", "Chest Pain": cp_label, "BP": resting_bp, "Cholesterol": serum_cholestoral, "Sugar": fasting_blood_sugar, "Max HR": thalach, "Target": "Risk" if pred[0]==1 else "Normal"}
            report_bytes = generate_pdf_report(patient_data, risk_pct, flags)
            st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
            st.download_button("Export Clinical PDF", data=report_bytes, file_name=f"PulseAI_Report_{age}.pdf", mime="application/pdf")

        with res_col2:
            help_text = "This chart shows how each of your specific biomarkers influenced the final outcome. Red bars increased your risk, Green bars lowered it."
            st.markdown(f"#### Risk Contribution Analysis", help=help_text)
            local_plot_df = local_df[local_df['Drive'].abs() > 0.01].sort_values(by='Drive')
            fig_local = px.bar(local_plot_df, x='Drive', y='Feature', orientation='h', color='Drive', color_continuous_scale='RdYlGn_r', template='plotly_dark')
            fig_local.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_local, width='stretch')

        if flags:
            st.markdown("<br>### Clinical Highlights", unsafe_allow_html=True)
            for t, r in flags: st.markdown(f"**{t}**: {r}")

elif selected == 'Developer':
    st.markdown("<h1>Developer Metadata</h1>", unsafe_allow_html=True)
    st.markdown(f"""<div style="border:1px solid var(--border-slate); padding:2rem;"><p><strong>Primary Maintainer:</strong> Priyansh Dua</p><p><strong>Registry:</strong> priyanshduaofficial@gmail.com</p><p><strong>Infrastructure:</strong> Streamlit / Plotly / FPDF2 / Python 3.14</p><br><a href="https://www.linkedin.com/in/priyansh-dua-b8829431b/" class="stButton" style="text-decoration:none; display:inline-block; padding:10px 20px; background:#e63946; color:white;">Connect on LinkedIn</a></div>""", unsafe_allow_html=True)

st.markdown("""<div class="footer">DEVELOPED WITH PRECISION BY <a href="https://www.linkedin.com/in/priyansh-dua-b8829431b/" style="color:#e63946; text-decoration:none;">PRIYANSH DUA</a> | &copy; 2026</div>""", unsafe_allow_html=True)
