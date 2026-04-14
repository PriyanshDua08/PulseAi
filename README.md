# PulseAI: Precision Cardiac Diagnostics ❤️‍🩹

PulseAI is a professional-grade, machine learning diagnostic suite designed for clinical heart disease risk prediction. Moving beyond simple binary outcomes, PulseAI acts as a transparent clinical assistant that evaluates patient biomarkers, provides statistical model explainability (XAI), and dynamically flags health risks.

Built with **Python, Scikit-Learn, Streamlit, and Plotly**, PulseAI is designed to align with modern medical standards, offering a high-density, brutalist UI tailored for medical professionals.

---

## 🌟 Key Features

* **Advanced Predictive Engine**: Powered by a tuned Logistic Regression model utilizing normalized biomedical inputs to accurately predict cardiac risk probability.
* **Explainable AI (XAI)**:
  * **Global Model Fingerprint**: Visualize the aggregate statistical significance of different biomarkers across the entire training dataset.
  * **Risk Contribution Analysis**: Unpack the "why" behind every individual prediction with interactive horizontal bar charts highlighting the top risk drivers (e.g., *Major Vessels*, *Chest Pain Type*).
* **Dynamic Insight Engine**: Automatically synthesizes and flags critical biomarkers based on ACC/AHA thresholds and correlation models (e.g., detecting masked Angina patterns even when blood pressure is normal).
* **Clinical Report Generation**: 1-Click export to high-fidelity, branded PDF reports using FPDF2, providing patients and clinicians with actionable diagnostic takeaways.

## 📊 The Dataset

The predictive logic is trained on clinical data (`heartdisease.csv`) evaluating 13 standardized cardiac metrics:
- Demographics: Age, Sex
- Clinical Presentation: Chest Pain Type (Angina Profiling), Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar
- Electrocardiography: Resting ECG, Max Heart Rate, Exercise Induced Angina, ST Depression, ST Slope
- Scans & Hematology: Fluoroscopy-colored Major Vessels, Thalassemia Pattern

## 🔧 Technical Stack & Dependencies

- **Language**: Python 3.14
- **Machine Learning**: `scikit-learn`, `numpy`, `pandas`
- **Frontend / UI**: `streamlit`, `streamlit-option-menu`
- **Data Visualization**: `plotly`
- **Document Generation**: `fpdf2`

Install all clinical dependencies via:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Diagnostic Suite

To launch the PulseAI server locally:

```bash
streamlit run model.py
```
This will start the local server and open the PulseAI dashboard in your default browser at `http://localhost:8501`.


## 📬 Contact the Maintainer

**Priyansh Dua**  
Primary Maintainer & Architect  
&nbsp;&nbsp;<a href="https://www.linkedin.com/in/priyansh-dua-b8829431b/"><img src="https://www.felberpr.com/wp-content/uploads/linkedin-logo.png" width="30"></img></a>  
© 2026 Priyansh Dua
