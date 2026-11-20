import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. SYNTHETIC CLINICAL DATASET GENERATION
# ---------------------------------------------------------
@st.cache_data
def generate_clinical_data(n_samples=1500):
    np.random.seed(42)
    
    # Features: Symptoms, Vital Signs, Lab Results, Medical History
    data = pd.DataFrame({
        'Age': np.random.randint(18, 85, n_samples),
        'Pain_Duration_Hrs': np.random.randint(1, 72, n_samples),
        'Pain_Score_1_to_10': np.random.randint(1, 11, n_samples),
        'RLQ_Pain': np.random.binomial(1, 0.3, n_samples),         # Right Lower Quadrant (Appendicitis indicator)
        'RUQ_Pain': np.random.binomial(1, 0.25, n_samples),        # Right Upper Quadrant (Cholecystitis indicator)
        'Flank_Pain': np.random.binomial(1, 0.25, n_samples),      # Renal Colic indicator
        'Nausea_Vomiting': np.random.binomial(1, 0.5, n_samples),
        'Temperature_C': np.round(np.random.normal(37.2, 0.9, n_samples), 1),
        'Heart_Rate_bpm': np.random.randint(60, 130, n_samples),
        'Systolic_BP': np.random.randint(90, 160, n_samples),
        'WBC_Count': np.round(np.random.normal(9.5, 4.0, n_samples), 1), # White Blood Cells (10^9/L)
        'CRP_Level': np.round(np.random.exponential(15, n_samples), 1),  # C-Reactive Protein (mg/L)
        'History_Gallstones': np.random.binomial(1, 0.15, n_samples),
        'History_KidneyStones': np.random.binomial(1, 0.15, n_samples)
    })
    
    # Clip realistic ranges
    data['WBC_Count'] = np.clip(data['WBC_Count'], 3.5, 25.0)
    data['Temperature_C'] = np.clip(data['Temperature_C'], 36.0, 40.5)

    # Rule-based clinical logic to assign realistic Diseases & Severity
    diseases = []
    severities = []

    for _, row in data.iterrows():
        # Determine Disease
        if row['RLQ_Pain'] == 1 and row['WBC_Count'] > 11.5 and row['Temperature_C'] > 37.8:
            disease = 'Acute Appendicitis'
        elif row['RUQ_Pain'] == 1 and (row['History_Gallstones'] == 1 or row['CRP_Level'] > 20):
            disease = 'Acute Cholecystitis'
        elif row['Flank_Pain'] == 1 and (row['History_KidneyStones'] == 1 or row['Pain_Score_1_to_10'] >= 7):
            disease = 'Renal Colic (Kidney Stone)'
        else:
            disease = 'Acute Gastroenteritis'
        diseases.append(disease)

        # Determine Severity Risk Level (Low, Moderate, Critical)
        risk_score = 0
        if row['Pain_Score_1_to_10'] >= 8: risk_score += 2
        elif row['Pain_Score_1_to_10'] >= 5: risk_score += 1
        if row['WBC_Count'] > 14.0: risk_score += 2
        if row['Temperature_C'] > 38.5: risk_score += 1
        if row['Heart_Rate_bpm'] > 110: risk_score += 1
        if disease in ['Acute Appendicitis', 'Acute Cholecystitis']: risk_score += 1

        if risk_score >= 4:
            severities.append('High / Critical')
        elif risk_score >= 2:
            severities.append('Moderate')
        else:
            severities.append('Low')

    data['Disease'] = diseases
    data['Severity'] = severities
    return data

# ---------------------------------------------------------
# 2. TRAIN XGBOOST MODELS (DISEASE & SEVERITY)
# ---------------------------------------------------------
@st.cache_resource
def train_models(df):
    X = df.drop(columns=['Disease', 'Severity'])
    
    le_disease = LabelEncoder()
    y_disease = le_disease.fit_transform(df['Disease'])
    
    le_severity = LabelEncoder()
    y_severity = le_severity.fit_transform(df['Severity'])

    X_train, X_test, yd_train, yd_test, ys_train, ys_test = train_test_split(
        X, y_disease, y_severity, test_size=0.2, random_state=42
    )

    # Disease Prediction Model (XGBoost)
    model_disease = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='mlogloss')
    model_disease.fit(X_train, yd_train)

    # Severity Prediction Model (XGBoost)
    model_severity = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='mlogloss')
    model_severity.fit(X_train, ys_train)

    return X_train, model_disease, model_severity, le_disease, le_severity

# ---------------------------------------------------------
# 3. STREAMLIT CLINICAL DECISION SUPPORT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Explainable CDSS - Acute Abdominal Pain", layout="wide")

st.title("🏥 Explainable AI Clinical Decision Support System (CDSS)")
st.markdown("**Acute Abdominal Pain Severity & Disease Prediction using XGBoost, SHAP & LIME**")
st.divider()

# Load Data & Models
df = generate_clinical_data()
X_train, model_disease, model_severity, le_disease, le_severity = train_models(df)

# Sidebar: Patient Input Form
st.sidebar.header("📋 Patient Clinical Parameters")

age = st.sidebar.slider("Age", 18, 85, 34)
pain_dur = st.sidebar.slider("Pain Duration (Hours)", 1, 72, 12)
pain_score = st.sidebar.slider("Pain Score (1 - 10)", 1, 10, 8)

st.sidebar.subheader("Symptoms & Location")
rlq = st.sidebar.checkbox("Right Lower Quadrant (RLQ) Pain", value=True)
ruq = st.sidebar.checkbox("Right Upper Quadrant (RUQ) Pain", value=False)
flank = st.sidebar.checkbox("Flank / Lower Back Pain", value=False)
nausea = st.sidebar.checkbox("Nausea / Vomiting", value=True)

st.sidebar.subheader("Vital Signs & Lab Results")
temp = st.sidebar.number_input("Temperature (°C)", 36.0, 41.0, 38.4, step=0.1)
hr = st.sidebar.number_input("Heart Rate (bpm)", 50, 160, 105)
sbp = st.sidebar.number_input("Systolic BP (mmHg)", 80, 200, 120)
wbc = st.sidebar.number_input("WBC Count (10^9/L)", 3.0, 30.0, 14.8, step=0.1)
crp = st.sidebar.number_input("CRP Level (mg/L)", 0.0, 200.0, 22.0, step=1.0)

st.sidebar.subheader("Medical History")
hist_gall = st.sidebar.checkbox("History of Gallstones", value=False)
hist_kidney = st.sidebar.checkbox("History of Kidney Stones", value=False)

# Create input dataframe
input_data = pd.DataFrame([{
    'Age': age,
    'Pain_Duration_Hrs': pain_dur,
    'Pain_Score_1_to_10': pain_score,
    'RLQ_Pain': int(rlq),
    'RUQ_Pain': int(ruq),
    'Flank_Pain': int(flank),
    'Nausea_Vomiting': int(nausea),
    'Temperature_C': temp,
    'Heart_Rate_bpm': hr,
    'Systolic_BP': sbp,
    'WBC_Count': wbc,
    'CRP_Level': crp,
    'History_Gallstones': int(hist_gall),
    'History_KidneyStones': int(hist_kidney)
}])

# Make Predictions
disease_probs = model_disease.predict_proba(input_data)[0]
pred_disease_idx = np.argmax(disease_probs)
pred_disease = le_disease.inverse_transform([pred_disease_idx])[0]

severity_probs = model_severity.predict_proba(input_data)[0]
pred_severity_idx = np.argmax(severity_probs)
pred_severity = le_severity.inverse_transform([pred_severity_idx])[0]

# Display Diagnosis & Triage Priority
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Underlying Disease", pred_disease, f"{disease_probs[pred_disease_idx]*100:.1f}% Confidence")
with col2:
    st.metric("Predicted Severity Level", pred_severity, f"{severity_probs[pred_severity_idx]*100:.1f}% Confidence")
with col3:
    if pred_severity == "High / Critical":
        st.error("🚨 TRIAGE: Immediate Surgical / Emergency Consult")
    elif pred_severity == "Moderate":
        st.warning("⚠️ TRIAGE: Urgent Imaging & Lab Monitoring")
    else:
        st.success("✅ TRIAGE: Standard Outpatient / Observation")

st.divider()

# ---------------------------------------------------------
# 4. EXPLAINABLE AI (SHAP & LIME)
# ---------------------------------------------------------
st.subheader("🔍 Explainable AI (XAI) Clinical Justification")
tab1, tab2 = st.tabs(["SHAP Explanation (Feature Impact)", "LIME Explanation (Local Rule Breakdown)"])

with tab1:
    st.write(f"**Why did the model predict `{pred_disease}`?** (SHAP Force/Waterfall Contributions)")
    explainer_shap = shap.TreeExplainer(model_disease)
    shap_values = explainer_shap.shap_values(input_data)
    
    # Extract SHAP values for the predicted class
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_disease_idx][0]
    else:
        class_shap = shap_values[0, :, pred_disease_idx]

    shap_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Patient Value': input_data.iloc[0].values,
        'SHAP Impact': class_shap
    }).sort_values(by='SHAP Impact', key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#ff4b4b' if val > 0 else '#1f77b4' for val in shap_df['SHAP Impact'][:8][::-1]]
    ax.barh(shap_df['Feature'][:8][::-1], shap_df['SHAP Impact'][:8][::-1], color=colors)
    ax.set_xlabel("SHAP Value (Positive = Increases Likelihood, Negative = Decreases Likelihood)")
    ax.set_title(f"Top Factors Driving {pred_disease} Prediction")
    st.pyplot(fig)

with tab2:
    st.write(f"**Local Interpretable Model-Agnostic Explanations (LIME) for `{pred_disease}`**")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        class_names=le_disease.classes_.tolist(),
        mode='classification'
    )
    exp = lime_explainer.explain_instance(
        data_row=input_data.iloc[0].values,
        predict_fn=model_disease.predict_proba,
        num_features=8,
        top_labels=1
    )
    
    lime_list = exp.as_list(label=pred_disease_idx)
    lime_df = pd.DataFrame(lime_list, columns=['Clinical Condition Rule', 'Contribution Weight'])
    st.dataframe(lime_df, use_container_width=True)