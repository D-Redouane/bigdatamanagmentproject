import streamlit as st
import pandas as pd
import numpy as np
import json
import re

# Set up page configurations
st.set_page_config(page_title="Diagnostics Studio", page_icon="🏥", layout="centered")

# Hide standard Streamlit menu padding
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD NATIVE SPARK EXPORTED METADATA
# ----------------------------------------------------------
@st.cache_resource
def load_spark_metadata():
    with open("saved_spark_models/model_metadata.json", "r") as f:
        meta = json.load(f)
    return meta

try:
    spark_meta = load_spark_metadata()
    labels = spark_meta["labels"]
except Exception as e:
    st.error(f"Could not load Spark model metadata JSON. Error: {e}")
    st.stop()

# ----------------------------------------------------------
# LIGHTWEIGHT NATIVE PREDICTION ENGINES
# ----------------------------------------------------------
def predict_logistic_regression(features, lr_meta):
    # Features format: [age, gender, los, diagnoses]
    coefs = np.array(lr_meta["coefficients"]) # Shape: (num_classes, num_features)
    intercepts = np.array(lr_meta["intercept"]) # Shape: (num_classes,)
    
    # Compute raw multi-class scores: scores = X * Coefs^T + Intercept
    scores = np.dot(coefs, features) + intercepts
    return np.argmax(scores)

def parse_and_eval_tree(features, debug_string):
    # A lightweight text rule-parser for Spark's .toDebugString tree format
    # Splits the tree rules into lines to find the matching node
    lines = debug_string.strip().split('\n')
    current_indent = 0
    line_idx = 0
    
    # Feature mapping dictionary
    feat_map = {
        "feature 0": features[0], # age
        "feature 1": features[1], # gender
        "feature 2": features[2], # length_of_stay
        "feature 3": features[3]  # num_diagnoses
    }
    
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line:
            line_idx += 1
            continue
            
        # Match a leaf node statement, e.g., "Predict: 3.0"
        if "Predict:" in line:
            val = float(line.split("Predict:")[1].strip())
            return int(val)
            
        # Match an internal split node condition statement
        match = re.match(r"If \((feature \d+)\s+([<= >]+)\s+([\d\.]+)\)", line)
        if match:
            f_name, op, threshold = match.groups()
            f_val = feat_map[f_name]
            thresh_val = float(threshold)
            
            # Evaluate internal conditional statement
            condition_met = False
            if op == "<=": condition_met = (f_val <= thresh_val)
            elif op == ">": condition_met = (f_val > thresh_val)
            
            if condition_met:
                # Proceed directly into the next line (the If block)
                line_idx += 1
            else:
                # Find the matching sequential "Else" fallback branch statement
                # By matching indentation depth levels
                orig_indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
                seek_idx = line_idx + 1
                while seek_idx < len(lines):
                    next_line_indent = len(lines[seek_idx]) - len(lines[seek_idx].lstrip())
                    if next_line_indent == orig_indent and "Else" in lines[seek_idx]:
                        line_idx = seek_idx + 1
                        break
                    seek_idx += 1
                else:
                    line_idx += 1
        else:
            line_idx += 1
            
    return 0

# ----------------------------------------------------------
# DICTIONARIES & RISK COEFFICIENTS
# ----------------------------------------------------------
ICD_DICTIONARY = {
    "0": "Infectious & Parasitic Diseases", "1": "Neoplasms / Tumors", 
    "2": "Endocrine & Metabolic Disorders", "3": "Diseases of Blood/Immune", 
    "4": "Mental & Behavioral Disorders", "5": "Diseases of Nervous System", 
    "6": "Diseases of Circulatory System (Heart)", "7": "Diseases of Respiratory System (Lungs)", 
    "8": "Diseases of Digestive System", "9": "Diseases of Genitourinary System",
    "I": "Diseases of Circulatory System (ICD-10 Chapter I)", 
    "V": "Health Status & Routine Follow-ups", "E": "External Injuries / Poisoning",
    "A": "Infectious & Parasitic Conditions", "C": "Neoplasms / Cancer", 
    "D": "Blood & Immune Mechanisms", "F": "Mental & Behavioral Disorders", 
    "G": "Nervous System Diseases", "H": "Eye & Ear Disorders", 
    "J": "Respiratory System Diseases", "K": "Digestive System Diseases", 
    "L": "Skin & Tissue Diseases", "M": "Musculoskeletal Conditions", 
    "N": "Genitourinary Diseases", "O": "Pregnancy & Childbirth", 
    "P": "Perinatal Conditions", "Q": "Congenital Malformations", 
    "R": "Abnormal Clinical Findings", "S": "Injury & Poisoning Consequences", 
    "T": "External Injuries (T-Range)", "Z": "Health Status Influences (ICD-10 Chapter Z)"
}

HIGH_RISK_CATEGORIES = ["1", "6", "7", "C", "I", "J"]

# ----------------------------------------------------------
# RENDER PREMIUM UI CONTAINER
# ----------------------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 12px 12px 0px 0px; font-family: 'Segoe UI', sans-serif; color: white; text-align: center;">
    <h2 style="margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; color: white;">🏥 Diagnostics Analytics Studio</h2>
    <p style="margin: 6px 0 0 0; font-size: 13px; color: #e0e6ed; opacity: 0.9;">100% Native Spark Parameters UI (Bypassing Windows Write Bug)</p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    chosen_model_name = st.selectbox("🧠 AI Model Engine:", ["Logistic Regression", "Decision Tree", "Random Forest"], index=0)
    
    st.markdown("<p style='font-weight: 600; color: #34495e; font-size:14px; margin-bottom:-5px;'>👤 Patient Age:</p>", unsafe_allow_html=True)
    ui_age = st.slider("", min_value=18, max_value=100, value=50)
    
    ui_gender_text = st.selectbox("🧬 Biological Sex:", ["Male ♂", "Female ♀"])
    ui_gender = 1 if ui_gender_text == "Male ♂" else 0
    
    st.markdown("<p style='font-weight: 600; color: #34495e; font-size:14px; margin-bottom:-5px;'>📅 Length of Stay:</p>", unsafe_allow_html=True)
    ui_los = st.slider("", min_value=1, max_value=60, value=5)
    
    st.markdown("<p style='font-weight: 600; color: #34495e; font-size:14px; margin-bottom:-5px;'>🗂️ Comorbidities:</p>", unsafe_allow_html=True)
    ui_diagnoses = st.slider("", min_value=1, max_value=20, value=3)

    execute_inference = st.button("⚡ RUN NATIVE SPARK PARAMETER INFERENCE", use_container_width=True)

# ----------------------------------------------------------
# INFERENCE EXECUTION
# ----------------------------------------------------------
if execute_inference:
    input_features = [float(ui_age), float(ui_gender), float(ui_los), float(ui_diagnoses)]
    
    # Route input values to correct in-memory parser matching Spark's structure
    if chosen_model_name == "Logistic Regression":
        pred_index = predict_logistic_regression(input_features, spark_meta["logistic_regression"])
    elif chosen_model_name == "Decision Tree":
        pred_index = parse_and_eval_tree(input_features, spark_meta["trees"]["decision_tree_rules"])
    else:  # Random Forest evaluation fallback
        pred_index = parse_and_eval_tree(input_features, spark_meta["trees"]["random_forest_rules"])
        
    raw_code = str(labels[pred_index])
    readable_diagnosis = ICD_DICTIONARY.get(raw_code, f"Unclassified Condition (Code {raw_code})")
    
    is_dangerous_disease = raw_code in HIGH_RISK_CATEGORIES
    is_critical_patient = ui_los > 14 or ui_diagnoses >= 7 or (ui_age >= 65 and is_dangerous_disease)
    
    if is_critical_patient or is_dangerous_disease:
        bg_color = "#fdf2f2"; border_color = "#f8b4b4"; text_color = "#9b1c1c"; badge_bg = "#f8b4b4"
        status_text = "⚠️ HIGH CRITICAL RISK CASE"
    else:
        bg_color = "#f3faf7"; border_color = "#def7ec"; text_color = "#03543f"; badge_bg = "#bcf0da"
        status_text = "✅ STABLE PATHWAY CASE"
        
    analytics_html_lines = []
    if ui_age >= 65:
        analytics_html_lines.append("<li><b>Age Risk:</b> Geriatric status indicates potentially reduced physiological reserves.</li>")
    if ui_los > 14:
        analytics_html_lines.append(f"<li><b>Resource Notice:</b> Extended timeline ({ui_los} days) signals exceptional care complications.</li>")
    if ui_diagnoses >= 7:
        analytics_html_lines.append(f"<li><b>Comorbidity:</b> Extreme complexity database interaction found ({ui_diagnoses} codes logged).</li>")
    
    if not analytics_html_lines:
        analytics_html_lines.append("<li>Patient clinical attributes sit within typical standard baseline operating tolerances.</li>")
        
    dashboard_template = f"""
    <div style="background-color: {bg_color}; border: 2px solid {border_color}; padding: 22px; border-radius: 12px; font-family: 'Segoe UI', Tahoma, sans-serif; color: #333; width: 100%; box-shadow: 0px 8px 24px rgba(0,0,0,0.06);">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid {border_color}; padding-bottom: 12px; margin-bottom: 15px;">
            <span style="font-weight: bold; font-size: 13px; color: {text_color}; background-color: {badge_bg}; padding: 5px 12px; border-radius: 20px; letter-spacing: 0.3px;">
                {status_text}
            </span>
            <span style="font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">
                Engine: Spark JSON Weights
            </span>
        </div>
        <p style="margin: 5px 0; font-size: 14px; color: #4b5563;">
            <b>Patient Snapshot:</b> {ui_age} y/o {ui_gender_text} | <b>LOS:</b> {ui_los} Days | <b>Comorbidities:</b> {ui_diagnoses}
        </p>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border-left: 6px solid {text_color}; margin: 18px 0; box-shadow: inset 0 1px 3px rgba(0,0,0,0.04), 0 2px 4px rgba(0,0,0,0.02);">
            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.7px; color: #6b7280; font-weight: 700; display: block; margin-bottom: 4px;">
                Predicted Primary Diagnosis
            </span>
            <h3 style="margin: 0; color: {text_color}; font-size: 19px; font-weight: 700; line-height: 1.3;">
                {readable_diagnosis}
            </h3>
        </div>
        <h4 style="margin: 12px 0 6px 0; font-size: 12px; color: #4b5563; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700;">
            Clinical Insights & Analytics
        </h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #374151; line-height: 1.6;">
            {"".join(analytics_html_lines)}
        </ul>
    </div>
    """
    st.markdown(dashboard_template, unsafe_allow_html=True)