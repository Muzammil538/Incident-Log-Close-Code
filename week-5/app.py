# ============================================================
# 🚀 STREAMLIT INCIDENT ROOT CAUSE RECOMMENDATION SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy.sparse import hstack, csr_matrix

# ============================================================
# LOAD TRAINED FILES
# ============================================================

model = joblib.load('best_model.pkl')

tfidf = joblib.load('tfidf_vectorizer.pkl')

selector = joblib.load('feature_selector.pkl')

feature_columns = joblib.load('feature_columns.pkl')

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Incident Root Cause Recommendation System",

    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🚀 Incident Root Cause Recommendation System")

st.markdown("""
AI-powered automated incident root-cause recommendation system using Machine Learning and NLP.
""")

# ============================================================
# INPUT SECTION
# ============================================================

st.header("📌 Incident Details")

col1, col2 = st.columns(2)

# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    incident_state = st.selectbox(

        "Incident State",

        [

            "Closed",

            "Resolved",

            "Active"
        ]
    )

    impact = st.selectbox(

        "Impact",

        [

            "High",

            "Medium",

            "Low"
        ]
    )

    urgency = st.selectbox(

        "Urgency",

        [

            "High",

            "Medium",

            "Low"
        ]
    )

    priority = st.selectbox(

        "Priority",

        [

            "P1",

            "P2",

            "P3",

            "P4"
        ]
    )

# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    # ============================================
    # CATEGORY DROPDOWN
    # ============================================

    category = st.selectbox(

        "Category",

        [

            "Software",

            "Hardware",

            "Network",

            "Database",

            "Security",

            "Email",

            "Server",

            "Application",

            "Storage"
        ]
    )

    # ============================================
    # SUBCATEGORY DROPDOWN
    # ============================================

    subcategory_options = {

        "Software": [

            "Application Crash",

            "Installation",

            "Performance",

            "Bug",

            "Compatibility"
        ],

        "Hardware": [

            "Disk Failure",

            "CPU Issue",

            "Memory Failure",

            "Peripheral Issue"
        ],

        "Network": [

            "Connectivity",

            "DNS",

            "VPN",

            "Firewall",

            "Bandwidth"
        ],

        "Database": [

            "Connection Failure",

            "Slow Query",

            "Database Crash"
        ],

        "Security": [

            "Authentication",

            "Permission",

            "Access Violation"
        ],

        "Email": [

            "Email Not Working",

            "SMTP Failure",

            "Mailbox Issue"
        ],

        "Server": [

            "Server Down",

            "High Load",

            "Service Failure"
        ],

        "Application": [

            "Login Failure",

            "Timeout",

            "API Failure"
        ],

        "Storage": [

            "Disk Full",

            "Storage Failure",

            "Backup Failure"
        ]
    }

    subcategory = st.selectbox(

        "Subcategory",

        subcategory_options[category]
    )

# ============================================================
# SYMPTOM INPUT
# ============================================================

symptom = st.text_area(

    "📝 Incident Symptom Description",

    "Describe the issue..."
)

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔍 Predict Root Cause"):

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = {

        'incident_state': incident_state,

        'impact': impact,

        'urgency': urgency,

        'priority': priority,

        'category': category,

        'subcategory': subcategory
    }

    df = pd.DataFrame([input_data])

    # ========================================================
    # ONE HOT ENCODING
    # ========================================================

    df = pd.get_dummies(df)

    # Add missing columns efficiently
    missing_cols = list(
        set(feature_columns) - set(df.columns)
    )

    missing_df = pd.DataFrame(

        0,

        index=df.index,

        columns=missing_cols
    )

    # Concatenate
    df = pd.concat(
        [df, missing_df],
        axis=1
    )

    # Match exact order
    df = df[feature_columns]

    # Defragment dataframe
    df = df.copy()

    # ========================================================
    # TF-IDF TRANSFORM
    # ========================================================

    text_features = tfidf.transform([symptom])

    # ========================================================
    # COMBINE FEATURES
    # ========================================================

    X_struct = csr_matrix(df.values)

    X_final = hstack([

        X_struct,

        text_features
    ])

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    X_final = selector.transform(X_final)

    # ========================================================
    # TOP-K PREDICTION
    # ========================================================

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_final)[0]

        classes = model.classes_

        # Top 3 predictions
        top_k_idx = np.argsort(probs)[-3:][::-1]

        top_predictions = [

            (classes[i], probs[i])

            for i in top_k_idx
        ]

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(X_final)[0]

        classes = model.classes_

        top_k_idx = np.argsort(scores)[-3:][::-1]

        # Convert to pseudo probability
        scores_exp = np.exp(scores)

        probs = scores_exp / np.sum(scores_exp)

        top_predictions = [

            (classes[i], probs[i])

            for i in top_k_idx
        ]

    # ========================================================
    # LABEL MAP
    # ========================================================

    label_map = {

        1: "Access Issue",

        3: "Database/Backend Failure",

        4: "Hardware Issue",

        5: "Software/Application Failure",

        6: "Application Failure",

        7: "Network Connectivity Issue",

        8: "Network Infrastructure Failure",

        9: "Configuration/Deployment Issue",

        10: "Authentication/Permission Failure",

        11: "System Hardware Failure",

        16: "Critical Service Failure"
    }

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.subheader("🎯 Top Root Cause Recommendations")

    for idx, (pred, prob) in enumerate(top_predictions):

        label = label_map.get(

            pred,

            f"Close Code {pred}"
        )

        confidence = round(prob * 100, 2)

        st.info(f"""
        #{idx+1} Recommendation

        Root Cause:
        {label}

        
        """)
        
        # Confidence:
        # {confidence}%

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

