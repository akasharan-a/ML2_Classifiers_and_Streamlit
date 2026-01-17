import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Material ML Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR MATERIAL DESIGN ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    
    /* Material Card Style */
    div.stElementContainer {
        border-radius: 8px;
    }
    
    .block-container {
        padding-top: 2rem;
    }

    /* Customizing buttons to look more Material */
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background-color: #bb86fc;
        color: #000;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        box-shadow: 0 3px 5px rgba(0,0,0,0.3);
    }
    
    .stButton>button:hover {
        background-color: #9965f4;
        box-shadow: 0 5px 15px rgba(187, 134, 252, 0.4);
        color: #000;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 1px solid #333;
    }

    /* Input boxes */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #2c2c2c;
        color: white;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- APP HEADER ---
st.title("ML Classifier")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    
    st.subheader("Model Selection")
    model_type = st.selectbox(
        "Choose Algorithm", 
        ("Random Forest", "Logistic Regression", "SVM")
    )
    
    test_size = st.slider("Test Set Size (%)", 10, 50, 20)
    random_state = st.number_input("Random Seed", value=42)

# --- MAIN CONTENT ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 1. DATA PREVIEW
    with st.container():
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select Target Variable (Y)", df.columns)
        with col2:
            feature_cols = st.multiselect("Select Features (X)", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])

    if st.button("🚀 Train & Evaluate"):
        if not feature_cols:
            st.error("Please select at least one feature.")
        else:
            # Preprocessing (Basic dropna for demo)
            data = df[feature_cols + [target_col]].dropna()
            X = data[feature_cols]
            y = data[target_col]
            
            # Simple encoding for strings if necessary
            X = pd.get_dummies(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )

            # Model Selection
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100)
            elif model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 2. METRICS
            st.markdown("---")
            st.subheader("📈 Performance Metrics")
            m_col1, m_col2, m_col3 = st.columns(3)
            
            acc = accuracy_score(y_test, y_pred)
            m_col1.metric("Accuracy", f"{acc:.2%}")
            m_col2.metric("Features Used", len(feature_cols))
            m_col3.metric("Samples", len(data))

            # 3. VISUALIZATIONS
            st.markdown("---")
            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred)
                labels = sorted(y.unique())

            with v_col2:
                st.write("**Classification Report**")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)

else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to get started.")
    
    # Placeholder Sample Data Info
    st.markdown("""
    ### Quick Start Guide
    1. **Upload**: Drag and drop your dataset.
    2. **Configure**: Select your features and target class.
    3. **Train**: Click the 'Train' button to see results instantly.
    
    *Designed with Material principles for a clean, accessible ML experience.*
    """)