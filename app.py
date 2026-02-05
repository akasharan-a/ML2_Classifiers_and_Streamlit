import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.metrics import confusion_matrix, accuracy_score , recall_score, precision_score, f1_score , matthews_corrcoef ,roc_auc_score
from model.preprocessing import decode_target ,encode_target,target_categories

st.set_page_config(
    page_title="Fabric Quality Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)
if 'results' not in st.session_state:
    st.session_state.results = None

st.title("👕 Fabric Quality Classifier",text_alignment='center')
st.markdown("---")

df_test= None
model_map = {
            "Logistic Regression": "LogisticRegression",
            "Decision Tree": "DecisionTreeClassifier",
            "KNN": "KNeighborsClassifier",
            "Random Forest": "RandomForestClassifier",
            "Naive Bayes": "NaiveBayes",
            "XGBoost": "XGBClassifier"}

with st.sidebar:
    st.header("📄 Data Source")
    data_option = st.radio("Choose Data:", ["Upload CSV","Use Sample Data"])
    sample_data_path = "data/Industrial Fabric Quality Inspection Dataset - Test.csv"
    with open(sample_data_path, "rb") as f:
            st.download_button("🔽Download Sample Data", f, file_name="Industrial Fabric Quality Inspection Dataset - Test.csv",use_container_width=True)

    if data_option == "Use Sample Data":
        df_test = pd.read_csv(sample_data_path)

    else:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df_test = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file.")

    st.header("✨ Model Selection")
    model_type = st.selectbox(
        "Choose Algorithm",
        tuple(model_map.keys()),
    )
    
    st.divider()
    evaluate = st.button("👉 Evaluate",use_container_width=True,help='Click to run evaluation on the selected model and dataset.',type='primary')
    
if df_test is not None:

    with st.container():
        st.subheader("📊 Data Preview (100 rows)")
        st.dataframe(df_test.head(100), use_container_width=True)    
    if  evaluate:
    #Model
        st.header(model_type,text_alignment="center")
                
        filename = model_map[model_type]
        model_pipeline = joblib.load(f"model_files/{filename}.pkl")

        try:
            y_pred = model_pipeline.predict(df_test)
            y_score = model_pipeline.predict_proba(df_test)
            y_pred_df = pd.DataFrame(decode_target(y_pred), columns=["Predicted_Fabric_Quality"])
            st.markdown("---")
            st.subheader("✅ Predictions")
            with st.expander("View Predictions"):
                st.dataframe(y_pred_df, width="content")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    #Metrics
         
        st.markdown("---")
        st.subheader("📈 Performance Metrics")

        y_true = encode_target(df_test['fabric_quality'])
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred,average='weighted')
        rec = recall_score(y_true, y_pred,average='weighted')
        f1 = f1_score(y_true, y_pred,average='weighted')
        
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score,multi_class='ovr',average='weighted')
        
        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{acc:.3}",border=True)
        b.metric("Precision", f"{prec:.3}",border=True)
        c.metric("Recall", f"{rec:.3}",border=True)
        d ,e,f = st.columns(3)
        d.metric("F1-Score", f"{f1:.3}",border=True)                  
        e.metric("MCC", f"{mcc:.3}",border=True)
        f.metric("AUC", f"{auc:.3}",border=True)
        labels = list(target_categories.keys())
        cm = confusion_matrix(decode_target(y_true), decode_target(y_pred) ,labels=labels)


        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        sns.heatmap(
            cm, 
            annot=True, 
            linewidths= 0.5,
            fmt='d', 
            cmap='rocket', 
            xticklabels=labels, 
            yticklabels=labels,
            cbar=False,
            ax=ax
        )
        fig.supxlabel('Predicted Label',color='gray')
        fig.supylabel('True Label', color='gray')
        ax.tick_params(colors='gray') 
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        ax.title.set_color('gray')
        
        st.subheader("🖼️ Confusion Matrix")
        st.pyplot(fig, transparent=True,width=700)
    else:
        st.info(" Click the Evaluate button in the sidebar to run model evaluation.")

else:
    st.info(" Please upload a CSV file in the sidebar to get started or click Use Sample Data.")

    st.markdown(
        """
    ### Quick Start Guide
    1. **Upload**: Drag and drop your dataset (Use download button in sidebar to get sample data). 
        
        or 
        
       **Click** 'Use Sample Data' to load sample dataset.
    2. **Select Model**: Choose a classification algorithm from the sidebar.
    3. **Evaluate**: Click the 'Evaluate' button to see predictions and metrics.
    
    """
    )
