import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

# Simple version without external modules
st.set_page_config(page_title="AI Data Storyteller", layout="wide")

st.title("🤖 AI-Powered Data Storyteller")
st.write("Upload your CSV file for automated analysis")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read data
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        
        st.subheader("📈 Basic Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Simple visualization
        st.subheader("🎨 Basic Visualizations")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig, ax = plt.subplots()
                df[numeric_cols[0]].hist(ax=ax)
                ax.set_title(f"Distribution of {numeric_cols[0]}")
                st.pyplot(fig)
            
            with col2:
                # Correlation heatmap if multiple numeric columns
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)
                    ax.set_title("Correlation Heatmap")
                    st.pyplot(fig)
        
        st.success("Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")