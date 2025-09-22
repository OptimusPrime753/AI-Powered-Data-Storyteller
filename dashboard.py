import streamlit as st
import pandas as pd
from data_analyzer import DataAnalyzer
from report_generator import ReportGenerator
import os

st.set_page_config(page_title="AI Data Storyteller", layout="wide")

st.title("🤖 AI-Powered Data Storyteller")
st.write("Upload your CSV file to get automated insights, visualizations, and reports!")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Save uploaded file
    with open("temp_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize analyzer
    analyzer = DataAnalyzer()
    report_gen = ReportGenerator()
    
    # Load and display data
    df_preview = analyzer.load_data("temp_data.csv")
    st.subheader("Data Preview")
    st.dataframe(df_preview)
    
    # Generate insights
    if st.button("Generate Analysis"):
        with st.spinner("Analyzing data..."):
            # Basic analysis
            insights = analyzer.basic_analysis()
            
            # Display insights
            st.subheader("📊 Automated Insights")
            for insight in insights:
                st.write(f"• {insight}")
            
            # Generate visualizations
            st.subheader("📈 Visualizations")
            plots = analyzer.generate_visualizations()
            
            # Display plots
            col1, col2 = st.columns(2)
            for i, plot in enumerate(plots):
                if os.path.exists(plot):
                    if i % 2 == 0:
                        with col1:
                            st.image(plot, caption=plot.replace('.png', ''))
                    else:
                        with col2:
                            st.image(plot, caption=plot.replace('.png', ''))
            
            # Generate and download report
            st.subheader("📄 Report Generation")
            report_path = report_gen.generate_report(insights, plots)
            
            with open(report_path, "rb") as file:
                st.download_button(
                    label="Download Executive Report (PDF)",
                    data=file,
                    file_name="data_analysis_report.pdf",
                    mime="application/pdf"
                )
    
    # Cleanup
    if os.path.exists("temp_data.csv"):
        os.remove("temp_data.csv")