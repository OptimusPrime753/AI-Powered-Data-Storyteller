import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Data Storyteller",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .insight-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class DataStoryTeller:
    def __init__(self):
        self.df = None
        self.insights = []
        self.visualizations = []
        
    def validate_dataset(self, df):
        """Validate uploaded dataset"""
        if df is None:
            return False, "No dataset provided"
        if df.empty:
            return False, "Dataset is empty"
        if df.shape[0] < 2:
            return False, "Dataset must have at least 2 rows"
        if df.shape[1] < 2:
            return False, "Dataset must have at least 2 columns"
        return True, "Dataset is valid"
    
    def perform_eda(self, df):
        """Perform Exploratory Data Analysis"""
        eda_results = {}
        
        # Basic info
        eda_results['shape'] = df.shape
        eda_results['columns'] = df.columns.tolist()
        eda_results['dtypes'] = df.dtypes.to_dict()
        eda_results['missing_values'] = df.isnull().sum().to_dict()
        eda_results['memory_usage'] = df.memory_usage(deep=True).sum()
        
        # Summary statistics
        eda_results['summary_stats'] = df.describe().to_dict()
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        eda_results['categorical_analysis'] = {}
        for col in categorical_cols:
            if len(df[col].unique()) <= 20:  # Only for columns with reasonable unique values
                eda_results['categorical_analysis'][col] = df[col].value_counts().to_dict()
        
        # Numerical correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            eda_results['correlations'] = df[numeric_cols].corr().to_dict()
        
        return eda_results
    
    def generate_insights(self, eda_results):
        """Generate natural language insights from EDA results"""
        insights = []
        
        # Dataset overview insights
        rows, cols = eda_results['shape']
        insights.append(f"📊 **Dataset Overview**: The dataset contains {rows:,} rows and {cols} columns, providing a comprehensive view of the data.")
        
        # Missing values insights
        missing_data = eda_results['missing_values']
        total_missing = sum(missing_data.values())
        if total_missing > 0:
            missing_cols = [col for col, count in missing_data.items() if count > 0]
            missing_pct = (total_missing / (rows * cols)) * 100
            insights.append(f"⚠️ **Data Quality**: {len(missing_cols)} columns have missing values, representing {missing_pct:.1f}% of all data points. Consider data cleaning strategies.")
        else:
            insights.append(f"✅ **Data Quality**: Excellent! No missing values detected across all columns.")
        
        # Numerical insights
        summary_stats = eda_results.get('summary_stats', {})
        if summary_stats:
            numeric_cols = list(summary_stats.keys())
            insights.append(f"🔢 **Numerical Analysis**: Found {len(numeric_cols)} numerical columns. Key metrics include mean, median, and distribution characteristics.")
            
            # Identify potential outliers
            for col in numeric_cols[:3]:  # Limit to first 3 columns for brevity
                if 'std' in summary_stats[col] and 'mean' in summary_stats[col]:
                    mean_val = summary_stats[col]['mean']
                    std_val = summary_stats[col]['std']
                    if std_val > 0:
                        cv = (std_val / mean_val) * 100
                        if cv > 50:
                            insights.append(f"📈 **Variability Alert**: Column '{col}' shows high variability (CV: {cv:.1f}%), suggesting potential outliers or diverse data distribution.")
        
        # Categorical insights
        categorical_analysis = eda_results.get('categorical_analysis', {})
        if categorical_analysis:
            for col, value_counts in categorical_analysis.items():
                unique_count = len(value_counts)
                most_common = max(value_counts.keys(), key=value_counts.get)
                most_common_pct = (value_counts[most_common] / rows) * 100
                insights.append(f"🏷️ **Category Analysis**: '{col}' has {unique_count} unique values. Most frequent: '{most_common}' ({most_common_pct:.1f}% of data).")
        
        # Correlation insights
        correlations = eda_results.get('correlations', {})
        if correlations:
            strong_correlations = []
            for col1 in correlations:
                for col2 in correlations[col1]:
                    if col1 != col2:
                        corr_value = correlations[col1][col2]
                        if abs(corr_value) > 0.7:
                            strong_correlations.append((col1, col2, corr_value))
            
            if strong_correlations:
                insights.append(f"🔗 **Strong Relationships**: Found {len(strong_correlations)} strong correlations (|r| > 0.7) between numerical variables.")
            else:
                insights.append(f"🔗 **Relationships**: No strong correlations detected. Variables appear relatively independent.")
        
        return insights
    
    def create_visualizations(self, df):
        """Create meaningful visualizations"""
        figures = []
        
        # 1. Dataset Overview - Column types distribution
        dtype_counts = df.dtypes.value_counts()
        fig1 = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index.astype(str),
            title="📊 Dataset Column Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_layout(height=400)
        figures.append(("Column Types Distribution", fig1))
        
        # 2. Missing Values Heatmap
        if df.isnull().sum().sum() > 0:
            missing_matrix = df.isnull().astype(int)
            fig2 = px.imshow(
                missing_matrix.T,
                title="🕳️ Missing Values Pattern",
                labels=dict(x="Records", y="Columns", color="Missing"),
                color_continuous_scale="Reds"
            )
            fig2.update_layout(height=400)
            figures.append(("Missing Values Pattern", fig2))
        
        # 3. Numerical columns distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Select up to 4 numerical columns for subplot
            selected_numeric = numeric_cols[:4]
            
            if len(selected_numeric) == 1:
                fig3 = px.histogram(
                    df, x=selected_numeric[0],
                    title=f"📈 Distribution of {selected_numeric[0]}",
                    marginal="box"
                )
            else:
                fig3 = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[f"Distribution of {col}" for col in selected_numeric[:4]],
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )
                
                for i, col in enumerate(selected_numeric[:4]):
                    row = i // 2 + 1
                    col_pos = i % 2 + 1
                    fig3.add_trace(
                        go.Histogram(x=df[col], name=col, showlegend=False),
                        row=row, col=col_pos
                    )
                
                fig3.update_layout(height=600, title_text="📈 Numerical Columns Distributions")
            
            figures.append(("Numerical Distributions", fig3))
        
        # 4. Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig4 = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="🔗 Correlation Matrix Heatmap",
                color_continuous_scale="RdBu"
            )
            fig4.update_layout(height=500)
            figures.append(("Correlation Matrix", fig4))
        
        # 5. Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            # Select first categorical column with reasonable unique values
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 20:
                    value_counts = df[col].value_counts()
                    fig5 = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"🏷️ Distribution of {col}",
                        labels={'x': col, 'y': 'Count'}
                    )
                    fig5.update_layout(height=400)
                    figures.append((f"Categorical Distribution - {col}", fig5))
                    break
        
        return figures
    
    def generate_pdf_report(self, df, insights, figures):
        """Generate PDF report"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 20)
        
        # Title
        pdf.cell(0, 15, 'AI-Powered Data Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Generation date
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        pdf.ln(10)
        
        # Executive Summary
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        rows, cols = df.shape
        summary_text = f"""
Dataset Overview:
- Total Records: {rows:,}
- Total Columns: {cols}
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
- Data Types: {dict(df.dtypes.value_counts())}

Key Findings:
This automated analysis reveals important patterns and characteristics in your dataset.
The insights below highlight data quality, distributions, and relationships that can
inform business decisions and further analysis strategies.
        """
        
        # Split text into lines and add to PDF
        for line in summary_text.strip().split('\n'):
            if line.strip():
                pdf.cell(0, 6, line.strip(), 0, 1)
        
        pdf.ln(10)
        
        # Insights section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Key Insights', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        for i, insight in enumerate(insights[:8], 1):  # Limit to 8 insights
            # Remove markdown formatting for PDF
            clean_insight = insight.replace('**', '').replace('*', '')
            pdf.cell(0, 6, f"{i}. {clean_insight}", 0, 1)
            pdf.ln(2)
        
        # Data summary table
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Data Summary Statistics', 0, 1)
        pdf.ln(5)
        
        # Add basic statistics
        pdf.set_font('Arial', '', 10)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe()
            pdf.cell(0, 6, f"Numerical Columns Summary (showing first 5 columns):", 0, 1)
            pdf.ln(3)
            
            # Headers
            pdf.set_font('Arial', 'B', 8)
            pdf.cell(30, 8, 'Statistic', 1, 0, 'C')
            for col in stats_df.columns[:5]:
                pdf.cell(30, 8, col[:10], 1, 0, 'C')
            pdf.ln()
            
            # Data rows
            pdf.set_font('Arial', '', 8)
            for stat in ['mean', 'std', 'min', '50%', 'max']:
                pdf.cell(30, 6, stat, 1, 0, 'C')
                for col in stats_df.columns[:5]:
                    value = stats_df.loc[stat, col]
                    pdf.cell(30, 6, f"{value:.2f}" if not pd.isna(value) else "N/A", 1, 0, 'C')
                pdf.ln()
        
        # Save PDF to bytes
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return pdf_bytes

# Initialize the storyteller
storyteller = DataStoryTeller()

# Main Dashboard
st.markdown('<h1 class="main-header">🤖 AI-Powered Data Storyteller</h1>', unsafe_allow_html=True)

st.markdown("""
Welcome to the AI-Powered Data Storyteller! Upload your CSV dataset and get:
- 📊 Automated data analysis and insights
- 📈 Interactive visualizations
- 📋 Executive summary report
- 💾 Downloadable PDF report
""")

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload a CSV file to analyze"
    )
    
    if uploaded_file is not None:
        # Load and validate data
        try:
            df = pd.read_csv(uploaded_file)
            storyteller.df = df
            
            is_valid, validation_message = storyteller.validate_dataset(df)
            
            if is_valid:
                st.success("✅ Dataset loaded successfully!")
                st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            else:
                st.error(f"❌ {validation_message}")
                df = None
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            df = None
    else:
        df = None
        st.info("👆 Upload a CSV file to get started")

# Main content area
if df is not None:
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🔍 AI Insights", "📈 Visualizations", "📋 Report"])
    
    with tab1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header">AI-Generated Insights</div>', unsafe_allow_html=True)
        
        if st.button("🧠 Generate AI Insights", type="primary"):
            with st.spinner("Analyzing your data with AI..."):
                # Perform EDA
                eda_results = storyteller.perform_eda(df)
                
                # Generate insights
                insights = storyteller.generate_insights(eda_results)
                storyteller.insights = insights
                
                # Display insights
                for insight in insights:
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        if hasattr(storyteller, 'insights') and storyteller.insights:
            st.subheader("💡 Key Insights Summary")
            for insight in storyteller.insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="section-header">Interactive Visualizations</div>', unsafe_allow_html=True)
        
        if st.button("📊 Generate Visualizations", type="primary"):
            with st.spinner("Creating visualizations..."):
                figures = storyteller.create_visualizations(df)
                storyteller.visualizations = figures
        
        # Display visualizations
        if hasattr(storyteller, 'visualizations') and storyteller.visualizations:
            for title, fig in storyteller.visualizations:
                st.subheader(title)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Click 'Generate Visualizations' to create interactive charts")
    
    with tab4:
        st.markdown('<div class="section-header">Executive Report</div>', unsafe_allow_html=True)
        
        if st.button("📋 Generate Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                # Ensure we have insights
                if not hasattr(storyteller, 'insights') or not storyteller.insights:
                    eda_results = storyteller.perform_eda(df)
                    storyteller.insights = storyteller.generate_insights(eda_results)
                
                # Generate PDF
                try:
                    pdf_bytes = storyteller.generate_pdf_report(df, storyteller.insights, [])
                    
                    # Create download button
                    st.success("✅ Report generated successfully!")
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
        
        # Display report preview
        if hasattr(storyteller, 'insights') and storyteller.insights:
            st.subheader("📋 Report Preview")
            
            st.markdown("### Executive Summary")
            st.info(f"""
            **Dataset Overview:**
            - Total Records: {df.shape[0]:,}
            - Total Columns: {df.shape[1]}
            - Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
            
            This automated analysis provides comprehensive insights into your dataset's structure,
            quality, and key patterns that can inform business decisions.
            """)
            
            st.markdown("### Key Insights")
            for i, insight in enumerate(storyteller.insights[:5], 1):
                st.markdown(f"**{i}.** {insight}")

else:
    # Welcome screen with sample data option
    st.markdown("""
    ## 🚀 Getting Started
    
    To begin your data analysis journey:
    
    1. **Upload your CSV file** using the sidebar
    2. **Explore the dataset** in the Dataset Overview tab
    3. **Generate AI insights** to understand your data patterns
    4. **Create visualizations** to see your data come to life
    5. **Download a comprehensive report** for sharing and presentation
    
    ### ✨ Features:
    - **Automated EDA**: Get instant summary statistics and data quality insights
    - **AI-Powered Analysis**: Natural language insights generated from your data
    - **Interactive Visualizations**: Dynamic charts and graphs
    - **Executive Reports**: Professional PDF reports ready for presentation
    - **Real-time Processing**: All analysis happens instantly in your browser
    
    Ready to discover the stories hidden in your data? Upload a CSV file to get started! 📊
    """)
    
    # Demo section
    st.markdown("---")
    st.markdown("### 🎯 Try with Sample Data")
    
    if st.button("🎲 Load Sample Dataset"):
        # Generate sample data
        np.random.seed(42)
        sample_data = {
            'Date': pd.date_range('2023-01-01', periods=1000, freq='D'),
            'Sales': np.random.normal(10000, 2000, 1000),
            'Marketing_Spend': np.random.normal(5000, 1000, 1000),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 1000),
            'Customer_Satisfaction': np.random.normal(4.2, 0.8, 1000)
        }
        
        # Add some correlations
        sample_data['Revenue'] = sample_data['Sales'] * 1.2 + np.random.normal(0, 1000, 1000)
        sample_data['Profit_Margin'] = (sample_data['Revenue'] - sample_data['Marketing_Spend']) / sample_data['Revenue']
        
        df = pd.DataFrame(sample_data)
        storyteller.df = df
        
        st.success("✅ Sample dataset loaded! Switch to the other tabs to explore.")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🤖 AI-Powered Data Storyteller | Built with Streamlit & ❤️
    <br>
    Transform your data into actionable insights with the power of AI
</div>
""", unsafe_allow_html=True)