# app_pdf_patch_visuals.py
import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
import io
import re
import traceback
from pathlib import Path

# Plotly and plotting (visualizations)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AI Data Storyteller",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS (kept minimal)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.analysis_results = {}

    def load_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                self.df = pd.read_csv(uploaded_file)
            else:
                # try excel
                self.df = pd.read_excel(uploaded_file)
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, f"Error loading data: {e}"

    def perform_comprehensive_analysis(self):
        """Perform comprehensive EDA"""
        if self.df is None:
            return {}

        results = {}
        results["shape"] = self.df.shape
        results["columns"] = list(self.df.columns)
        results["data_types"] = self.df.dtypes.astype(str).to_dict()
        results["missing_values"] = self.df.isnull().sum().to_dict()
        results["missing_percentage"] = (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        results["duplicates"] = int(self.df.duplicated().sum())

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        results["numeric_columns"] = numeric_cols
        results["numeric_stats"] = self.df[numeric_cols].describe().transpose().to_dict() if numeric_cols else {}

        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        results["categorical_columns"] = categorical_cols
        results["categorical_stats"] = {}
        for col in categorical_cols:
            vc = self.df[col].value_counts(dropna=False)
            results["categorical_stats"][col] = {
                "unique_count": int(self.df[col].nunique(dropna=False)),
                "top_categories": vc.head(5).to_dict()
            }

        if len(numeric_cols) > 1:
            results["correlation_matrix"] = self.df[numeric_cols].corr()
        else:
            results["correlation_matrix"] = None

        return results


class InteractiveVisualizations:
    def __init__(self, df):
        self.df = df

    def create_correlation_heatmap(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        corr_matrix = self.df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap"
        )
        fig.update_layout(height=600)
        return fig

    def create_distribution_plots(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None
        cols = min(2, len(numeric_cols))
        rows = (len(numeric_cols) + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Distribution of {col}" for col in numeric_cols])
        for i, col in enumerate(numeric_cols):
            row = (i // cols) + 1
            col_num = (i % cols) + 1
            fig.add_trace(go.Histogram(x=self.df[col], name=col, nbinsx=30), row=row, col=col_num)
        fig.update_layout(height=300 * rows, title_text="Distribution Analysis", showlegend=False)
        return fig

    def create_scatter_matrix(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        selected_cols = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
        fig = px.scatter_matrix(self.df[selected_cols], title="Scatter Matrix", height=800)
        return fig

    def create_categorical_analysis(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            fig = px.box(self.df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}", height=500)
            return fig
        return None


class ReportGenerator:
    def __init__(self):
        pass

    def clean_text(self, text):
        """Sanitize text for fpdf (latin-1). Replace bullets and remove non-latin characters."""
        if text is None:
            return ""
        s = str(text)
        # Replace bullets and special characters
        s = s.replace("•", "- ").replace("–", "-").replace("—", "-")
        s = s.replace("**", "").replace("__", "")  # Remove markdown formatting
        # Remove emojis and non-ASCII characters
        try:
            s = re.sub(r'[^\x00-\x7F]+', '', s)
        except:
            pass
        return s

    def _write_multiline(self, pdf: FPDF, text: str, line_height: float = 6):
        """Properly handle multiline text with correct margins"""
        safe_text = self.clean_text(text)
        # Calculate proper width (page width - left margin - right margin)
        effective_width = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.multi_cell(effective_width, line_height, txt=safe_text, align='L')
        pdf.ln(1)  # Small space after each paragraph

    def _write_section_header(self, pdf: FPDF, text: str):
        """Write section headers with proper formatting"""
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 0, 128)  # Dark blue color for headers
        self._write_multiline(pdf, text, line_height=8)
        pdf.set_text_color(0, 0, 0)  # Reset to black
        pdf.ln(2)

    def _write_subheader(self, pdf: FPDF, text: str):
        """Write subheaders"""
        pdf.set_font("Arial", 'B', 12)
        self._write_multiline(pdf, text, line_height=7)
        pdf.set_font("Arial", '', 11)
        pdf.ln(1)

    def generate_pdf_bytes(self, analysis_results, insights):
        """
        Generate PDF in memory and return bytes with proper formatting.
        """
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Set proper margins
            pdf.set_left_margin(15)
            pdf.set_right_margin(15)
            pdf.set_top_margin(15)
            
            pdf.add_page()
            
            # Title Section
            pdf.set_font("Arial", 'B', 18)
            pdf.set_text_color(0, 0, 128)  # Dark blue
            title_width = pdf.w - 30  # Account for margins
            pdf.cell(title_width, 10, txt="AI-Powered Data Analysis Report", ln=True, align='C')
            pdf.ln(5)
            
            # Timestamp
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(100, 100, 100)  # Gray color
            pdf.cell(title_width, 8, 
                    txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    ln=True, align='C')
            pdf.ln(10)
            pdf.set_text_color(0, 0, 0)  # Reset to black

            # Executive Summary Section
            self._write_section_header(pdf, "Executive Summary")
            pdf.set_font("Arial", '', 11)
            
            if insights:
                for insight in insights[:6]:  # Limit to 6 insights
                    cleaned_insight = self.clean_text(insight)
                    # Add bullet point
                    pdf.set_font("Arial", '', 11)
                    effective_width = pdf.w - pdf.l_margin - pdf.r_margin - 5  # Space for bullet
                    pdf.cell(5, 6, txt="-")  # Bullet point
                    pdf.multi_cell(effective_width, 6, txt=cleaned_insight, align='L')
                    pdf.ln(1)
            else:
                self._write_multiline(pdf, "No insights generated.", line_height=6)
            
            pdf.ln(5)

            # Dataset Overview Section
            pdf.add_page()
            self._write_section_header(pdf, "Dataset Overview")
            
            pdf.set_font("Arial", '', 11)
            shape = analysis_results.get('shape', (0, 0))
            missing_total = sum(analysis_results.get('missing_values', {}).values()) if analysis_results.get('missing_values') else 0
            duplicates = analysis_results.get('duplicates', 0)
            
            overview_data = [
                f"Rows: {shape[0]:,}",
                f"Columns: {shape[1]}",
                f"Total Missing Values: {missing_total:,}",
                f"Duplicate Rows: {duplicates}"
            ]
            
            for item in overview_data:
                pdf.cell(5, 6, txt="-")
                pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 5, 6, txt=item, align='L')
                pdf.ln(1)

            # Numeric Summary Section
            pdf.ln(5)
            self._write_section_header(pdf, "Numeric Summary")
            
            numeric_cols = analysis_results.get('numeric_columns', [])[:5]  # Limit to 5 columns
            numeric_stats = analysis_results.get('numeric_stats', {})
            
            if numeric_cols:
                pdf.set_font("Arial", '', 10)
                for col in numeric_cols:
                    stats = numeric_stats.get(col, {})
                    if stats:
                        pdf.set_font("Arial", 'B', 10)
                        self._write_multiline(pdf, f"{col}:", line_height=6)
                        pdf.set_font("Arial", '', 10)
                        stats_text = f"  Count: {stats.get('count', 'N/A')}, Min: {stats.get('min', 'N/A'):.2f}, Max: {stats.get('max', 'N/A'):.2f}, Mean: {stats.get('mean', 'N/A'):.2f}"
                        self._write_multiline(pdf, stats_text, line_height=6)
                        pdf.ln(1)
            else:
                self._write_multiline(pdf, "No numeric columns detected.", line_height=6)

            # Categorical Summary Section
            pdf.ln(5)
            self._write_section_header(pdf, "Categorical Summary")
            
            categorical_cols = analysis_results.get('categorical_columns', [])[:5]  # Limit to 5 columns
            categorical_stats = analysis_results.get('categorical_stats', {})
            
            if categorical_cols:
                pdf.set_font("Arial", '', 10)
                for col in categorical_cols:
                    stats = categorical_stats.get(col, {})
                    if stats:
                        pdf.set_font("Arial", 'B', 10)
                        self._write_multiline(pdf, f"{col}:", line_height=6)
                        pdf.set_font("Arial", '', 10)
                        unique_count = stats.get('unique_count', 'N/A')
                        top_categories = stats.get('top_categories', {})
                        top_items = list(top_categories.items())[:3]  # Show top 3
                        top_text = ", ".join([f"{k} ({v})" for k, v in top_items])
                        
                        self._write_multiline(pdf, f"  Unique values: {unique_count}", line_height=6)
                        self._write_multiline(pdf, f"  Top categories: {top_text}", line_height=6)
                        pdf.ln(1)
            else:
                self._write_multiline(pdf, "No categorical columns detected.", line_height=6)

            # Recommendations Section
            pdf.ln(5)
            self._write_section_header(pdf, "Recommendations & Conclusions")
            
            recommendations = [
                "Address columns with high missing values through imputation or removal",
                "Investigate and handle duplicate rows if they are erroneous",
                "Explore strong correlations for feature engineering considerations",
                "Validate distributions and handle outliers in numeric variables",
                "Consider data collection strategies if dataset size is limited"
            ]
            
            pdf.set_font("Arial", '', 11)
            for i, recommendation in enumerate(recommendations, 1):
                pdf.cell(5, 6, txt=f"{i}.")
                pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 5, 6, 
                              txt=f" {recommendation}", align='L')
                pdf.ln(1)

            # Generate PDF bytes
            pdf_output = pdf.output(dest='S')
            
            if isinstance(pdf_output, str):
                pdf_bytes = pdf_output.encode('latin-1')
            else:
                pdf_bytes = bytes(pdf_output) if isinstance(pdf_output, bytearray) else str(pdf_output).encode('latin-1')
            
            return pdf_bytes

        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            print("PDF generation error:", traceback.format_exc())
            return None

def main():
    st.markdown('<div class="main-header">🤖 AI-Powered Data Storyteller</div>', unsafe_allow_html=True)
    st.markdown("Upload your dataset to unlock powerful AI-driven insights and interactive visualizations")

    # session state init
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'insights' not in st.session_state:
        st.session_state.insights = []

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            success, message = st.session_state.analyzer.load_data(uploaded_file)
            if success:
                st.success("✅ " + message)
                if st.button("🚀 Start AI Analysis", type="primary"):
                    with st.spinner("Performing comprehensive analysis..."):
                        st.session_state.analysis_results = st.session_state.analyzer.perform_comprehensive_analysis()
                        analyzer = st.session_state.analyzer

                        def make_insights(results):
                            ins = []
                            if not results:
                                return ins
                            rows, cols = results.get('shape', (0, 0))
                            ins.append(f"Dataset contains {rows:,} rows and {cols} columns.")
                            total_missing = sum(results.get('missing_values', {}).values())
                            if total_missing > 0:
                                ins.append(f"Total missing values: {total_missing:,}.")
                            if results.get('numeric_columns'):
                                ins.append(f"{len(results['numeric_columns'])} numeric columns detected.")
                            if results.get('categorical_columns'):
                                ins.append(f"{len(results['categorical_columns'])} categorical columns detected.")
                            if results.get('duplicates', 0) > 0:
                                ins.append(f"{results.get('duplicates')} duplicate rows detected.")
                            return ins

                        st.session_state.insights = make_insights(st.session_state.analysis_results)
                        st.session_state.analysis_done = True
                        try:
                            st.rerun()
                        except Exception:
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass
            else:
                st.error("❌ " + message)

        st.markdown("---")
        st.header("🔧 Settings")
        st.selectbox("Theme", ["Light", "Dark"], key="theme")
        st.slider("Chart Quality", 1, 10, 8, key="chart_quality")

    # Main content
    if st.session_state.analysis_done and st.session_state.analyzer.df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🤖 AI Insights", "📈 Visualizations", "📄 Report"])

        with tab1:
            st.header("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{st.session_state.analysis_results['shape'][0]:,}")
            with col2:
                st.metric("Total Columns", st.session_state.analysis_results['shape'][1])
            with col3:
                missing_total = sum(st.session_state.analysis_results['missing_values'].values())
                st.metric("Missing Values", f"{missing_total:,}")
            with col4:
                st.metric("Duplicate Rows", st.session_state.analysis_results['duplicates'])

            st.subheader("Data Preview")
            st.dataframe(st.session_state.analyzer.df.head(10), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Types")
                dtype_df = pd.DataFrame({
                    'Column': st.session_state.analysis_results['columns'],
                    'Data Type': [str(st.session_state.analysis_results['data_types'].get(col, '')) for col in st.session_state.analysis_results['columns']]
                })
                st.dataframe(dtype_df, use_container_width=True)
            with col2:
                st.subheader("Missing Values Analysis")
                missing_df = pd.DataFrame({
                    'Column': list(st.session_state.analysis_results['missing_values'].keys()),
                    'Missing Count': list(st.session_state.analysis_results['missing_values'].values()),
                    'Missing %': list(st.session_state.analysis_results['missing_percentage'].values())
                })
                st.dataframe(missing_df, use_container_width=True)

        with tab2:
            st.header("🤖 AI-Generated Insights")
            for i, insight in enumerate(st.session_state.insights, 1):
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    default_value = i in [2, 5]
                    st.checkbox("", key=f"insight_{i}", value=default_value)
                with col2:
                    st.markdown(
                        f'<div style="background-color: #e8f4fd; padding: 12px; border-left: 4px solid #1f77b4; border-radius: 5px; margin: 5px 0;">{insight}</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.analysis_results.get('numeric_columns'):
                    st.subheader("📈 Numeric Statistics")
                    numeric_stats = pd.DataFrame(st.session_state.analysis_results['numeric_stats'])
                    st.dataframe(numeric_stats, use_container_width=True)
            with col2:
                if st.session_state.analysis_results.get('categorical_columns'):
                    st.subheader("📝 Categorical Analysis")
                    for col in st.session_state.analysis_results['categorical_columns'][:3]:
                        st.write(f"**{col}**: {st.session_state.analysis_results['categorical_stats'][col]['unique_count']} unique values")
                        top_cats = st.session_state.analysis_results['categorical_stats'][col]['top_categories']
                        st.write("Top categories:", ", ".join([f"{k} ({v})" for k, v in list(top_cats.items())[:3]]))

        with tab3:
            st.header("📈 Interactive Visualizations")
            visualizer = InteractiveVisualizations(st.session_state.analyzer.df)

            viz_option = st.selectbox(
                "Choose Visualization Type",
                ["Correlation Heatmap", "Distribution Analysis", "Scatter Matrix", "Categorical Analysis", "All Visualizations"]
            )

            if viz_option == "Correlation Heatmap":
                fig = visualizer.create_correlation_heatmap()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis")

            elif viz_option == "Distribution Analysis":
                fig = visualizer.create_distribution_plots()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns found for distribution analysis")

            elif viz_option == "Scatter Matrix":
                fig = visualizer.create_scatter_matrix()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for scatter matrix")

            elif viz_option == "Categorical Analysis":
                fig = visualizer.create_categorical_analysis()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need both categorical and numeric columns for this analysis")

            elif viz_option == "All Visualizations":
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = visualizer.create_correlation_heatmap()
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                    fig3 = visualizer.create_distribution_plots()
                    if fig3:
                        st.plotly_chart(fig3, use_container_width=True)
                with col2:
                    fig2 = visualizer.create_scatter_matrix()
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
                    fig4 = visualizer.create_categorical_analysis()
                    if fig4:
                        st.plotly_chart(fig4, use_container_width=True)

        with tab4:
            st.header("📄 Report Generation")
            st.subheader("Executive Summary")
            st.write("Generate a clean PDF report (text-only) with data overview, insights and conclusion.")

            report_gen = ReportGenerator()
            if st.button("📥 Generate PDF Report", type="primary"):
                with st.spinner("Generating report..."):
                    try:
                        pdf_bytes = report_gen.generate_pdf_bytes(st.session_state.analysis_results, st.session_state.insights)
                        if pdf_bytes:
                            st.download_button(
                                label="📄 Download AI Analysis Report (PDF)",
                                data=pdf_bytes,
                                file_name="ai_data_analysis_report.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("PDF generation failed. Check app logs for details.")
                    except Exception:
                        st.error("Unexpected error while generating report. See traceback below.")
                        st.text(traceback.format_exc())

            st.subheader("Report Preview")
            st.write("**Key sections included in the report:**")
            st.write("✅ Executive Summary")
            st.write("✅ Dataset Overview")
            st.write("✅ AI Insights")
            st.write("✅ Statistical Analysis")
            st.write("✅ Recommendations")

    else:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>🚀 Welcome to AI Data Storyteller</h2>
            <p>Upload your dataset to get started with AI-powered analysis!</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Try with Sample Data", use_container_width=True):
                sample_data = {
                    'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
                    'Sales': np.random.normal(1000, 200, 100).cumsum(),
                    'Customers': np.random.poisson(50, 100),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
                    'Revenue': np.random.exponential(500, 100)
                }
                sample_df = pd.DataFrame(sample_data)
                st.session_state.analyzer.df = sample_df
                st.session_state.analysis_results = st.session_state.analyzer.perform_comprehensive_analysis()
                st.session_state.insights = [
                    f"Dataset contains {st.session_state.analysis_results['shape'][0]:,} rows and {st.session_state.analysis_results['shape'][1]} columns.",
                    f"{len(st.session_state.analysis_results['numeric_columns'])} numeric columns detected.",
                    f"{len(st.session_state.analysis_results['categorical_columns'])} categorical columns detected."
                ]
                st.session_state.analysis_done = True
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()