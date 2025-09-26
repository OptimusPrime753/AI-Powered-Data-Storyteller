# app_pdf_patch_fixed.py
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


class ReportGenerator:
    def __init__(self):
        # nothing to store here for now
        pass

    def clean_text(self, text):
        """Sanitize text for fpdf (latin-1). Replace bullets and remove non-latin characters."""
        if text is None:
            return ""
        s = str(text)
        # replace common bullets/emojis/dashes with ascii equivalents
        s = s.replace("•", "- ")
        s = s.replace("–", "-").replace("—", "-")
        # remove emoji ranges and other non-ascii characters
        # wide emoji block
        try:
            s = re.sub(r'[\U0001F300-\U0001FAFF]', '', s)
        except re.error:
            # if regex above not supported on this narrow build, skip it
            pass
        s = re.sub(r'[^\x00-\x7F]+', '', s)  # remove remaining non-ascii
        return s

    def _write_multiline(self, pdf: FPDF, text: str, line_height: float = 7):
        """Write sanitized multiline text to pdf using a safe width."""
        safe = self.clean_text(text)
        usable_w = float(pdf.w) - float(pdf.l_margin) - float(pdf.r_margin)
        if usable_w <= 0:
            # set conservative margins
            pdf.set_left_margin(10)
            pdf.set_right_margin(10)
            usable_w = float(pdf.w) - float(pdf.l_margin) - float(pdf.r_margin)
            if usable_w <= 0:
                raise RuntimeError("Page width too small to render text.")
        # multi_cell with usable width
        pdf.multi_cell(usable_w, line_height, txt=safe)
        return

    def generate_pdf_bytes(self, analysis_results, insights):
        """
        Generate PDF in memory and return bytes.
        Returns: bytes (PDF) on success, or None on failure.
        """
        try:
            # Create FPDF and configure
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            # Use a basic latin font available by default
            DEFAULT_FONT = "Arial"
            pdf.set_font(DEFAULT_FONT, size=12)

            # Title
            pdf.set_font(DEFAULT_FONT, 'B', 16)
            usable_w = float(pdf.w) - float(pdf.l_margin) - float(pdf.r_margin)
            pdf.cell(usable_w, 10, txt=self.clean_text("AI-Powered Data Analysis Report"), ln=True, align='C')
            pdf.ln(4)

            # Timestamp
            pdf.set_font(DEFAULT_FONT, 'I', 10)
            pdf.cell(usable_w, 8, txt=self.clean_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True)
            pdf.ln(6)

            # Executive Summary
            pdf.set_font(DEFAULT_FONT, 'B', 14)
            self._write_multiline(pdf, "Executive Summary", line_height=8)
            pdf.ln(2)
            pdf.set_font(DEFAULT_FONT, size=11)
            for insight in (insights or [])[:6]:
                # Clean insight and write as a short paragraph
                txt = self.clean_text(insight).replace("**", "")
                self._write_multiline(pdf, f"- {txt}", line_height=7)
            pdf.ln(4)

            # Dataset Overview
            pdf.add_page()
            pdf.set_font(DEFAULT_FONT, 'B', 14)
            self._write_multiline(pdf, "Dataset Overview", line_height=9)
            pdf.ln(2)
            pdf.set_font(DEFAULT_FONT, size=11)
            shape = analysis_results.get('shape', (0, 0))
            missing_total = sum(analysis_results.get('missing_values', {}).values()) if analysis_results.get('missing_values') else 0
            duplicates = analysis_results.get('duplicates', 0)
            self._write_multiline(pdf, f"- Rows: {shape[0]:,}", line_height=7)
            self._write_multiline(pdf, f"- Columns: {shape[1]}", line_height=7)
            self._write_multiline(pdf, f"- Missing Values (total): {missing_total:,}", line_height=7)
            self._write_multiline(pdf, f"- Duplicate Rows: {duplicates}", line_height=7)

            # Numeric summary
            pdf.ln(4)
            pdf.set_font(DEFAULT_FONT, 'B', 14)
            self._write_multiline(pdf, "Numeric Summary (first few columns)", line_height=9)
            pdf.ln(2)
            pdf.set_font(DEFAULT_FONT, size=10)
            numeric_cols = analysis_results.get('numeric_columns', [])[:6]
            numeric_stats = analysis_results.get('numeric_stats', {}) or {}
            if numeric_cols:
                for col in numeric_cols:
                    stats = numeric_stats.get(col, {})
                    # stats might be dict-like; safely fetch values
                    cnt = stats.get('count', '')
                    mn = stats.get('min', '')
                    mx = stats.get('max', '')
                    mean = stats.get('mean', '')
                    std = stats.get('std', '')
                    line = f"{col}: count={cnt}, min={mn}, max={mx}, mean={mean}, std={std}"
                    self._write_multiline(pdf, line, line_height=6.5)
            else:
                self._write_multiline(pdf, "No numeric columns detected.", line_height=7)

            # Categorical summary
            pdf.ln(4)
            pdf.set_font(DEFAULT_FONT, 'B', 14)
            self._write_multiline(pdf, "Categorical Summary (first few columns)", line_height=9)
            pdf.ln(2)
            pdf.set_font(DEFAULT_FONT, size=10)
            categorical_cols = analysis_results.get('categorical_columns', [])[:6]
            categorical_stats = analysis_results.get('categorical_stats', {}) or {}
            if categorical_cols:
                for col in categorical_cols:
                    stats = categorical_stats.get(col, {})
                    unique = stats.get('unique_count', '')
                    top = stats.get('top_categories', {})
                    top_str = ", ".join([f"{k} ({v})" for k, v in list(top.items())[:5]])
                    line = f"{col}: unique={unique}; top: {top_str}"
                    self._write_multiline(pdf, line, line_height=6.5)
            else:
                self._write_multiline(pdf, "No categorical columns detected.", line_height=7)

            # Recommendations & Conclusion
            pdf.ln(6)
            pdf.set_font(DEFAULT_FONT, 'B', 14)
            self._write_multiline(pdf, "Recommendations & Conclusion", line_height=9)
            pdf.ln(3)
            recs = [
                "Address columns with high missing values (imputation or removal).",
                "Investigate and remove/resolve duplicate rows if they are erroneous.",
                "Explore strong correlations for feature engineering or multicollinearity.",
                "Handle outliers and validate distributions of numeric variables.",
                "Consider collecting more data if the dataset is small (<100 records)."
            ]
            pdf.set_font(DEFAULT_FONT, size=10)
            for i, r in enumerate(recs, 1):
                self._write_multiline(pdf, f"{i}. {self.clean_text(r)}", line_height=6.5)

            # Output PDF to bytes (fpdf2: dest='S' returns bytes; older fpdf may return str)
            out = pdf.output(dest='S')  # fpdf/fpdf2 may return str, bytes, bytearray, memoryview

            # Normalize to bytes robustly
            if isinstance(out, str):
                pdf_bytes = out.encode('latin-1', errors='replace')
            elif isinstance(out, bytearray):
                pdf_bytes = bytes(out)
            elif isinstance(out, memoryview):
                pdf_bytes = out.tobytes()
            elif isinstance(out, bytes):
                pdf_bytes = out
            else:
                # fallback
                pdf_bytes = str(out).encode('latin-1', errors='replace')

            return pdf_bytes

        except Exception as e:
            print("Error generating PDF:", e)
            traceback.print_exc()
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
                        # generate insights based on analysis results (simple)
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
                        # safe rerun for modern streamlit
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
            st.info("Interactive visualizations disabled in this patch demo (keeps focus on robust PDF generation).")

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
                    except Exception as e:
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
                # regenerate insights
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
