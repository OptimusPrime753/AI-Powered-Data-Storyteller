import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import tempfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="AI-Powered Data Insights Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
        try:
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            else:
                self.df = pd.read_excel(uploaded_file)
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def perform_comprehensive_analysis(self):
        if self.df is None:
            return {}
        results = {}
        results['shape'] = self.df.shape
        results['columns'] = list(self.df.columns)
        results['data_types'] = self.df.dtypes.to_dict()
        results['missing_values'] = self.df.isnull().sum().to_dict()
        results['missing_percentage'] = (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        results['duplicates'] = self.df.duplicated().sum()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results['numeric_columns'] = list(numeric_cols)
        results['numeric_stats'] = self.df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        results['categorical_columns'] = list(categorical_cols)
        results['categorical_stats'] = {}
        for col in categorical_cols:
            results['categorical_stats'][col] = {
                'unique_count': self.df[col].nunique(),
                'top_categories': self.df[col].value_counts().head(5).to_dict()
            }
        if len(numeric_cols) > 1:
            results['correlation_matrix'] = self.df[numeric_cols].corr()
        return results

    def generate_ai_insights(self, analysis_results):
        insights = []
        insights.append(f"Dataset Overview: The dataset contains {analysis_results['shape'][0]:,} rows and {analysis_results['shape'][1]} columns.")
        total_missing = sum(analysis_results['missing_values'].values())
        if total_missing > 0:
            insights.append(f"Data Quality: {total_missing:,} missing values detected across the dataset.")
        if analysis_results['numeric_columns']:
            numeric_insight = f"Numeric Analysis: {len(analysis_results['numeric_columns'])} numeric columns found."
            if len(analysis_results['numeric_columns']) > 0:
                col = analysis_results['numeric_columns'][0]
                stats = analysis_results['numeric_stats'].get(col, {})
                if 'mean' in stats:
                    numeric_insight += f" {col} has mean {stats['mean']:.2f} and std {stats['std']:.2f}."
            insights.append(numeric_insight)
        if analysis_results['categorical_columns']:
            cat_insight = f"Categorical Analysis: {len(analysis_results['categorical_columns'])} categorical columns identified."
            insights.append(cat_insight)
        if 'correlation_matrix' in analysis_results:
            corr_matrix = analysis_results['correlation_matrix']
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append(
                            f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]} ({corr_matrix.iloc[i, j]:.2f})"
                        )
            if high_corr_pairs:
                insights.append(f"Strong Correlations: Found between {', '.join(high_corr_pairs[:3])}")
        if analysis_results['duplicates'] > 0:
            insights.append(f"Data Quality: {analysis_results['duplicates']} duplicate rows found.")
        return insights

class InteractiveVisualizations:
    def __init__(self, df):
        self.df = df
        self.color = '#1f77b4'

    def create_correlation_heatmap(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        corr_matrix = self.df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale=px.colors.diverging.RdBu,
            origin='lower',
            labels=dict(color='Correlation'),
            title="Correlation Heatmap"
        )
        fig.update_layout(
            height=600,
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(family="Arial", size=14),
            coloraxis_colorbar=dict(
                title="Correlation",
                thickness=15,
                tickvals=[-1, 0, 1],
                ticktext=["-1", "0", "1"],
            )
        )
        return fig

    def create_distribution_plots(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None
        cols = min(2, len(numeric_cols))
        rows = (len(numeric_cols) + cols - 1) // cols
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Distribution of {col}" for col in numeric_cols]
        )
        for i, col in enumerate(numeric_cols):
            row = (i // cols) + 1
            col_num = (i % cols) + 1
            fig.add_trace(
                go.Histogram(
                    x=self.df[col],
                    name=col,
                    nbinsx=30,
                    marker_color=self.color,
                    opacity=0.75,
                    hovertemplate=f"{col}: "+"%{x}<br>Count: %{y}<extra></extra>"
                ),
                row=row, col=col_num
            )
        fig.update_layout(
            height=rows * 350,
            title_text="Distribution Analysis",
            showlegend=False,
            font=dict(family='Arial', size=12),
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white'
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        return fig

    def create_scatter_matrix(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        selected_cols = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
        fig = px.scatter_matrix(
            self.df[selected_cols],
            height=800,
            title="Scatter Matrix",
            dimensions=selected_cols,
            color_discrete_sequence=[self.color]*len(selected_cols)
        )
        fig.update_layout(
            font=dict(family='Arial', size=12),
            margin=dict(l=50, r=50, t=60, b=50),
            dragmode='select'
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        return fig

    def create_categorical_analysis(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return None
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        fig = px.box(
            self.df,
            x=cat_col,
            y=num_col,
            title=f"{num_col} by {cat_col}",
            color_discrete_sequence=[self.color],
            notched=True
        )
        fig.update_layout(
            height=500,
            font=dict(family='Arial', size=12),
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white'
        )
        fig.update_xaxes(title=cat_col)
        fig.update_yaxes(title=num_col, showgrid=True, gridwidth=1, gridcolor='lightgrey')
        return fig

class ReportGenerator(FPDF):
    def sanitize_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        replacements = {
            '•': '-', '📊': '', '⚠️': '', '🔢': '', '📝': '', '📈': '', '🔍': '', '🚀': '',
            '✅': '', '📁': '', '📄': '', '🎯': '', '🤖': '', '📥': '', '🔧': ''
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = ''.join([c if ord(c) < 256 else '?' for c in text])
        return text

    def header(self):
        self.set_font('Arial', 'B', 18)
        self.set_fill_color(40, 86, 151)
        self.set_text_color(255,255,255)
        self.cell(0, 18, "AI Data Analysis Report", ln=True, align='C', fill=True)
        self.ln(2)
        self.set_text_color(0,0,0)
        self.set_draw_color(40, 86, 151)
        self.rect(7, 7, 196, 283)
        self.ln(3)

    def footer(self):
        self.set_y(-18)
        self.set_draw_color(40, 86, 151)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font('Arial', 'I', 8)
        self.set_text_color(40, 86, 151)
        self.cell(0, 12, f'Page {self.page_no()}', align='C')

    def section_header(self, title):
        self.set_fill_color(40, 86, 151)
        self.set_text_color(255,255,255)
        self.set_font("Arial", 'B', 15)
        self.cell(0, 12, self.sanitize_text(title), ln=1, fill=True)
        self.set_text_color(0,0,0)
        self.set_font('Arial', '', 12)
        self.ln(2)

    def add_key_metrics(self, analysis_results):
        metrics = [
            ("Rows", f"{analysis_results['shape'][0]:,}"),
            ("Columns", f"{analysis_results['shape'][1]}"),
            ("Missing Values", f"{sum(analysis_results['missing_values'].values()):,}"),
            ("Duplicate Rows", f"{analysis_results['duplicates']}")
        ]
        self.set_fill_color(211, 226, 241)
        self.ln(2)
        for title, value in metrics:
            self.set_font('Arial', 'B', 12)
            self.cell(60, 18, self.sanitize_text(title), align='C', fill=True)
            self.set_font('Arial', 'B', 16)
            self.set_text_color(40, 86, 151)
            self.cell(35, 18, self.sanitize_text(value), align='C', fill=True)
            self.set_text_color(0,0,0)
            self.ln(20)
        self.ln(2)
        self.set_font("Arial", '', 12)

    def add_table(self, headers, rows, wrap_last_column=False):
        available_w = self.w - 30
        if wrap_last_column:
            col_widths = [available_w * 0.20, available_w * 0.20, available_w * 0.60]
        else:
            col_widths = [available_w / len(headers)] * len(headers)
        th_bg = (40,86,151)
        th_text = (255,255,255)
        td_bg = (245, 249, 255)
        td_alt_bg = (230,237,247)
        self.set_font("Arial", 'B', 11)
        self.set_fill_color(*th_bg)
        self.set_text_color(*th_text)
        for k, h in enumerate(headers):
            self.cell(col_widths[k], 10, self.sanitize_text(h), border=1, align='C', fill=True)
        self.ln()
        self.set_font("Arial", '', 10)
        self.set_text_color(0,0,0)
        for i, row in enumerate(rows):
            self.set_fill_color(*[td_bg if i%2==0 else td_alt_bg][0])
            if wrap_last_column:
                self.cell(col_widths[0], 9, self.sanitize_text(str(row[0])), border=1, align='C', fill=True)
                self.cell(col_widths[1], 9, self.sanitize_text(str(row[1])), border=1, align='C', fill=True)
                x_current = self.get_x()
                y_current = self.get_y()
                self.multi_cell(col_widths[2], 9, self.sanitize_text(str(row[2])), border=1, align='L', fill=True)
                self.set_xy(10, max(self.get_y(), y_current + 9))
            else:
                for k, cell in enumerate(row):
                    self.cell(col_widths[k], 9, self.sanitize_text(str(cell)), border=1, align='C', fill=True)
                self.ln()

    def generate_pdf_report(self, analysis_results, insights, output_file="ai_analysis_report.pdf"):
        self.add_page()
        self.ln(2)
        self.set_font("Arial", 'I', 10)
        self.set_fill_color(255,255,255)
        self.cell(0, 8, self.sanitize_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=1)
        self.ln(5)
        self.section_header("Executive Summary")
        for insight in insights[:3]:
            clean_insight = self.sanitize_text(insight.replace('**', ''))
            self.set_fill_color(245, 255, 245)
            self.multi_cell(0, 8, clean_insight, fill=True)
            self.ln(2)
        self.section_header("Dataset Overview")
        self.add_key_metrics(analysis_results)
        self.section_header("Data Types")
        dtype_headers = ['Column', 'Type']
        dtype_rows = [[col, str(analysis_results['data_types'][col])] for col in analysis_results['columns']]
        self.add_table(dtype_headers, dtype_rows)
        self.ln(4)
        self.section_header("Missing Values")
        missing_headers = ['Column', 'Missing Count', 'Missing %']
        missing_rows = [
            [col, analysis_results['missing_values'][col], f"{analysis_results['missing_percentage'][col]:.2f}%"]
            for col in analysis_results['columns']
        ]
        self.add_table(missing_headers, missing_rows)
        self.ln(4)
        if analysis_results['numeric_columns']:
            self.section_header("Numeric Statistics")
            num_stats = analysis_results['numeric_stats']
            headers = list(num_stats.keys())
            stats_rows = []
            stat_types = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            for stat in stat_types:
                row = [stat] + [f"{num_stats[h][stat]:.2f}" if stat in num_stats[h] else "-" for h in headers]
                stats_rows.append(row)
            self.add_table(['Stat'] + headers, stats_rows)
            self.ln(4)
        if analysis_results['categorical_columns']:
            self.section_header("Categorical Columns")
            cat_headers = ['Column', 'Unique Count', 'Top Categories']
            cat_rows = []
            for col in analysis_results['categorical_columns'][:5]:
                top_cats = analysis_results['categorical_stats'][col]['top_categories']
                cats_str = ", ".join([f"{k}({v})" for k, v in top_cats.items()])
                cat_rows.append([col, analysis_results['categorical_stats'][col]['unique_count'], cats_str])
            self.add_table(cat_headers, cat_rows, wrap_last_column=True)
            self.ln(4)
        self.section_header("AI Insights (Full)")
        for insight in insights:
            self.set_fill_color(245, 255, 245)
            self.multi_cell(0, 8, self.sanitize_text(insight.replace('**', '')), fill=True)
            self.ln(2)
        self.output(output_file)
        return output_file

def main():
    st.markdown('<div class="main-header">🤖 AI-Powered Data Insights Hub</div>', unsafe_allow_html=True)
    st.markdown("Upload your dataset to unlock powerful AI-driven insights and interactive visualizations")
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            success, message = st.session_state.analyzer.load_data(uploaded_file)
            if success:
                st.success("✅ " + message)
                if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                    with st.spinner("Performing comprehensive analysis..."):
                        st.session_state.analysis_results = st.session_state.analyzer.perform_comprehensive_analysis()
                        st.session_state.insights = st.session_state.analyzer.generate_ai_insights(st.session_state.analysis_results)
                        st.session_state.analysis_done = True
                        st.rerun()  # <--- THIS IS THE FIX!
            else:
                st.error("❌ " + message)
        st.markdown("---")
        st.header("🔧 Settings")
        st.selectbox("Theme", ["Light", "Dark"], key="theme")
        st.slider("Chart Quality", 1, 10, 8, key="chart_quality")
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
                    'Data Type': [str(st.session_state.analysis_results['data_types'][col]) for col in st.session_state.analysis_results['columns']]
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
                if st.session_state.analysis_results['numeric_columns']:
                    st.subheader("Numeric Statistics")
                    numeric_stats = pd.DataFrame(st.session_state.analysis_results['numeric_stats'])
                    st.dataframe(numeric_stats, use_container_width=True)
            with col2:
                if st.session_state.analysis_results['categorical_columns']:
                    st.subheader("Categorical Analysis")
                    for col in st.session_state.analysis_results['categorical_columns'][:3]:
                        st.write(f"{col}: {st.session_state.analysis_results['categorical_stats'][col]['unique_count']} unique values")
                        top_cats = st.session_state.analysis_results['categorical_stats'][col]['top_categories']
                        st.write("Top categories:", ", ".join([f"{k} ({v})" for k, v in list(top_cats.items())[:3]]))
        with tab3:
            st.header("Interactive Visualizations")
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
            st.header("Report Generation")
            st.subheader("Executive Summary")
            st.write("Generate a comprehensive PDF report with all analysis findings.")
            report_gen = ReportGenerator()
            if st.button("Generate PDF Report", type="primary"):
                with st.spinner("Generating report..."):
                    report_path = report_gen.generate_pdf_report(
                        st.session_state.analysis_results, 
                        st.session_state.insights
                    )
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="Download AI Analysis Report",
                            data=file,
                            file_name="ai_data_analysis_report.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
            st.subheader("Report Preview")
            st.write("Key sections included in the report:")
            st.write("Executive Summary")
            st.write("Dataset Overview")
            st.write("AI Insights")
            st.write("Statistical Analysis")
            st.write("Recommendations")
    else:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>Welcome to AI-Powered Data Insights Hub</h2>
            <p>Upload your dataset to get started with AI-powered analysis!</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Try with Sample Data", use_container_width=True):
                sample_data = {
                    'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
                    'Sales': np.random.normal(1000, 200, 100).cumsum(),
                    'Customers': np.random.poisson(50, 100),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
                    'Revenue': np.random.exponential(500, 100)
                }
                sample_df = pd.DataFrame(sample_data)
                sample_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                sample_df.to_csv(sample_file.name, index=False)
                st.session_state.analyzer.df = sample_df
                st.session_state.analysis_results = st.session_state.analyzer.perform_comprehensive_analysis()
                st.session_state.insights = st.session_state.analyzer.generate_ai_insights(st.session_state.analysis_results)
                st.session_state.analysis_done = True
                st.rerun()  # <--- THIS IS THE FIX!

if __name__ == "__main__":
    main()
