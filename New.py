# corrected_ai_data_storyteller.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
import io
import re

# Page configuration
st.set_page_config(
    page_title="AI Data Storyteller",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .tab-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 10px 0;
    }
    .checkbox-container {
        display: flex;
        align-items: flex-start;
        margin: 10px 0;
    }
    .checkbox-col {
        width: 40px;
        padding-top: 15px;
    }
    .insight-col {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)


class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.analysis_results = {}

    def load_data(self, uploaded_file):
        """Load data from uploaded file, try to infer dates where possible"""
        try:
            filename = uploaded_file.name.lower()
            if filename.endswith('.csv'):
                # Read CSV (no guarantee about dates — we'll try to parse later)
                self.df = pd.read_csv(uploaded_file)
            else:
                # Excel
                self.df = pd.read_excel(uploaded_file)
            # Try to convert object columns that look like dates
            for col in self.df.select_dtypes(include=['object']).columns:
                try:
                    converted = pd.to_datetime(self.df[col], errors='coerce', infer_datetime_format=True)
                    # If a reasonable fraction converted, keep it as datetime
                    if converted.notnull().sum() / max(1, len(converted)) > 0.5:
                        self.df[col] = converted
                except Exception:
                    pass
            return True, "Data loaded successfully!"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def perform_comprehensive_analysis(self):
        """Perform comprehensive EDA"""
        if self.df is None:
            return {}

        results = {}

        # Basic statistics
        results['shape'] = self.df.shape
        results['columns'] = list(self.df.columns)
        results['data_types'] = self.df.dtypes.astype(str).to_dict()
        results['missing_values'] = self.df.isnull().sum().to_dict()
        results['missing_percentage'] = (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        results['duplicates'] = int(self.df.duplicated().sum())

        # Numeric columns analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        results['numeric_columns'] = numeric_cols
        # produce a per-column stats mapping for easy consumption later
        if len(numeric_cols) > 0:
            results['numeric_stats'] = self.df[numeric_cols].describe().transpose().to_dict()  # column -> {count, mean, std,...}
        else:
            results['numeric_stats'] = {}

        # Categorical columns analysis
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        results['categorical_columns'] = categorical_cols
        results['categorical_stats'] = {}
        for col in categorical_cols:
            vc = self.df[col].value_counts(dropna=False)
            results['categorical_stats'][col] = {
                'unique_count': int(self.df[col].nunique(dropna=False)),
                'top_categories': vc.head(5).to_dict()
            }

        # Correlation analysis + strong correlations
        if len(numeric_cols) > 1:
            corr = self.df[numeric_cols].corr()
            results['correlation_matrix'] = corr
            strong_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if pd.notnull(val) and abs(val) > 0.7:
                        strong_pairs.append({
                            'var1': corr.columns[i],
                            'var2': corr.columns[j],
                            'correlation': float(val)
                        })
            results['strong_correlations'] = strong_pairs
        else:
            results['correlation_matrix'] = None
            results['strong_correlations'] = []

        return results

    def generate_ai_insights(self, analysis_results):
        """Generate AI-like insights based on data analysis"""
        insights = []

        if not analysis_results:
            return insights

        # Dataset overview insights
        rows = analysis_results.get('shape', (0, 0))[0]
        cols = analysis_results.get('shape', (0, 0))[1]
        insights.append(f"📊 **Dataset Overview**: The dataset contains {rows:,} rows and {cols} columns.")

        # Missing values insights
        total_missing = sum(analysis_results.get('missing_values', {}).values())
        if total_missing > 0:
            insights.append(f"⚠️ **Data Quality**: {total_missing:,} missing values detected across the dataset.")

        # Numeric columns insights
        numeric_cols = analysis_results.get('numeric_columns', [])
        if numeric_cols:
            numeric_insight = f"🔢 **Numeric Analysis**: {len(numeric_cols)} numeric columns found."
            # pick first numeric column and show mean/std if present
            col = numeric_cols[0]
            stats = analysis_results.get('numeric_stats', {}).get(col, {})
            mean = stats.get('mean')
            std = stats.get('std')
            if mean is not None and std is not None:
                try:
                    numeric_insight += f" {col} has mean {mean:.2f} and std {std:.2f}."
                except Exception:
                    numeric_insight += f" {col} has mean {mean} and std {std}."
            insights.append(numeric_insight)

        # Categorical insights
        categorical_cols = analysis_results.get('categorical_columns', [])
        if categorical_cols:
            cat_insight = f"📝 **Categorical Analysis**: {len(categorical_cols)} categorical columns identified."
            insights.append(cat_insight)

        # Correlation insights
        strong_corrs = analysis_results.get('strong_correlations', [])
        if strong_corrs:
            top_pairs = [f"{p['var1']} & {p['var2']} ({p['correlation']:.2f})" for p in strong_corrs[:3]]
            insights.append(f"📈 **Strong Correlations**: Found between {', '.join(top_pairs)}")

        # Data quality insights (duplicates)
        if analysis_results.get('duplicates', 0) > 0:
            insights.append(f"🔍 **Data Quality**: {analysis_results['duplicates']} duplicate rows found.")

        return insights


class InteractiveVisualizations:
    def __init__(self, df):
        self.df = df

    def create_correlation_heatmap(self):
        """Create interactive correlation heatmap"""
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
        """Create interactive distribution plots"""
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
                go.Histogram(x=self.df[col], name=col, nbinsx=30),
                row=row, col=col_num
            )

        fig.update_layout(height=300*rows, title_text="Distribution Analysis", showlegend=False)
        return fig

    def create_scatter_matrix(self):
        """Create interactive scatter matrix"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None

        # Take first 4 numeric columns for scatter matrix
        selected_cols = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
        fig = px.scatter_matrix(
            self.df[selected_cols],
            title="Scatter Matrix",
            height=800
        )
        return fig

    def create_time_series_plot(self):
        """Create time series plot if date column exists"""
        date_columns = self.df.select_dtypes(include=['datetime64']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(date_columns) > 0 and len(numeric_cols) > 0:
            date_col = date_columns[0]
            numeric_col = numeric_cols[0]

            # Ensure date is sorted
            df_sorted = self.df.sort_values(by=date_col)
            fig = px.line(
                df_sorted,
                x=date_col,
                y=numeric_col,
                title=f"{numeric_col} over Time",
                height=400
            )
            return fig
        return None

    def create_categorical_analysis(self):
        """Create categorical data visualizations"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]

            fig = px.box(
                self.df,
                x=cat_col,
                y=num_col,
                title=f"{num_col} by {cat_col}",
                height=500
            )
            return fig
        return None


class ReportGenerator:
    def clean_text(self, text):
        """Remove problematic unicode and markdown markers"""
        # Remove all non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Remove bold markdown markers
        text = text.replace('**', '')
        return text

    def save_plot_to_image(self, plt_fig, filename):
        """Save matplotlib figure to image file"""
        try:
            plt_fig.savefig(filename, bbox_inches='tight', dpi=150, format='PNG')
            return True
        except Exception as e:
            print(f"Error saving plot: {e}")
            return False

    def generate_charts(self, df, analysis_results, temp_dir):
        """Generate charts for the PDF report"""
        charts_generated = []

        # Set style for better PDF rendering
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

        numeric_cols = analysis_results.get('numeric_columns', [])
        categorical_cols = analysis_results.get('categorical_columns', [])

        try:
            # Chart 1: Correlation Heatmap
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                            fmt='.2f', ax=ax, cbar_kws={'shrink': 0.8})
                ax.set_title('Correlation Heatmap of Numeric Variables', pad=20, fontsize=14, fontweight='bold')
                plt.tight_layout()
                chart_path = os.path.join(temp_dir, 'correlation_heatmap.png')
                if self.save_plot_to_image(fig, chart_path):
                    charts_generated.append(('correlation_heatmap.png', 'Correlation Heatmap'))
                plt.close(fig)

            # Chart 2: Distribution of first numeric column
            if len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[numeric_cols[0]].hist(bins=20, edgecolor='black', alpha=0.7, ax=ax)
                ax.set_title(f'Distribution of {numeric_cols[0]}', fontsize=14, fontweight='bold')
                ax.set_xlabel(numeric_cols[0], fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                chart_path = os.path.join(temp_dir, 'distribution.png')
                if self.save_plot_to_image(fig, chart_path):
                    charts_generated.append(('distribution.png', f'Distribution of {numeric_cols[0]}'))
                plt.close(fig)

            # Chart 3: Top categories for first categorical column
            if len(categorical_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                top_categories = df[categorical_cols[0]].value_counts().head(10)
                colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
                top_categories.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
                ax.set_title(f'Top 10 Categories in {categorical_cols[0]}', fontsize=14, fontweight='bold')
                ax.set_xlabel(categorical_cols[0], fontweight='bold')
                ax.set_ylabel('Count', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                chart_path = os.path.join(temp_dir, 'categories.png')
                if self.save_plot_to_image(fig, chart_path):
                    charts_generated.append(('categories.png', f'Top Categories in {categorical_cols[0]}'))
                plt.close(fig)

            # Chart 4: Missing values heatmap
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_percentage = (missing_data / len(df)) * 100
                missing_percentage[missing_percentage > 0].plot(kind='bar', ax=ax, alpha=0.7)
                ax.set_title('Missing Values by Column (%)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Columns', fontweight='bold')
                ax.set_ylabel('Missing Percentage', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                chart_path = os.path.join(temp_dir, 'missing_values.png')
                if self.save_plot_to_image(fig, chart_path):
                    charts_generated.append(('missing_values.png', 'Missing Values Analysis'))
                plt.close(fig)

            # Chart 5: Box plot for numeric variables
            if len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[numeric_cols].boxplot(ax=ax)
                ax.set_title('Distribution of Numeric Variables', fontsize=14, fontweight='bold')
                ax.set_ylabel('Values', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                chart_path = os.path.join(temp_dir, 'boxplot.png')
                if self.save_plot_to_image(fig, chart_path):
                    charts_generated.append(('boxplot.png', 'Box Plot of Numeric Variables'))
                plt.close(fig)

        except Exception as e:
            print(f"Error generating charts: {e}")

        return charts_generated

    def generate_insights_and_recommendations(self, analysis_results, df):
        """Generate detailed insights and recommendations"""
        insights = []
        recommendations = []

        if not analysis_results:
            return insights, recommendations

        # Data Quality Insights
        total_missing = sum(analysis_results.get('missing_values', {}).values())
        total_cells = analysis_results.get('shape', (0, 0))[0] * analysis_results.get('shape', (0, 0))[1]
        completeness = (1 - (total_missing / max(1, total_cells))) * 100

        insights.append(f"Data Quality: Dataset has {completeness:.1f}% completeness score")
        if completeness < 90:
            recommendations.append("Address missing values through imputation or data collection")

        if analysis_results.get('duplicates', 0) > 0:
            insights.append(f"Data Cleaning: Found {analysis_results['duplicates']} duplicate records")
            recommendations.append("Review and remove duplicate entries to ensure data integrity")

        # Numeric Columns Insights
        if analysis_results.get('numeric_columns'):
            num_col = analysis_results['numeric_columns'][0]
            stats = analysis_results.get('numeric_stats', {}).get(num_col, {})
            if stats:
                insights.append(f"Numeric Analysis: {num_col} ranges from {stats.get('min', 0):.2f} to {stats.get('max', 0):.2f} (mean: {stats.get('mean', 0):.2f})")

            # Check for outliers
            try:
                q1 = df[num_col].quantile(0.25)
                q3 = df[num_col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[num_col] < q1 - 1.5 * iqr) | (df[num_col] > q3 + 1.5 * iqr)]
                if len(outliers) > 0:
                    insights.append(f"Outlier Detection: Found {len(outliers)} potential outliers in {num_col}")
                    recommendations.append("Investigate potential outliers for data quality issues")
            except Exception:
                pass

        # Correlation Insights
        if analysis_results.get('strong_correlations'):
            strongest = analysis_results['strong_correlations'][0]
            insights.append(f"Correlation: Strong relationship between {strongest['var1']} and {strongest['var2']} (r={strongest['correlation']:.3f})")
            recommendations.append(f"Explore the relationship between {strongest['var1']} and {strongest['var2']} for business insights")

        # Categorical Insights
        if analysis_results.get('categorical_columns'):
            cat_col = analysis_results['categorical_columns'][0]
            unique_count = analysis_results['categorical_stats'][cat_col]['unique_count']
            top_cats = analysis_results['categorical_stats'][cat_col]['top_categories']
            top_category = list(top_cats.keys())[0] if top_cats else None
            insights.append(f"Categorical: {cat_col} has {unique_count} unique values, dominant category: {top_category}")

            if unique_count > 20:
                recommendations.append(f"Consider grouping less frequent categories in {cat_col} for better analysis")

        # Dataset Size Insights
        nrows = analysis_results.get('shape', (0, 0))[0]
        if nrows > 10000:
            insights.append("Scale: Large dataset suitable for machine learning models")
            recommendations.append("Consider using advanced analytics and machine learning techniques")
        elif nrows < 100:
            insights.append("Scale: Small dataset - results should be interpreted with caution")
            recommendations.append("Consider collecting more data for robust analysis")

        return insights, recommendations

    def generate_pdf_report(self, analysis_results, insights, df, output_file="comprehensive_analysis_report.pdf"):
        """Generate comprehensive PDF report with charts and detailed analysis"""
        # Create temporary directory for charts
        temp_dir = tempfile.mkdtemp()

        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Page 1: Title Page
            pdf.add_page()
            pdf.set_font("Arial", 'B', 24)
            pdf.cell(200, 20, txt="Comprehensive Data Analysis Report", ln=True, align='C')
            pdf.ln(15)

            pdf.set_font("Arial", 'I', 14)
            pdf.cell(200, 10, txt="AI-Powered Data Storyteller", ln=True, align='C')
            pdf.ln(20)

            pdf.set_font("Arial", '', 12)
            pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
            pdf.ln(30)

            # Generate charts
            charts = self.generate_charts(df, analysis_results, temp_dir)

            # Page 2: Executive Summary
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(200, 10, txt="Executive Summary", ln=True)
            pdf.ln(10)

            pdf.set_font("Arial", '', 12)
            summary_text = f"""
This comprehensive analysis report provides insights from a dataset containing {analysis_results['shape'][0]:,} records 
and {analysis_results['shape'][1]} variables. The analysis covers data quality assessment, statistical summaries, 
correlation analysis, and visualization of key patterns and relationships within the data.
            """
            pdf.multi_cell(0, 8, txt=self.clean_text(summary_text))
            pdf.ln(10)

            # Key Findings
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Key Findings:", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", '', 11)
            for i, insight in enumerate(insights[:6], 1):
                clean_insight = self.clean_text(insight)
                pdf.multi_cell(0, 6, txt=f"• {clean_insight}")
                pdf.ln(1)

            # Page 3: Dataset Overview
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(200, 10, txt="Dataset Overview", ln=True)
            pdf.ln(10)

            # Basic Statistics Table
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(100, 8, txt="Metric", border=1)
            pdf.cell(100, 8, txt="Value", border=1, ln=True)

            pdf.set_font("Arial", '', 11)
            overview_data = [
                ("Total Records", f"{analysis_results['shape'][0]:,}"),
                ("Total Variables", f"{analysis_results['shape'][1]}"),
                ("Numeric Variables", f"{len(analysis_results['numeric_columns'])}"),
                ("Categorical Variables", f"{len(analysis_results['categorical_columns'])}"),
                ("Missing Values", f"{sum(analysis_results['missing_values'].values()):,}"),
                ("Duplicate Records", f"{analysis_results['duplicates']}"),
                ("Data Completeness", f"{(1 - (sum(analysis_results['missing_values'].values()) / (analysis_results['shape'][0] * analysis_results['shape'][1]))) * 100:.1f}%")
            ]

            for metric, value in overview_data:
                pdf.cell(100, 8, txt=metric, border=1)
                pdf.cell(100, 8, txt=value, border=1, ln=True)

            pdf.ln(10)

            # Data Types Summary
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Data Types Summary:", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", '', 11)
            for col in analysis_results['columns'][:10]:  # Show first 10 columns
                dtype = str(analysis_results['data_types'].get(col, ''))
                pdf.multi_cell(0, 6, txt=f"• {col}: {dtype}")
                pdf.ln(1)

            # Add Charts to PDF
            if charts:
                for chart_filename, chart_title in charts:
                    try:
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 16)
                        pdf.cell(200, 10, txt=chart_title, ln=True)
                        pdf.ln(5)

                        chart_path = os.path.join(temp_dir, chart_filename)
                        if os.path.exists(chart_path):
                            pdf.image(chart_path, x=10, y=30, w=190)
                            pdf.ln(100)

                            # Add chart description
                            pdf.set_font("Arial", 'I', 10)
                            if 'correlation' in chart_filename.lower():
                                desc = "Heatmap showing relationships between numeric variables. Red indicates positive correlation, blue indicates negative correlation."
                            elif 'distribution' in chart_filename.lower():
                                desc = "Histogram showing frequency distribution of the variable. Helps identify data distribution patterns."
                            elif 'categor' in chart_filename.lower():
                                desc = "Bar chart showing frequency of top categories. Useful for understanding categorical variable distribution."
                            elif 'missing' in chart_filename.lower():
                                desc = "Visualization of missing data patterns across variables. Helps identify data quality issues."
                            elif 'boxplot' in chart_filename.lower():
                                desc = "Box plot showing distribution, median, and outliers for numeric variables."
                            else:
                                desc = "Visual analysis of dataset patterns and relationships."

                            pdf.multi_cell(0, 6, txt=desc)

                    except Exception as e:
                        print(f"Error adding chart {chart_filename}: {e}")
                        continue

            # Page: Detailed Insights and Recommendations
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(200, 10, txt="Detailed Analysis & Recommendations", ln=True)
            pdf.ln(10)

            # Generate detailed insights
            detailed_insights, recommendations = self.generate_insights_and_recommendations(analysis_results, df)

            # Detailed Insights
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Detailed Insights:", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", '', 11)
            for insight in detailed_insights:
                pdf.multi_cell(0, 6, txt=f"• {self.clean_text(insight)}")
                pdf.ln(1)

            pdf.ln(5)

            # Recommendations
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="Actionable Recommendations:", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", '', 11)
            for i, recommendation in enumerate(recommendations, 1):
                pdf.multi_cell(0, 6, txt=f"{i}. {self.clean_text(recommendation)}")
                pdf.ln(1)

            # Conclusion Page
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(200, 10, txt="Conclusion", ln=True)
            pdf.ln(10)

            pdf.set_font("Arial", '', 12)
            conclusion_text = f"""
This comprehensive analysis provides valuable insights into the dataset's structure, quality, and patterns. 
The findings highlight key relationships, data quality considerations, and opportunities for further analysis.

Key takeaways include the identification of {len(analysis_results['numeric_columns'])} numeric and 
{len(analysis_results['categorical_columns'])} categorical variables, with overall data completeness of 
{(1 - (sum(analysis_results['missing_values'].values()) / (analysis_results['shape'][0] * analysis_results['shape'][1]))) * 100:.1f}%.
            """
            pdf.multi_cell(0, 8, txt=self.clean_text(conclusion_text))

            # Save PDF
            pdf.output(output_file)

            # Clean up temporary files
            for chart_filename, _ in charts:
                chart_path = os.path.join(temp_dir, chart_filename)
                if os.path.exists(chart_path):
                    try:
                        os.remove(chart_path)
                    except Exception:
                        pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass

            return output_file

        except Exception as e:
            print(f"Error generating PDF: {e}")
            # Fallback to simple text report
            try:
                txt_path = output_file.replace('.pdf', '.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("Comprehensive Data Analysis Report\n")
                    f.write("="*60 + "\n\n")
                    f.write("EXECUTIVE SUMMARY\n")
                    f.write("-"*20 + "\n")
                    for insight in insights:
                        f.write(f"- {self.clean_text(insight)}\n")

                    f.write("\nDETAILED ANALYSIS\n")
                    f.write("-"*20 + "\n")
                    detailed_insights, recommendations = self.generate_insights_and_recommendations(analysis_results, df)
                    for insight in detailed_insights:
                        f.write(f"- {insight}\n")

                    f.write("\nRECOMMENDATIONS\n")
                    f.write("-"*20 + "\n")
                    for rec in recommendations:
                        f.write(f"- {rec}\n")

                return txt_path
            except Exception:
                return None


def main():
    # Header
    st.markdown('<div class="main-header">🤖 AI-Powered Data Storyteller</div>', unsafe_allow_html=True)
    st.markdown("Upload your dataset to unlock powerful AI-driven insights and interactive visualizations")

    # Initialize session state
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
                if st.button("🚀 Start AI Analysis"):
                    with st.spinner("Performing comprehensive analysis..."):
                        st.session_state.analysis_results = st.session_state.analyzer.perform_comprehensive_analysis()
                        st.session_state.insights = st.session_state.analyzer.generate_ai_insights(st.session_state.analysis_results)
                        st.session_state.analysis_done = True
                        st.experimental_rerun()
            else:
                st.error("❌ " + message)

        st.markdown("---")
        st.header("🔧 Settings")
        st.selectbox("Theme", ["Light", "Dark"], key="theme")
        st.slider("Chart Quality", 1, 10, 8, key="chart_quality")

    # Main content area with tabs or welcome
    if st.session_state.analysis_done and st.session_state.analyzer.df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🤖 AI Insights", "📈 Visualizations", "📄 Report"])

        with tab1:
            st.header("📊 Dataset Overview")

            # Key metrics
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

            # Data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.analyzer.df.head(10), use_container_width=True)

            # Data types information
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

            # Display insights with checkboxes and blue bars
            for i, insight in enumerate(st.session_state.insights, 1):
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    default_value = i in [2, 5]  # Pre-check some insights
                    st.checkbox("", key=f"insight_{i}", value=default_value)
                with col2:
                    st.markdown(
                        f'<div style="background-color: #e8f4fd; padding: 12px; border-left: 4px solid #1f77b4; border-radius: 5px; margin: 5px 0;">{insight}</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("---")

            # Detailed analysis sections
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.analysis_results.get('numeric_columns'):
                    st.subheader("📈 Numeric Statistics")
                    numeric_stats = pd.DataFrame.from_dict(st.session_state.analysis_results['numeric_stats'], orient='index')
                    st.dataframe(numeric_stats, use_container_width=True)

            with col2:
                if st.session_state.analysis_results.get('categorical_columns'):
                    st.subheader("📝 Categorical Analysis")
                    for col in st.session_state.analysis_results['categorical_columns'][:3]:  # Show first 3
                        st.write(f"**{col}**: {st.session_state.analysis_results['categorical_stats'][col]['unique_count']} unique values")
                        top_cats = st.session_state.analysis_results['categorical_stats'][col]['top_categories']
                        st.write("Top categories:", ", ".join([f"{k} ({v})" for k, v in list(top_cats.items())[:3]]))

        with tab3:
            st.header("📈 Interactive Visualizations")

            # Initialize visualizer
            visualizer = InteractiveVisualizations(st.session_state.analyzer.df)

            # Visualization selection
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
            st.write("Generate a comprehensive PDF report with all analysis findings, charts, and recommendations.")

            report_gen = ReportGenerator()

            if st.button("📥 Generate Comprehensive PDF Report"):
                with st.spinner("Generating detailed report... This may take a few moments."):
                    if (st.session_state.analysis_results and
                        st.session_state.insights and
                        st.session_state.analyzer.df is not None):

                        report_path = report_gen.generate_pdf_report(
                            st.session_state.analysis_results,
                            st.session_state.insights,
                            st.session_state.analyzer.df
                        )

                        if report_path and report_path.endswith('.pdf') and os.path.exists(report_path):
                            with open(report_path, "rb") as file:
                                st.download_button(
                                    label="📄 Download AI Analysis Report (PDF)",
                                    data=file,
                                    file_name="comprehensive_data_analysis_report.pdf",
                                    mime="application/pdf"
                                )
                            st.success("✅ Comprehensive PDF report generated successfully!")

                            # Clean up the generated file after providing the button (file stays until process ends)
                            try:
                                os.remove(report_path)
                            except Exception:
                                pass

                        elif report_path and report_path.endswith('.txt') and os.path.exists(report_path):
                            with open(report_path, "r", encoding='utf-8') as file:
                                st.download_button(
                                    label="📄 Download Report (TXT)",
                                    data=file.read(),
                                    file_name="data_analysis_report.txt",
                                    mime="text/plain"
                                )
                            st.warning("⚠️ PDF generation failed. Text report generated instead.")
                        else:
                            st.error("❌ Report generation failed. Please check the data and try again.")
                    else:
                        st.error("❌ No data available for report generation. Please analyze data first.")
    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>🚀 Welcome to AI Data Storyteller</h2>
            <p>Upload your dataset to get started with AI-powered analysis!</p>
        </div>
        """, unsafe_allow_html=True)

        # Sample data option
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Try with Sample Data"):
                # Create sample data
                sample_data = {
                    'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
                    'Sales': np.random.normal(1000, 200, 100).cumsum(),
                    'Customers': np.random.poisson(50, 100),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
                    'Revenue': np.random.exponential(500, 100)
                }
                sample_df = pd.DataFrame(sample_data)

                # Load sample data into session state
                st.session_state.analyzer.df = sample_df
                st.session_state.analysis_results = st.session_state.analyzer.perform_comprehensive_analysis()
                st.session_state.insights = st.session_state.analyzer.generate_ai_insights(st.session_state.analysis_results)
                st.session_state.analysis_done = True
                st.experimental_rerun()


if __name__ == "__main__":
    main()
