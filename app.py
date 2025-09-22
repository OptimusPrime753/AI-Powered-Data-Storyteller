# Test the data analyzer
from data_analyzer import DataAnalyzer
from report_generator import ReportGenerator

# Test with sample data
analyzer = DataAnalyzer()
analyzer.load_data("sample_data.csv")  # Use any CSV you have

insights = analyzer.basic_analysis()
plots = analyzer.generate_visualizations()

# Generate report
report_gen = ReportGenerator()
report_path = report_gen.generate_report(insights, plots)

print("Analysis complete! Report saved as:", report_path)