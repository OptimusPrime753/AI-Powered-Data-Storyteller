from fpdf import FPDF
import os

class ReportGenerator:
    def __init__(self):
        self.pdf = FPDF()
    
    def generate_report(self, insights, plot_paths, output_file="report.pdf"):
        self.pdf.add_page()
        
        # Title
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(200, 10, txt="Data Analysis Executive Summary", ln=True, align='C')
        self.pdf.ln(10)
        
        # Insights
        self.pdf.set_font("Arial", size=12)
        for insight in insights:
            self.pdf.multi_cell(0, 10, txt=insight)
            self.pdf.ln(5)
        
        # Add plots
        for plot_path in plot_paths:
            if os.path.exists(plot_path):
                self.pdf.add_page()
                self.pdf.image(plot_path, x=10, y=10, w=180)
        
        self.pdf.output(output_file)
        return output_file