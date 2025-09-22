import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

class DataAnalyzer:
    def __init__(self):
        self.df = None
        # Use a simple LLM for insights
        self.llm = pipeline("text-generation", model="gpt2")
    
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        return self.df.head()
    
    def basic_analysis(self):
        insights = []
        insights.append(f"Dataset shape: {self.df.shape}")
        insights.append(f"Columns: {list(self.df.columns)}")
        insights.append(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Generate simple insights using LLM
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary = self.df[numeric_cols].describe().to_string()
            prompt = f"Summarize this data summary in 2 sentences:\n{summary}"
            llm_insight = self.llm(prompt, max_length=100)[0]['generated_text']
            insights.append(f"LLM Insight: {llm_insight}")
        
        return insights
    
    def generate_visualizations(self):
        plots = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Plot 1: Correlation Heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.df[numeric_cols].corr(), annot=True)
            plt.title("Correlation Heatmap")
            plots.append("correlation_heatmap.png")
            plt.savefig("correlation_heatmap.png")
            plt.close()
        
        # Plot 2: Distribution of first numeric column
        if len(numeric_cols) > 0:
            plt.figure(figsize=(10, 6))
            self.df[numeric_cols[0]].hist()
            plt.title(f"Distribution of {numeric_cols[0]}")
            plots.append("distribution_plot.png")
            plt.savefig("distribution_plot.png")
            plt.close()
        
        # Plot 3: Bar plot for categorical data
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            plt.figure(figsize=(10, 6))
            self.df[categorical_cols[0]].value_counts().head(10).plot(kind='bar')
            plt.title(f"Top 10 values in {categorical_cols[0]}")
            plots.append("bar_plot.png")
            plt.savefig("bar_plot.png")
            plt.close()
        
        return plots