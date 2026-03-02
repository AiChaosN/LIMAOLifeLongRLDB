import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gnto_metrics(csv_file):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found.")
        return
        
    df = pd.read_csv(csv_file)
    
    # Filter valid data
    # predicted_cost might be None or negative?
    # actual_latency should be positive.
    
    # Check for missing values
    df = df.dropna(subset=['actual_latency', 'predicted_cost'])
    
    # Filter out non-positive values for log scale
    valid_df = df[(df['actual_latency'] > 0) & (df['predicted_cost'] > 0)].copy()
    
    if len(valid_df) < len(df):
        print(f"Warning: {len(df) - len(valid_df)} data points removed due to non-positive or missing values.")

    actual = valid_df['actual_latency']
    predicted = valid_df['predicted_cost']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(actual, predicted, alpha=0.6, edgecolors='w', s=50, c='#d62728') # Red for GNTO
    
    # Log scale
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Actual Latency (s) [Log Scale]')
    plt.ylabel('Predicted Cost (GNTO Reward) [Log Scale]')
    plt.title('GNTO: Predicted Cost vs Actual Latency')
    
    # Add grid
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Calculate correlations
    if len(valid_df) > 1:
        pearson_corr = valid_df['actual_latency'].corr(valid_df['predicted_cost'], method='pearson')
        spearman_corr = valid_df['actual_latency'].corr(valid_df['predicted_cost'], method='spearman')
        
        valid_df['log_actual'] = np.log(valid_df['actual_latency'])
        valid_df['log_predicted'] = np.log(valid_df['predicted_cost'])
        log_pearson_corr = valid_df['log_actual'].corr(valid_df['log_predicted'], method='pearson')
        
        stats_text = (
            f"Pearson Correlation: {pearson_corr:.4f}\n"
            f"Spearman Rank Correlation: {spearman_corr:.4f}\n"
            f"Log-Log Pearson Correlation: {log_pearson_corr:.4f}"
        )
        
        # Add text box
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        print("-" * 30)
        print(stats_text)
        print("-" * 30)

    plt.tight_layout()
    
    output_path = 'gnto_cost_vs_latency_scatter.png'
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    plot_gnto_metrics("gnto_predictions.csv")
