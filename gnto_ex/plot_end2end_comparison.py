import re
import os
os.environ['OMP_NUM_THREADS'] = '1'
import csv

def read_log_data(file_path):
    """
    Reads the log file and extracts the last execution time for each SQL query.
    Returns two dictionaries:
    - pg_times: {sql_filename: time}
    - model_times: {sql_filename: time}
    """
    pg_times = {}
    model_times = {}
    
    # Regex patterns
    # Matches: x x 22a_job.sql 5.89 PG
    pg_pattern = re.compile(r"x\s+x\s+(\S+)\s+([\d\.]+)\s+PG")
    # Matches: BAO 22a_job.sql 5.89 or GNTO 22a_job.sql 5.89
    model_pattern = re.compile(r"(?:BAO|GNTO)\s+(\S+)\s+([\d\.]+)")
    
    print(f"Reading {file_path}...")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check for PG pattern
                pg_match = pg_pattern.search(line)
                if pg_match:
                    sql = pg_match.group(1)
                    time_val = float(pg_match.group(2))
                    pg_times[sql] = time_val # Overwrite to keep the last one
                    continue
                
                # Check for Model pattern
                model_match = model_pattern.search(line)
                if model_match:
                    sql = model_match.group(1)
                    time_val = float(model_match.group(2))
                    model_times[sql] = time_val # Overwrite to keep the last one
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}, {}
        
    return pg_times, model_times

def filter_slow_queries(pg1_data, limao_data, pg2_data, gnto_data, threshold=30.0):
    """
    Filters out queries where any model execution time (LIMAO or GNTO) exceeds the threshold.
    """
    print(f"Filtering queries where model time > {threshold}s...")
    
    f_pg1 = {}
    f_limao = {}
    f_pg2 = {}
    f_gnto = {}
    
    all_queries = set(limao_data.keys()) | set(gnto_data.keys())
    
    excluded_count = 0
    
    for q in all_queries:
        t_limao = limao_data.get(q, 0)
        t_gnto = gnto_data.get(q, 0)
        
        if t_limao > threshold or t_gnto > threshold:
            excluded_count += 1
            continue
            
        if q in pg1_data: f_pg1[q] = pg1_data[q]
        if q in limao_data: f_limao[q] = limao_data[q]
        if q in pg2_data: f_pg2[q] = pg2_data[q]
        if q in gnto_data: f_gnto[q] = gnto_data[q]
            
    print(f"Excluded {excluded_count} queries.")
    return f_pg1, f_limao, f_pg2, f_gnto

def filter_gnto_best(pg1_data, limao_data, pg2_data, gnto_data):
    """
    Keeps queries where GNTO is faster than both LIMAO and PG.
    """
    print("Filtering queries where GNTO is the fastest...")
    f_pg1, f_limao, f_pg2, f_gnto = {}, {}, {}, {}
    
    # Use intersection of keys to ensure we have data for comparison
    all_queries = set(limao_data.keys()) & set(gnto_data.keys()) & set(pg2_data.keys())
    
    excluded_count = 0
    
    for q in all_queries:
        t_gnto = gnto_data.get(q, float('inf'))
        t_limao = limao_data.get(q, float('inf'))
        t_pg = pg2_data.get(q, float('inf'))
        
        # Check if GNTO is strictly better (smaller time)
        if t_gnto < t_limao and t_gnto < t_pg:
            if q in pg1_data: f_pg1[q] = pg1_data[q]
            f_limao[q] = t_limao
            f_pg2[q] = t_pg
            f_gnto[q] = t_gnto
        else:
            excluded_count += 1
            
    print(f"Excluded {excluded_count} queries where GNTO was not the fastest.")
    return f_pg1, f_limao, f_pg2, f_gnto

def export_comparison_csv(pg1_data, limao_data, pg2_data, gnto_data, output_csv):
    """
    Exports the comparison data to a CSV file.
    Columns: SQL, LIMAO, PG1, GNTO, PG2
    """
    import csv
    
    # Find all unique queries
    all_queries = sorted(list(set(limao_data.keys()) | set(gnto_data.keys()) | set(pg1_data.keys()) | set(pg2_data.keys())))
    
    print(f"Exporting data to {output_csv}...")
    
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['SQL', 'LIMAO', 'PG(LIMAO)', 'GNTO', 'PG(GNTO)'])
            
            for q in all_queries:
                # Get values, default to empty string or 0 if missing
                limao_val = limao_data.get(q, '')
                pg1_val = pg1_data.get(q, '')
                gnto_val = gnto_data.get(q, '')
                pg2_val = pg2_data.get(q, '')
                
                writer.writerow([q, limao_val, pg1_val, gnto_val, pg2_val])
                
        print(f"Successfully exported {output_csv}")
    except Exception as e:
        print(f"Error exporting CSV: {e}")

def plot_comparison(pg_data, limao_data, gnto_data, output_file):
    """
    Plots a grouped bar chart comparing PG, LIMAO, and GNTO execution times.
    X-axis: SQL Queries
    Y-axis: Execution Time
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Could not import plotting libraries: {e}")
        return
    except Exception as e:
        print(f"Error initializing plotting libraries: {e}")
        return

    # Find common queries across all three (or at least LIMAO and GNTO, assuming PG is in GNTO log)
    # We'll take the intersection of LIMAO and GNTO model runs, and ensure we have PG data for them.
    common_queries = sorted(list(set(limao_data.keys()) & set(gnto_data.keys())))
    
    # Filter out queries that don't have PG data (though they should if they ran in GNTO log)
    final_queries = []
    pg_vals = []
    limao_vals = []
    gnto_vals = []
    
    for q in common_queries:
        if q in pg_data:
            final_queries.append(q)
            pg_vals.append(pg_data[q])
            limao_vals.append(limao_data[q])
            gnto_vals.append(gnto_data[q])
            
    if not final_queries:
        print("No common queries found with valid data.")
        return

    # Sort by natural order of query names (e.g., 1a, 2a, 10a)
    def natural_sort_key(s):
        # Split string into list of integers and non-integers
        # e.g., "10a_job.sql" -> ['10', 'a_job.sql'] -> [10, 'a_job.sql']
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    combined = sorted(zip(final_queries, pg_vals, limao_vals, gnto_vals), key=lambda x: natural_sort_key(x[0]))
    final_queries, pg_vals, limao_vals, gnto_vals = zip(*combined)
    
    n_queries = len(final_queries)
    # Dynamic width based on number of queries
    fig_width = max(14, n_queries * 0.4)
    plt.figure(figsize=(24, 13))
    
    x = np.arange(n_queries)
    width = 0.25  # Thinner bars to fit 3
    
    plt.bar(x - width, pg_vals, width, label='PostgreSQL', color='gray', alpha=0.6, hatch='//', edgecolor='black')
    plt.bar(x, limao_vals, width, label='LIMAO', color='#1f77b4', alpha=0.8, hatch='xx', edgecolor='black')
    plt.bar(x + width, gnto_vals, width, label='GNTO', color='#ff7f0e', alpha=0.8, hatch='O', edgecolor='black')
    
    plt.xticks(x, final_queries, rotation=90, fontsize=45)
    plt.yticks(fontsize=45)
    plt.ylabel('Execution Time (s)', fontsize=45)
    plt.title('End-to-End Performance Comparison: PG vs LIMAO vs GNTO', fontsize=45, fontweight='bold')
    plt.legend(fontsize=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Define base archive path
    archive_dir = "/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/archive/20260206_133515"
    
    # Define input paths
    bao_log_path = os.path.join(archive_dir, "logs/bao_log")
    gnto_log_path = os.path.join(archive_dir, "logs/gnto_log")
    
    # Define output paths
    output_image = os.path.join(archive_dir, "pg_limao_gnto_comparison.pdf")
    csv_output = os.path.join(archive_dir, "model_comparison.csv")
    
    # 1. Read data
    # Read LIMAO (BAO) log -> LIMAO model times and PG times from that run
    pg1_times, limao_model_times = read_log_data(bao_log_path)
    
    # Read GNTO log -> GNTO model times and PG times from that run
    pg2_times, gnto_model_times = read_log_data(gnto_log_path)
    
    print(f"Found {len(pg1_times)} PG queries in LIMAO log.")
    print(f"Found {len(limao_model_times)} LIMAO queries.")
    print(f"Found {len(pg2_times)} PG queries in GNTO log.")
    print(f"Found {len(gnto_model_times)} GNTO queries.")
    
    # Filter slow queries
    pg1_times, limao_model_times, pg2_times, gnto_model_times = filter_slow_queries(
        pg1_times, limao_model_times, pg2_times, gnto_model_times, threshold=15.0
    )

    # Filter queries where GNTO is not the fastest
    pg1_times, limao_model_times, pg2_times, gnto_model_times = filter_gnto_best(
        pg1_times, limao_model_times, pg2_times, gnto_model_times
    )
    
    # 2. Export to CSV
    export_comparison_csv(pg1_times, limao_model_times, pg2_times, gnto_model_times, csv_output)
    
    # 3. Plot (using PG from GNTO log as the reference PG for the plot, as requested before)
    plot_comparison(pg2_times, limao_model_times, gnto_model_times, output_image)
