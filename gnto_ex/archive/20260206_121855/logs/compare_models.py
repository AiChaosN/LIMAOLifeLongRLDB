import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_server_log(file_path, target_model=None):
    """
    Parses a server log file to extract 'Logged reward' and 'Predicted reward / PG' values.
    Returns a dictionary with 'rewards' and 'predictions'.
    """
    rewards = []
    predictions = []
    
    # Regex patterns
    reward_pattern = re.compile(r"Logged reward of ([\d\.]+)")
    prediction_pattern = re.compile(r"Selected index \d+ after .* Predicted reward / PG: ([\d\.]+) / ([\d\.]+)")
    
    # State tracking
    current_model = None
    last_pred = None
    
    print(f"Parsing {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Detect model switch
                if "Using BAO Model" in line:
                    current_model = "BAO"
                elif "Using GNTO Model" in line:
                    current_model = "GNTO"
                
                # If target_model is specified, only process lines when that model is active
                if target_model and current_model and current_model != target_model:
                    continue

                # Extract Prediction
                pred_match = prediction_pattern.search(line)
                if pred_match:
                    pred_reward = float(pred_match.group(1))
                    pg_reward = float(pred_match.group(2))
                    last_pred = {
                        'pred': pred_reward,
                        'pg_est': pg_reward
                    }
                
                # Extract Reward (Actual execution time usually)
                reward_match = reward_pattern.search(line)
                if reward_match:
                    actual_reward = float(reward_match.group(1))
                    if last_pred:
                        predictions.append(last_pred)
                        rewards.append(actual_reward)
                        last_pred = None # Reset to ensure 1-to-1 mapping
                    else:
                        # Found a reward without an immediately preceding prediction 
                        # (could be from init or non-selected execution, or logs didn't align perfectly)
                        pass
                        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return [], []

    return rewards, predictions

def parse_run_log(file_path):
    """
    Parses client-side run logs (gnto_log or bao_log) to extract execution times.
    """
    data = []
    
    # Regex for lines like: BAO 22a_job.sql 1.059722900390625 or GNTO ...
    model_pattern = re.compile(r"(?:BAO|GNTO)\s+(\S+)\s+([\d\.]+)")
    # Regex for lines like: x x 22a_job.sql 0.4431135654449463 PG
    pg_pattern = re.compile(r"x\s+x\s+(\S+)\s+([\d\.]+)\s+PG")
    
    print(f"Parsing {file_path}...")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                model_match = model_pattern.search(line)
                if model_match:
                    data.append({
                        'type': 'Model',
                        'query': model_match.group(1),
                        'time': float(model_match.group(2))
                    })
                    continue
                
                pg_match = pg_pattern.search(line)
                if pg_match:
                    data.append({
                        'type': 'PG',
                        'query': pg_match.group(1),
                        'time': float(pg_match.group(2))
                    })

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []

    return data

def parse_log_to_dicts(file_path):
    """
    Parses a log file to extract execution times into two dictionaries:
    one for PG (default) and one for Model (BAO/GNTO).
    
    Returns:
        pg_dict: {query: [time1, time2, ...]}
        model_dict: {query: [time1, time2, ...]}
    """
    pg_dict = {}
    model_dict = {}
    
    # Regex patterns
    # Matches: BAO 22a_job.sql 1.05 or GNTO ...
    model_pattern = re.compile(r"(?:BAO|GNTO)\s+(\S+)\s+([\d\.]+)")
    # Matches: x x 22a_job.sql 0.44 PG
    pg_pattern = re.compile(r"x\s+x\s+(\S+)\s+([\d\.]+)\s+PG")
    
    print(f"Parsing {file_path} for dicts...")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check for Model pattern
                model_match = model_pattern.search(line)
                if model_match:
                    query = model_match.group(1)
                    time_val = float(model_match.group(2))
                    if query not in model_dict:
                        model_dict[query] = []
                    model_dict[query].append(time_val)
                    continue
                
                # Check for PG pattern
                pg_match = pg_pattern.search(line)
                if pg_match:
                    query = pg_match.group(1)
                    time_val = float(pg_match.group(2))
                    if query not in pg_dict:
                        pg_dict[query] = []
                    pg_dict[query].append(time_val)
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}, {}
        
    return pg_dict, model_dict

def plot_per_query_comparison(run_data, title, output_filename, output_dir):
    """
    Pairs up PG time vs Model time for each query and plots a side-by-side bar chart.
    """
    # Organize data: query -> {'PG': [], 'Model': []}
    query_data = {}
    for entry in run_data:
        q = entry['query']
        t = entry['time']
        typ = entry['type']
        if q not in query_data:
            query_data[q] = {'PG': [], 'Model': []}
        query_data[q][typ].append(t)
    
    # Extract averages
    queries = []
    pg_times = []
    model_times = []
    
    for q, values in query_data.items():
        if values['PG'] and values['Model']:
            # If there are multiple runs, take the average
            pg_avg = sum(values['PG']) / len(values['PG'])
            model_avg = sum(values['Model']) / len(values['Model'])
            
            queries.append(q)
            pg_times.append(pg_avg)
            model_times.append(model_avg)
            
    if not queries:
        print(f"No paired data found for {title}. Skipping plot.")
        return

    # Sort by PG time descending to see slowest queries first
    combined = sorted(zip(queries, pg_times, model_times), key=lambda x: x[1], reverse=True)
    queries, pg_times, model_times = zip(*combined)
    
    n_queries = len(queries)
    
    # Use horizontal bar chart for better readability of query names
    # Increase figure height dynamically based on number of queries
    fig_height = max(6, n_queries * 0.4) 
    plt.figure(figsize=(12, fig_height))
    
    y = np.arange(n_queries)
    height = 0.35
    
    plt.barh(y + height/2, pg_times, height, label='PG (Default)', color='gray', alpha=0.7)
    plt.barh(y - height/2, model_times, height, label='Model Optimized', color='green', alpha=0.8)
    
    plt.yticks(y, queries)
    plt.xlabel('Execution Time (s)')
    plt.title(f'{title}: Per-Query Execution Time Comparison')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Invert y axis so top queries are at top
    plt.gca().invert_yaxis()
    
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {output_filename}")
    plt.close()

def plot_model_comparison(bao_run_data, gnto_run_data, output_filename, output_dir):
    """
    Plots a side-by-side bar chart comparing BAO Model time vs GNTO Model time for each query.
    """
    # Extract Model times for BAO
    bao_times = {}
    for entry in bao_run_data:
        if entry['type'] == 'Model':
            q = entry['query']
            if q not in bao_times: bao_times[q] = []
            bao_times[q].append(entry['time'])
            
    # Extract Model times for GNTO
    gnto_times = {}
    for entry in gnto_run_data:
        if entry['type'] == 'Model':
            q = entry['query']
            if q not in gnto_times: gnto_times[q] = []
            gnto_times[q].append(entry['time'])

    # Find common queries
    common_queries = sorted(list(set(bao_times.keys()) & set(gnto_times.keys())))
    
    if not common_queries:
        print("No common queries found between BAO and GNTO for comparison.")
        return

    queries = []
    bao_avgs = []
    gnto_avgs = []
    
    for q in common_queries:
        queries.append(q)
        # Average if multiple runs
        bao_avgs.append(sum(bao_times[q])/len(bao_times[q]))
        gnto_avgs.append(sum(gnto_times[q])/len(gnto_times[q]))
        
    # Sort by BAO time descending
    combined = sorted(zip(queries, bao_avgs, gnto_avgs), key=lambda x: x[1], reverse=True)
    queries, bao_avgs, gnto_avgs = zip(*combined)
    
    n_queries = len(queries)
    fig_height = max(6, n_queries * 0.4)
    plt.figure(figsize=(12, fig_height))
    
    y = np.arange(n_queries)
    height = 0.35
    
    plt.barh(y + height/2, bao_avgs, height, label='BAO Model', color='blue', alpha=0.7)
    plt.barh(y - height/2, gnto_avgs, height, label='GNTO Model', color='orange', alpha=0.7)
    
    plt.yticks(y, queries)
    plt.xlabel('Execution Time (s)')
    plt.title('BAO vs GNTO: Per-Query Execution Time Comparison')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis()
    
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {output_filename}")
    plt.close()

def plot_multi_dict_comparison(dict_list, label_list, title, output_filename, output_dir):
    """
    Plots a grouped bar chart comparing the last execution time for each query 
    across multiple dictionaries.
    
    Args:
        dict_list: List of dictionaries {query: [time1, time2, ...]}
        label_list: List of labels corresponding to each dictionary (e.g., ['PG', 'BAO', 'GNTO'])
        title: Plot title
        output_filename: Filename to save
        output_dir: Directory to save
    """
    if not dict_list or not label_list or len(dict_list) != len(label_list):
        print("Error: Invalid input lists for comparison.")
        return

    # 1. Collect all unique queries (keys)
    all_queries = set()
    for d in dict_list:
        all_queries.update(d.keys())
    
    # Sort queries naturally or alphabetically
    queries = sorted(list(all_queries))
    
    if not queries:
        print(f"No queries found to plot for {title}.")
        return

    # 2. Extract data for plotting (last value of each list)
    plot_data = [] # List of lists, one per category
    for d in dict_list:
        category_values = []
        for q in queries:
            if q in d and d[q]:
                # Take the last value as requested
                category_values.append(d[q][-1])
            else:
                # Missing data for this query in this category
                category_values.append(0) 
        plot_data.append(category_values)

    # 3. Plotting
    n_queries = len(queries)
    n_categories = len(dict_list)
    
    # Dynamic height
    fig_height = max(6, n_queries * 0.4)
    plt.figure(figsize=(12, fig_height))
    
    y = np.arange(n_queries)
    total_height = 0.8 # Total height available for a group of bars
    bar_height = total_height / n_categories
    
    # Simple color cycle
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i in range(n_categories):
        # Calculate y-offset to group bars around the tick
        # i=0 starts at top (since y axis inverted later), but mathematically:
        # We want to center the group at y.
        # Start offset: - (total_height / 2) + (bar_height / 2)
        # Shift per item: + i * bar_height
        
        # Center of the group is 'y'.
        # Top-most bar (visually) corresponds to lowest y-value modification if not inverted yet...
        # Let's keep it simple: 
        # offset = (i - n_categories/2 + 0.5) * bar_height
        offset = (i - (n_categories - 1) / 2) * bar_height
        
        c = colors[i % len(colors)]
        plt.barh(y + offset, plot_data[i], height=bar_height, label=label_list[i], color=c, alpha=0.8)

    plt.yticks(y, queries)
    plt.xlabel('Execution Time (s)')
    plt.title(title)
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Invert y axis so top queries are at top
    plt.gca().invert_yaxis()
    
    save_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {output_filename}")
    plt.close()

def analyze_and_plot(bao_server_log, gnto_server_log, bao_run_log, gnto_run_log, output_dir="."):
    
    # 1. Parse Server Logs for Accuracy (Predicted vs Actual)
    # (Keeping this as it is useful validation)
    bao_rewards, bao_preds_data = parse_server_log(bao_server_log, target_model="BAO")
    gnto_rewards, gnto_preds_data = parse_server_log(gnto_server_log, target_model="GNTO")
    
    bao_predicted_vals = [p['pred'] for p in bao_preds_data]
    gnto_predicted_vals = [p['pred'] for p in gnto_preds_data]

    print(f"BAO Server Log: Found {len(bao_rewards)} paired samples.")
    print(f"GNTO Server Log: Found {len(gnto_rewards)} paired samples.")

    # 2. Parse Client Logs for Performance (Time vs PG)
    bao_run_data = parse_run_log(bao_run_log)
    gnto_run_data = parse_run_log(gnto_run_log)
    
    # --- New Plot: Per-Query Bar Chart ---
    print("Generating Per-Query Comparison Bar Charts...")
    plot_per_query_comparison(bao_run_data, "BAO Experiment", "bao_per_query.png", output_dir)
    plot_per_query_comparison(gnto_run_data, "GNTO Experiment", "gnto_per_query.png", output_dir)
    
    # --- Plot BAO vs GNTO Comparison ---
    print("Generating BAO vs GNTO Comparison Bar Chart...")
    plot_model_comparison(bao_run_data, gnto_run_data, "bao_vs_gnto_comparison.png", output_dir)
    
    # --- Plot 1: Prediction Accuracy (Scatter) ---
    plt.figure(figsize=(10, 6))
    
    if bao_rewards and bao_predicted_vals:
        plt.scatter(bao_rewards, bao_predicted_vals, alpha=0.6, label='BAO Model', color='blue', s=20)
    if gnto_rewards and gnto_predicted_vals:
        plt.scatter(gnto_rewards, gnto_predicted_vals, alpha=0.6, label='GNTO Model', color='orange', s=20)

    # Ideal line
    all_actuals = bao_rewards + gnto_rewards
    all_preds = bao_predicted_vals + gnto_predicted_vals
    if all_actuals:
        max_val = max(max(all_actuals), max(all_preds)) if all_preds else max(all_actuals)
        min_val = min(min(all_actuals), min(all_preds)) if all_preds else min(all_actuals)
        min_val = max(min_val, 0.1) 
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

    plt.title('Model Accuracy: Predicted vs Actual Execution Time')
    plt.xlabel('Actual Time (Logged Reward)')
    plt.ylabel('Predicted Time')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    print("Saved accuracy_comparison.png")

    # --- Statistics ---
    print("\n--- Summary Statistics ---")
    
    def stats(name, data_list, type_filter):
        times = [d['time'] for d in data_list if d['type'] == type_filter]
        if not times: return
        print(f"{name}: Count={len(times)}, Total={sum(times):.2f}s, Mean={np.mean(times):.2f}s, Median={np.median(times):.2f}s")
        
    stats("BAO Model", bao_run_data, 'Model')
    stats("BAO PG", bao_run_data, 'PG')
    stats("GNTO Model", gnto_run_data, 'Model')
    stats("GNTO PG", gnto_run_data, 'PG')


if __name__ == "__main__":
    # Define paths
    BASE_DIR = "/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/tmp"
    
    BAO_SERVER_LOG = os.path.join(BASE_DIR, "bao_server_log")
    GNTO_SERVER_LOG = os.path.join(BASE_DIR, "gnto_server_log")
    BAO_RUN_LOG = os.path.join(BASE_DIR, "bao_log")
    GNTO_RUN_LOG = os.path.join(BASE_DIR, "gnto_log")
    
    # Run analysis
    analyze_and_plot(
        bao_server_log=BAO_SERVER_LOG,
        gnto_server_log=GNTO_SERVER_LOG,
        bao_run_log=BAO_RUN_LOG,
        gnto_run_log=GNTO_RUN_LOG,
        output_dir=BASE_DIR
    )
