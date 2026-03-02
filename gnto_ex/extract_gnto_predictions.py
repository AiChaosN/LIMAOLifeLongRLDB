import re
import pandas as pd
import os

# Paths
base_dir = '/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/archive/20260206_133515/logs'
gnto_log_path = os.path.join(base_dir, 'gnto_log')
server_log_path = os.path.join(base_dir, 'gnto_server_log')
output_csv = 'gnto_predictions.csv'

def parse_gnto_log(path):
    queries = []
    current_iteration = 0
    with open(path, 'r') as f:
        for line in f:
            # Check for iteration header
            # === Executing queries using GNTO optimizer, global iteration 1/200, ...
            iter_match = re.search(r'global iteration (\d+)/', line)
            if iter_match:
                current_iteration = int(iter_match.group(1))
            
            # Pattern: GNTO <query_name> <execution_time>
            match = re.search(r'GNTO\s+(\S+\.sql)\s+([0-9.]+)', line)
            if match:
                queries.append({
                    'query': match.group(1),
                    'actual_latency': float(match.group(2)),
                    'iteration': current_iteration
                })
    return queries

def parse_server_log(path):
    predictions = []
    # We need to capture the prediction that leads to a logged reward
    # Pattern for prediction: Selected index \d+ .* Predicted reward / PG: ([0-9.]+) / ([0-9.]+)
    # Pattern for reward: Logged reward of ([0-9.]+)
    
    current_prediction = None
    
    with open(path, 'r') as f:
        for line in f:
            pred_match = re.search(r'Predicted reward / PG: ([0-9.]+) / ([0-9.]+)', line)
            if pred_match:
                # We keep the latest prediction seen before a reward
                current_prediction = float(pred_match.group(1))
            
            reward_match = re.search(r'Logged reward of ([0-9.]+)', line)
            if reward_match:
                if current_prediction is not None:
                    predictions.append({
                        'predicted_cost': current_prediction,
                        'logged_reward': float(reward_match.group(1))
                    })
                    current_prediction = None # Reset after matching
                else:
                    # Found reward but no preceding prediction?
                    print(f"Warning: Logged reward without preceding prediction at line: {line.strip()}")
                    predictions.append({
                        'predicted_cost': None,
                        'logged_reward': float(reward_match.group(1))
                    })
    return predictions

def main():
    queries = parse_gnto_log(gnto_log_path)
    predictions = parse_server_log(server_log_path)
    
    print(f"Found {len(queries)} queries in gnto_log")
    print(f"Found {len(predictions)} predictions in gnto_server_log")
    
    # Align
    min_len = min(len(queries), len(predictions))
    
    aligned_data = []
    for i in range(min_len):
        entry = queries[i]
        pred = predictions[i]
        entry['predicted_cost'] = pred['predicted_cost']
        entry['logged_reward'] = pred['logged_reward']
        aligned_data.append(entry)
        
    df = pd.DataFrame(aligned_data)
    
    # Filter to keep only the latest iteration for each query
    # Sort by iteration descending, then drop duplicates on 'query'
    df_latest = df.sort_values('iteration', ascending=False).drop_duplicates('query', keep='first')
    
    output_csv_latest = 'gnto_predictions_latest.csv'
    df_latest.to_csv(output_csv_latest, index=False)
    print(f"Saved {len(df_latest)} latest records to {output_csv_latest}")
    
    # Also save full history
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} aligned records to {output_csv}")
    
    # Check for the 112 queries
    unique_queries = df['query'].unique()
    print(f"Unique queries found: {len(unique_queries)}")

if __name__ == "__main__":
    main()
