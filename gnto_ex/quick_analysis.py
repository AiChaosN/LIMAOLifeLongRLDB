import re
import numpy as np
import os

def parse_run_log(file_path, model_name):
    data = []
    # Regex for lines like: BAO 22a_job.sql 1.059722900390625
    # In gnto_msg, it also uses "BAO" prefix but we know it's GNTO model
    model_pattern = re.compile(r"BAO\s+(\S+)\s+([\d\.]+)")
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                match = model_pattern.search(line)
                if match:
                    data.append({
                        'model': model_name,
                        'query': match.group(1),
                        'time': float(match.group(2))
                    })
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    return data

def analyze():
    # Paths based on previous exploration
    limao_log = "/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/archive/20260206_121855/logs/bao_log"
    gnto_log = "/home/AiChaosN/Project/Phd/project/LIMAOLifeLongRLDB/gnto_ex/archive/20260206_121855/logs/gnto_log"
    
    limao_data = parse_run_log(limao_log, "LIMAO")
    gnto_data = parse_run_log(gnto_log, "GNTO")
    
    # Filter out timeouts (32.0 seems to be timeout or max value in logs)
    # limao_valid = [d['time'] for d in limao_data if d['time'] < 32.0]
    # gnto_valid = [d['time'] for d in gnto_data if d['time'] < 32.0]
    
    # Use last value for each query to simulate "final" performance
    limao_last = {}
    for d in limao_data:
        limao_last[d['query']] = d['time']
        
    gnto_last = {}
    for d in gnto_data:
        gnto_last[d['query']] = d['time']
        
    limao_times = list(limao_last.values())
    gnto_times = list(gnto_last.values())
    
    print(f"--- Experiment 4: End-to-End Comparison (Last Iteration) ---")
    print(f"LIMAO (Bao): {len(limao_times)} unique queries")
    print(f"GNTO: {len(gnto_times)} unique queries")
    
    if limao_times:
        print(f"LIMAO Median Time: {np.median(limao_times):.4f} s")
        print(f"LIMAO Mean Time: {np.mean(limao_times):.4f} s")
        
    if gnto_times:
        print(f"GNTO Median Time: {np.median(gnto_times):.4f} s")
        print(f"GNTO Mean Time: {np.mean(gnto_times):.4f} s")

    # Find common queries to compare head-to-head
    common_queries = set(limao_last.keys()) & set(gnto_last.keys())
    print(f"\nCommon Queries: {len(common_queries)}")
    
    better_gnto = 0
    for q in common_queries:
        if gnto_last[q] < limao_last[q]:
            better_gnto += 1
            
    if common_queries:
        print(f"GNTO better than LIMAO in {better_gnto}/{len(common_queries)} ({better_gnto/len(common_queries)*100:.1f}%) queries")

if __name__ == "__main__":
    analyze()
