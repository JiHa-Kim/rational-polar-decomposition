import json
import statistics
import argparse
from collections import defaultdict

def load_results(path):
    results = defaultdict(list)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    # Use case, shape, dtype as key
                    k = (r['case'], r['shape'], r['dtype'])
                    results[k].append(r)
    except FileNotFoundError:
        print(f"File not found: {path}")
    return results

def format_markdown_table(dwh2_path, gns_path):
    dwh2_results = load_results(dwh2_path)
    gns_results = load_results(gns_path)
    
    all_keys = sorted(set(dwh2_results.keys()) | set(gns_results.keys()))
    
    header = [
        "Case", "Shape", "Dtype", 
        "DWH2 Med (ms)", "GNS Med (ms)", "Speedup",
        "DWH2 Ortho (F)", "GNS Ortho (F)",
        "DWH2 P-Err (F)", "GNS P-Err (F)"
    ]
    
    rows = []
    for k in all_keys:
        case, shape, dtype = k
        
        d_recs = dwh2_results.get(k, [])
        g_recs = gns_results.get(k, [])
        
        if not d_recs or not g_recs:
            continue
            
        d_med = statistics.median([r['median_ms'] for r in d_recs])
        g_med = statistics.median([r['median_ms'] for r in g_recs])
        speedup = g_med / d_med if d_med > 0 else 0
        
        d_ortho = statistics.mean([r.get('ortho_fro', 0.0) for r in d_recs])
        g_ortho = statistics.mean([r.get('ortho_fro', 0.0) for r in g_recs])
        
        d_perr = statistics.mean([r.get('p2_gram_rel_fro', 0.0) for r in d_recs])
        g_perr = statistics.mean([r.get('p2_gram_rel_fro', 0.0) for r in g_recs])
        
        rows.append([
            case, shape, dtype,
            f"{d_med:.2f}", f"{g_med:.2f}", f"{speedup:.2f}x",
            f"{d_ortho:.2e}", f"{g_ortho:.2e}",
            f"{d_perr:.2e}", f"{g_perr:.2e}"
        ])
    
    # Generate markdown string
    table = "| " + " | ".join(header) + " |\n"
    table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"
    
    return table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format DWH2 vs GNS comparison in a markdown table.")
    parser.add_argument("--dwh2", default="results/dwh2_baseline.jsonl", help="Path to DWH2 results")
    parser.add_argument("--gns", default="results/gns_baseline.jsonl", help="Path to GNS results")
    args = parser.parse_args()
    
    table_md = format_markdown_table(args.dwh2, args.gns)
    print(table_md)
