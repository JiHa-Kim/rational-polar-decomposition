import json
import statistics
import argparse
from collections import defaultdict


def load_results(path):
    results = defaultdict(list)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    # Use (shape, dtype) as grouping key, (case) as sub-key
                    k = (r["shape"], r["dtype"])
                    results[k].append(r)
    except FileNotFoundError:
        print(f"File not found: {path}")
    return results


def format_markdown_table(dwh2_path, gns_path):
    dwh2_raw = load_results(dwh2_path)
    gns_raw = load_results(gns_path)

    all_groups = sorted(set(dwh2_raw.keys()) | set(gns_raw.keys()))

    metrics = [
        ("Med ms", "median_ms", ".2f", False),
        ("Ortho", "ortho_proj", ".2e", True),
        ("Supp", "ortho_supp", ".2e", True),
        ("Skew", "p_skew_rel_fro", ".2e", True),
        ("P2-Err", "p2_gram_rel_fro", ".2e", True),
        ("Rec", "rec_resid", ".2e", True),
    ]

    output = ["# DWH2 vs GNS Benchmark Comparison\n"]

    for shape, dtype in all_groups:
        output.append(f"## Shape: {shape}, Dtype: {dtype}\n")
        
        header = ["Case", "Method"] + [m[0] for m in metrics]
        table = "| " + " | ".join(header) + " |\n"
        table += "| " + " | ".join(["---"] * len(header)) + " |\n"

        # Organize by case
        d_by_case = {r["case"]: r for r in dwh2_raw.get((shape, dtype), [])}
        g_by_case = {r["case"]: r for r in gns_raw.get((shape, dtype), [])}
        
        all_cases = sorted(set(d_by_case.keys()) | set(g_by_case.keys()))

        for case in all_cases:
            d_rec = d_by_case.get(case)
            g_rec = g_by_case.get(case)

            if not d_rec or not g_rec:
                continue

            d_row = [case, "DWH2"]
            g_row = ["", "GNS"]

            for _, key, prec, is_exp in metrics:
                dv = d_rec.get(key, 0.0)
                gv = g_rec.get(key, 0.0)

                fmt = f"{{:{prec}}}"
                ds, gs = fmt.format(dv), fmt.format(gv)

                # Skip comparison for metrics that are NaN or Inf
                import math
                if not math.isnan(dv) and not math.isnan(gv):
                    if dv < gv:
                        ds = f"**{ds}**"
                    elif gv < dv:
                        gs = f"**{gs}**"
                
                d_row.append(ds)
                g_row.append(gs)

            table += "| " + " | ".join(d_row) + " |\n"
            table += "| " + " | ".join(g_row) + " |\n"

        output.append(table)
        output.append("\n")

    return "\n".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format DWH2 vs GNS comparison in grouped markdown tables."
    )
    parser.add_argument(
        "--dwh2", default="results/dwh2_baseline.jsonl", help="Path to DWH2 results"
    )
    parser.add_argument(
        "--gns", default="results/gns_baseline.jsonl", help="Path to GNS results"
    )
    args = parser.parse_args()

    table_md = format_markdown_table(args.dwh2, args.gns)
    print(table_md)
