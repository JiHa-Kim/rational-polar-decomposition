import json
import statistics
import argparse
import math
from collections import defaultdict


def load_results(path):
    results = defaultdict(list)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    k = (r["shape"], r["dtype"])
                    results[k].append(r)
    except FileNotFoundError:
        print(f"File not found: {path}")
    return results


def _generate_table(metrics, d_by_case, g_by_case, all_cases):
    header = ["Case", "Method"] + [m[0] for m in metrics]
    table_lines = []
    table_lines.append("| " + " | ".join(header) + " |")
    table_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for case in all_cases:
        d_rec = d_by_case.get(case)
        g_rec = g_by_case.get(case)

        if not d_rec or not g_rec:
            continue

        d_row = [case, "DWH2"]
        g_row = ["", "GNS"]

        for _, key, prec in metrics:
            dv = d_rec.get(key, 0.0)
            gv = g_rec.get(key, 0.0)

            fmt = f"{{:{prec}}}"
            ds, gs = fmt.format(dv), fmt.format(gv)

            if not math.isnan(dv) and not math.isnan(gv):
                if dv < gv:
                    ds = f"**{ds}**"
                elif gv < dv:
                    gs = f"**{gs}**"
            
            d_row.append(ds)
            g_row.append(gs)

        table_lines.append("| " + " | ".join(d_row) + " |")
        table_lines.append("| " + " | ".join(g_row) + " |")

    return "\n".join(table_lines)


def format_markdown_table(dwh2_path, gns_path):
    dwh2_raw = load_results(dwh2_path)
    gns_raw = load_results(gns_path)

    all_groups = sorted(set(dwh2_raw.keys()) | set(gns_raw.keys()))

    primary_metrics = [
        ("Med ms", "median_ms", ".2f"),
        ("Ortho", "ortho_proj", ".2e"),
        ("Supp", "ortho_supp", ".2e"),
        ("P2-Err", "p2_gram_rel_fro", ".2e"),
        ("Rec", "rec_resid", ".2e"),
    ]
    
    skew_metrics = [
        ("Skew", "p_skew_rel_fro", ".2e"),
    ]

    output = [
        "# DWH2 vs GNS Benchmark Comparison",
        "",
        "## Metric Definitions",
        "- **Med ms**: Median execution time in milliseconds.",
        r"- **Ortho (Projection Defect)**: Relative Frobenius norm $\|S^2 - S\|_F / \|S\|_F$ where $S = Q^T Q$. Measures how close $Q^T Q$ is to being an idempotent projector.",
        r"- **Supp (Support Residual)**: Relative Frobenius norm $\|(I - S)G\|_F / \|G\|_F$ where $G = A^T A$. Measures how much of the input signal is lost by the approximate projection.",
        r"- **Skew (Symmetry Error)**: Relative Frobenius norm of the skew-symmetric part of $Q^T A$. Measures how far the recovered factor $P = \text{sym}(Q^T A)$ is from being the exact symmetric part.",
        r"- **P2-Err (Gram Reconstruction)**: Relative Frobenius norm $\|P^2 - G\|_F / \|G\|_F$. Measures how well the squared symmetric factor reconstructs the input Gram matrix.",
        r"- **Rec (Reconstruction Residual)**: Relative Frobenius norm $\|A - QP\|_F / \|A\|_F$. The total error in the polar decomposition $A \approx QP$.",
        "",
    ]

    for shape, dtype in all_groups:
        output.append(f"## Shape: {shape}, Dtype: {dtype}")
        output.append("")

        d_by_case = {r["case"]: r for r in dwh2_raw.get((shape, dtype), [])}
        g_by_case = {r["case"]: r for r in gns_raw.get((shape, dtype), [])}
        all_cases = sorted(set(d_by_case.keys()) | set(g_by_case.keys()))

        output.append("### Primary Quality & Performance")
        output.append(_generate_table(primary_metrics, d_by_case, g_by_case, all_cases))
        output.append("")

        output.append("### Symmetry Analysis (Secondary)")
        output.append(_generate_table(skew_metrics, d_by_case, g_by_case, all_cases))
        output.append("")

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
