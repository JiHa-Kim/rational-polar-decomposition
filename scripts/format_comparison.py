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
                    # Use case, shape, dtype as key
                    k = (r["case"], r["shape"], r["dtype"])
                    results[k].append(r)
    except FileNotFoundError:
        print(f"File not found: {path}")
    return results


def format_markdown_table(dwh2_path, gns_path):
    dwh2_results = load_results(dwh2_path)
    gns_results = load_results(gns_path)

    def sort_key(k):
        # Sort by case (alphabetical), then shape (numerical descending), then dtype
        case, shape, dtype = k
        try:
            m, n = map(int, shape.lower().split("x"))
            shape_val = m * 1000000 + n  # Large multiplier for m
        except Exception:
            shape_val = 0
        return (case, -shape_val, dtype)

    all_keys = sorted(set(dwh2_results.keys()) | set(gns_results.keys()), key=sort_key)

    header = [
        "Case",
        "Shape",
        "Dtype",
        "DWH2 Med (ms)",
        "GNS Med (ms)",
        "Speedup",
        "DWH2 Ortho (F)",
        "GNS Ortho (F)",
        "DWH2 P-Err (F)",
        "GNS P-Err (F)",
    ]

    rows = []
    for k in all_keys:
        case, shape, dtype = k

        d_recs = dwh2_results.get(k, [])
        g_recs = gns_results.get(k, [])

        if not d_recs or not g_recs:
            continue

        d_med = statistics.median([r["median_ms"] for r in d_recs])
        g_med = statistics.median([r["median_ms"] for r in g_recs])
        speedup = g_med / d_med if d_med > 0 else 0

        d_ortho = statistics.mean([r.get("ortho_proj", 0.0) for r in d_recs])
        g_ortho = statistics.mean([r.get("ortho_proj", 0.0) for r in g_recs])

        d_perr = statistics.mean([r.get("p2_gram_rel_fro", 0.0) for r in d_recs])
        g_perr = statistics.mean([r.get("p2_gram_rel_fro", 0.0) for r in g_recs])

        def highlight_better(v1, v2, precision=".2f", is_exp=False):
            fmt = f"{{:{precision}}}" if not is_exp else f"{{:{precision}}}"
            s1, s2 = fmt.format(v1), fmt.format(v2)
            if v1 < v2:
                return f"**{s1}**", s2
            elif v2 < v1:
                return s1, f"**{s2}**"
            return s1, s2

        d_ms_str, g_ms_str = highlight_better(d_med, g_med)
        d_ortho_str, g_ortho_str = highlight_better(d_ortho, g_ortho, ".2e", True)
        d_perr_str, g_perr_str = highlight_better(d_perr, g_perr, ".2e", True)

        rows.append(
            [
                case,
                shape,
                dtype,
                d_ms_str,
                g_ms_str,
                f"{speedup:.2f}x",
                d_ortho_str,
                g_ortho_str,
                d_perr_str,
                g_perr_str,
            ]
        )

    # Generate markdown string
    table = "| " + " | ".join(header) + " |\n"
    table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"

    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format DWH2 vs GNS comparison in a markdown table."
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
