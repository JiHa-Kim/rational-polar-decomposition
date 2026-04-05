import subprocess
import json
import os
import sys

cases = [
    "rank_1_heavy",
    "ill_conditioned",
    "lowrank_noise",
    "heavy_tail_t",
    "adversarial_condition",
]
dtypes = ["fp16", "bf16"]

results = []

base_dir = os.path.dirname(__file__)
profile_script = os.path.join(base_dir, "profile_instability.py")

for case in cases:
    for dtype in dtypes:
        print(f"Running profile for case={case}, dtype={dtype}...")
        json_file = f"temp_profile_{case}_{dtype}.json"
        cmd = [
            sys.executable,
            profile_script,
            "--case",
            case,
            "--dtype",
            dtype,
            "--apply-compare",
            "--json",
            json_file,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                results.append(data)
                os.remove(json_file)
        except Exception as e:
            print(f"Error running {case} {dtype}: {e}")

output_path = os.path.abspath(
    os.path.join(base_dir, "..", "results", "profile_results_all.json")
)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. Results saved to {output_path}")
