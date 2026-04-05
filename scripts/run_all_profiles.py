import subprocess
import json
import os

cases = [
    "rank_1_heavy",
    "ill_conditioned",
    "lowrank_noise",
    "heavy_tail_t",
    "adversarial_condition",
]
dtypes = ["fp16", "bf16"]

results = []

for case in cases:
    for dtype in dtypes:
        print(f"Running profile for case={case}, dtype={dtype}...")
        json_file = f"temp_profile_{case}_{dtype}.json"
        cmd = [
            "uv",
            "run",
            "python",
            "profile_instability.py",
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
            with open(json_file, "r") as f:
                data = json.load(f)
            results.append(data)
            os.remove(json_file)
        except Exception as e:
            print(f"Error running {case} {dtype}: {e}")

with open("profile_results_all.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done. Results saved to profile_results_all.json")
