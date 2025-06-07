import csv, numpy as np
from common import RESULTS_DIR, run_full_hybrid, x0, bounds

def test_multiple_initials():
    path = f"{RESULTS_DIR}/hybrid_multiple_runs.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"x0_{i}" for i in range(18)] + [f"xopt_{i}" for i in range(18)]
        writer.writerow(header)
        for run in range(10):
            jitter = 0.1*(np.random.rand(18)-0.5)
            x_start = x0*(1+jitter)
            x_end   = run_full_hybrid(x_start)
            writer.writerow(list(np.round(x_start,4)) + list(np.round(x_end,4)))
    print("Saved", path)

if __name__ == "__main__":
    test_multiple_initials()
    print("test_multiple_initials passed")
