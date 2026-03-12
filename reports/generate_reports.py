import matplotlib.pyplot as plt
import json
from pathlib import Path

report_dir = Path(__file__).resolve().parent

with open(report_dir / "results_fp.json") as f:
    fp = json.load(f)
with open(report_dir / "results_oop.json") as f:
    oop = json.load(f)

# ==============================
# OneMax combined
# ==============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Genetic Algorithm - OneMax", fontsize=14)

ax1.plot(oop["OneMax"]["history"])
ax1.set_title("OOP")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Best Fitness")

ax2.plot(fp["OneMax"]["history"])
ax2.set_title("FP")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best Fitness")

plt.tight_layout()
plt.savefig(report_dir / "onemax_curve.png")
plt.close()

# ==============================
# Knapsack combined
# ==============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Genetic Algorithm - Knapsack", fontsize=14)

ax1.plot(oop["Knapsack"]["history"])
ax1.set_title("OOP")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Best Fitness")

ax2.plot(fp["Knapsack"]["history"])
ax2.set_title("FP")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best Fitness")

plt.tight_layout()
plt.savefig(report_dir / "knapsack_curve.png")
plt.close()

print("Reports generated successfully!")
print(f"  -> {report_dir / 'onemax_curve.png'}")
print(f"  -> {report_dir / 'knapsack_curve.png'}")