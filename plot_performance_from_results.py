# plot_performance_from_results.py
# Reads model/training_results.csv and makes a grouped bar + average line chart.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "model" / "training_results.csv"
out_dir  = ROOT / "evaluation_results"
out_dir.mkdir(exist_ok=True)

# 1) Load
df = pd.read_csv(csv_path)

# 2) Clean display names (optional)
name_map = {
    "ckd": "CKD",
    "diabetes": "Diabetes",
    "flu": "Flu",
    "heart": "Heart Disease",
    "parkinsons": "Parkinson's",
    "pneumonia": "Pneumonia"
}
df["disease"] = df["disease"].map(name_map).fillna(df["disease"])

# 3) Order diseases for nicer view (optional)
order = ["CKD", "Diabetes", "Flu", "Heart Disease", "Parkinson's", "Pneumonia"]
df = df.set_index("disease").loc[order].reset_index()

# 4) Compute row-wise average to plot a trend line
df["avg"] = df[["accuracy","precision","recall","f1_score"]].mean(axis=1)

# 5) Plot
plt.figure(figsize=(11, 6))
x = range(len(df))
bar_w = 0.18

plt.bar([p - 1.5*bar_w for p in x], df["accuracy"],  width=bar_w, label="Accuracy")
plt.bar([p - 0.5*bar_w for p in x], df["precision"], width=bar_w, label="Precision")
plt.bar([p + 0.5*bar_w for p in x], df["recall"],    width=bar_w, label="Recall")
plt.bar([p + 1.5*bar_w for p in x], df["f1_score"],  width=bar_w, label="F1-Score")

# Average line
plt.plot(x, df["avg"], marker="o", linewidth=2, label="Average")

plt.xticks(list(x), df["disease"], rotation=12, ha="right")
plt.ylim(0.6, 1.0)
plt.ylabel("Score")
plt.title("Performance Comparison of Disease Models")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

out_file = out_dir / "model_performance_comparison.png"
plt.savefig(out_file, dpi=300)
plt.show()
print("Saved:", out_file.resolve())
