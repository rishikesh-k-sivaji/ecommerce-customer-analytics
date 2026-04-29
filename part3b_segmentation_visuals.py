# ================================================================
# Multi-Channel E-commerce with Behavioral Metrics
# PART 3b — SEGMENTATION VISUALISATIONS
# ================================================================
# Requires : customer_segments.csv  (produced by Part 3)
#            seg_results.pkl        (produced by Part 3)
# Output   : 6 PNG chart files
# ================================================================

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── STYLE ────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

SEG_COLORS = {
    "Champions":          "#4C72B0",
    "Loyal Customers":    "#55A868",
    "Potential Loyalists":"#8172B2",
    "At Risk":            "#DD8452",
    "Need Attention":     "#C44E52",
}
SAVE_OPTS = dict(dpi=150, bbox_inches="tight")

def save(name):
    plt.savefig(f"{name}.png", **SAVE_OPTS)
    plt.close()
    print(f"  [Saved] {name}.png")

# ── LOAD DATA ────────────────────────────────────────────────
df = pd.read_csv("customer_segments.csv")

with open("seg_results.pkl", "rb") as f:
    seg = pickle.load(f)

pca_coords  = seg["pca_coords"]
label_map   = seg["label_map"]
inertia     = seg["inertia"]
db_scores   = seg["db_scores"]
k_range     = seg["k_range"]
K           = seg["best_k"]
profile     = seg["profile"]
pca_obj     = seg["pca"]

seg_order = ["Champions", "Loyal Customers",
             "Potential Loyalists", "At Risk", "Need Attention"]
colors     = [SEG_COLORS[s] for s in seg_order]

print("=" * 55)
print("PART 3b — SEGMENTATION VISUALISATIONS")
print("=" * 55)
print(f"\nCustomers loaded : {df.shape[0]:,}")
print(f"Segments found   : {df['Segment_Label'].nunique()}\n")

# ================================================================
# CHART 1 — ELBOW + DAVIES-BOULDIN (shows exactly why k=5 was chosen)
# ================================================================
print("── Chart 1 : Elbow + Davies-Bouldin ──")

chosen_idx = k_range.index(K)
db_chosen  = db_scores[chosen_idx]
db_k2      = db_scores[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("How k=5 was selected — Elbow + Davies-Bouldin",
             fontsize=13, fontweight="bold")

# left plot — inertia elbow
ax1.plot(k_range, inertia, "o-", color="#4C72B0", linewidth=2.5, markersize=7)
ax1.axvline(K, color="#C44E52", linestyle="--", linewidth=1.8, label=f"chosen k={K}")
ax1.annotate(f"k={K}  ({inertia[chosen_idx]/1000:.0f}k)",
             xy=(K, inertia[chosen_idx]),
             xytext=(K + 0.8, inertia[chosen_idx] + 15000),
             fontsize=9, color="#C44E52",
             arrowprops=dict(arrowstyle="->", color="#C44E52"))
ax1.set_title("Inertia — elbow flattens after k=5", fontsize=11)
ax1.set_xlabel("k (number of clusters)")
ax1.set_ylabel("Inertia")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax1.legend(fontsize=10)

# right plot — davies-bouldin (lower = better clusters)
ax2.plot(k_range, db_scores, "s-", color="#55A868", linewidth=2.5, markersize=7)
ax2.axvline(K, color="#C44E52", linestyle="--", linewidth=1.8, label=f"chosen k={K}")

# show why k=2 (silhouette's pick) was rejected
ax2.annotate(f"k=2  DB={db_k2:.2f}\nsilhouette pick — too coarse",
             xy=(2, db_k2),
             xytext=(3.5, db_k2 + 0.03),
             fontsize=8, color="#DD8452",
             arrowprops=dict(arrowstyle="->", color="#DD8452"))

# show k=5 as the best
ax2.annotate(f"k={K}  DB={db_chosen:.2f}\nlowest score = best separation",
             xy=(K, db_chosen),
             xytext=(K + 0.8, db_chosen - 0.05),
             fontsize=8, color="#C44E52",
             arrowprops=dict(arrowstyle="->", color="#C44E52"))

ax2.set_title("Davies-Bouldin score — lower is better\nk=5 has the lowest score", fontsize=11)
ax2.set_xlabel("k (number of clusters)")
ax2.set_ylabel("Davies-Bouldin score (lower = better)")
ax2.legend(fontsize=10)

plt.tight_layout()
save("seg_chart01_elbow_and_db")

# ================================================================
# CHART 2 — CLUSTER SIZES
# ================================================================
print("── Chart 2 : Cluster sizes ──")
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Customer Count per Segment", fontsize=13, fontweight="bold")

sizes = df["Segment_Label"].value_counts().reindex(seg_order)
bars  = ax.bar(sizes.index, sizes.values,
               color=colors, edgecolor="white", width=0.6)

for bar, val in zip(bars, sizes.values):
    pct = val / sizes.sum() * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Number of Customers", fontsize=11)
ax.set_ylim(0, sizes.max() * 1.2)
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save("seg_chart02_cluster_sizes")

# ================================================================
# CHART 3 — PCA 2D SCATTER
# ================================================================
print("── Chart 3 : PCA 2D scatter ──")
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Customer Segments — PCA 2D View", fontsize=13, fontweight="bold")

for seg_name in seg_order:
    mask = df["Segment_Label"] == seg_name
    ax.scatter(
        pca_coords[mask, 0],
        pca_coords[mask, 1],
        s=5, alpha=0.4,
        c=SEG_COLORS[seg_name],
        label=f"{seg_name} (n={mask.sum():,})"
    )

ax.set_xlabel(f"PC1 — {pca_obj.explained_variance_ratio_[0]:.1%} variance", fontsize=11)
ax.set_ylabel(f"PC2 — {pca_obj.explained_variance_ratio_[1]:.1%} variance", fontsize=11)
ax.legend(markerscale=4, fontsize=9, loc="upper right")
plt.tight_layout()
save("seg_chart03_pca_scatter")

# ================================================================
# CHART 4 — CLUSTER PROFILE HEATMAP
# ================================================================
print("── Chart 4 : Cluster profile heatmap ──")

plot_cols = [
    "Customer_Lifetime_Value", "Monetary",
    "Loyalty_Score", "Customer_Satisfaction",
    "Churn_Risk_Score", "Recency",
    "Frequency", "Email_Open_Rate",
    "Newsletter_Subscribed", "Is_Weekend"
]
col_labels = [
    "CLV", "Monetary",
    "Loyalty", "Satisfaction",
    "Churn Risk", "Recency",
    "Frequency", "Email Open Rate",
    "Newsletter (0/1)", "Is Weekend (0/1)"
]

heat_data = profile.reindex(seg_order)[plot_cols]
# normalise each column 0–1 so colours are comparable
heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min() + 1e-8)

fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("Normalised Cluster Profile Heatmap", fontsize=13, fontweight="bold")

im = ax.imshow(heat_norm.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(seg_order)))
ax.set_yticklabels(seg_order, fontsize=10)

# annotate cells with raw values
for i, seg_name in enumerate(seg_order):
    for j, col in enumerate(plot_cols):
        raw = heat_data.loc[seg_name, col]
        txt = f"{raw:.0f}" if raw > 1 else f"{raw:.2f}"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=7.5, color="black")

plt.colorbar(im, ax=ax, label="Normalised value (0 = low, 1 = high)",
             fraction=0.02, pad=0.02)
plt.tight_layout()
save("seg_chart04_profile_heatmap")

# ================================================================
# CHART 5 — KEY METRICS BAR CHARTS
# ================================================================
print("── Chart 5 : Key metrics by segment ──")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Key Metrics by Customer Segment", fontsize=13, fontweight="bold")

metrics = [
    ("Customer_Lifetime_Value", "Avg CLV",              False),
    ("Revenue",                 "Avg Revenue",           False),
    ("Churn_Risk_Score",        "Avg Churn Risk Score",  False),
    ("Loyalty_Score",           "Avg Loyalty Score",     False),
    ("Newsletter_Subscribed",   "Newsletter Subscribed %", True),
    ("Is_Weekend",              "Weekend Purchases %",    True),
]

for ax, (col, title, is_pct) in zip(axes.flat, metrics):
    vals = df.groupby("Segment_Label")[col].mean().reindex(seg_order)
    bars = ax.bar(seg_order, vals.values * (100 if is_pct else 1),
                  color=colors, edgecolor="white", width=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    if is_pct:
        ax.set_ylabel("%")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.0f}%"))
    for bar, val in zip(bars, vals.values):
        display = f"{val*100:.0f}%" if is_pct else f"{val:,.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ax.get_ylim()[1] * 0.01,
                display, ha="center", va="bottom", fontsize=8)

plt.tight_layout()
save("seg_chart05_key_metrics")

# ================================================================
# CHART 6 — RFM SCORES BY SEGMENT
# ================================================================
print("── Chart 6 : RFM scores by segment ──")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("RFM Scores by Segment", fontsize=13, fontweight="bold")

rfm_cols   = ["R_score", "F_score", "M_score"]
rfm_titles = ["Recency Score\n(5 = most recent)",
              "Frequency Score\n(5 = most frequent)",
              "Monetary Score\n(5 = highest spend)"]

for ax, col, title in zip(axes, rfm_cols, rfm_titles):
    vals = df.groupby("Segment_Label")[col].mean().reindex(seg_order)
    bars = ax.bar(seg_order, vals.values,
                  color=colors, edgecolor="white", width=0.6)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 6)
    ax.set_ylabel("Avg Score (1–5)")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save("seg_chart06_rfm_scores")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 55)
print("All segmentation charts saved")
print("=" * 55)
print("""
  seg_chart01_elbow_and_db.png   — Elbow + Davies-Bouldin (why k=5 chosen, why k=2 rejected)
  seg_chart02_cluster_sizes.png  — Customer count per segment
  seg_chart03_pca_scatter.png    — PCA 2D cluster scatter plot
  seg_chart04_profile_heatmap.png— Normalised feature heatmap
  seg_chart05_key_metrics.png    — CLV, revenue, churn, loyalty, newsletter, weekend by segment
  seg_chart06_rfm_scores.png     — R, F, M scores per segment
""")
