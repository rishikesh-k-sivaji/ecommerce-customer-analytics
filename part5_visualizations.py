import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, confusion_matrix

# style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"
sns.set_palette(PALETTE)

FONT_TITLE = dict(fontsize=13, fontweight="bold")
FONT_AXIS  = dict(fontsize=10)
SAVE_OPTS  = dict(dpi=150, bbox_inches="tight")

def save(name):
    plt.savefig(f"{name}.png", **SAVE_OPTS)
    plt.close()
    print(f"  saved: {name}.png")


# load all data
print("loading data...")
df      = pd.read_csv("customer_clean.csv")
df_seg  = pd.read_csv("customer_segments.csv")
df_pred = pd.read_csv("customer_clean_with_predictions.csv")

with open("churn_results.pkl",  "rb") as f: churn = pickle.load(f)
with open("seg_results.pkl",    "rb") as f: seg   = pickle.load(f)
with open("clv_results.pkl",    "rb") as f: clv   = pickle.load(f)

print(f"customers loaded: {df.shape[0]:,}\n")



# chart 1 — target variable distributions

print("chart 1: target distributions")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Target Variable Distributions", **FONT_TITLE)

churn_counts = df["Churned"].value_counts().sort_index()
axes[0].pie(churn_counts,
            labels=["Not Churned (0)", "Churned (1)"],
            autopct="%1.1f%%", colors=[GREEN, RED],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0].set_title("Churn Distribution", **FONT_TITLE)

axes[1].hist(df["Customer_Lifetime_Value"], bins=60,
             color=BLUE, edgecolor="white", alpha=0.85)
axes[1].set_title("Customer Lifetime Value", **FONT_TITLE)
axes[1].set_xlabel("CLV", **FONT_AXIS)
axes[1].set_ylabel("Count", **FONT_AXIS)
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

rt = df["Revenue_Tier"].value_counts()
axes[2].bar(rt.index, rt.values, color=PALETTE[:len(rt)], edgecolor="white")
axes[2].set_title("Revenue Tier Distribution", **FONT_TITLE)
axes[2].set_ylabel("Count", **FONT_AXIS)

plt.tight_layout()
save("chart01_target_distributions")




# chart 9 — segmentation: elbow + pca + profiles

print("chart 9: segmentation overview")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Customer Segmentation (K-Means)", **FONT_TITLE)

# elbow with davies-bouldin on second axis
axes[0].plot(seg["k_range"], seg["inertia"], "o-",
             color=BLUE, linewidth=2, markersize=6)
axes[0].axvline(seg["best_k"], color=RED, linestyle="--",
                label=f"chosen k = {seg['best_k']}")
axes[0].set_title("Elbow Method", **FONT_TITLE)
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("Inertia")
axes[0].legend()

# davies-bouldin on twin axis
ax0b = axes[0].twinx()
ax0b.plot(seg["k_range"], seg["db_scores"], "s--",
          color=ORANGE, linewidth=1.5, alpha=0.8, label="Davies-Bouldin")
ax0b.set_ylabel("Davies-Bouldin (lower=better)", color=ORANGE)
ax0b.tick_params(axis="y", labelcolor=ORANGE)

# pca scatter — one dot per customer coloured by segment
pca_coords = seg["pca_coords"]
label_map  = seg["label_map"]
seg_order  = ["Champions", "Loyal Customers",
              "Potential Loyalists", "At Risk", "Need Attention"]
seg_colors = ["#4C72B0", "#55A868", "#8172B2", "#DD8452", "#C44E52"]

for i, lbl in enumerate(seg_order):
    if lbl in df_seg["Segment_Label"].values:
        mask = df_seg["Segment_Label"] == lbl
        axes[1].scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                        s=4, alpha=0.4, c=seg_colors[i], label=lbl)
pca_obj = seg["pca"]
axes[1].set_title("PCA 2D — Customer Segments", **FONT_TITLE)
axes[1].set_xlabel(f"PC1 ({pca_obj.explained_variance_ratio_[0]:.1%} var)")
axes[1].set_ylabel(f"PC2 ({pca_obj.explained_variance_ratio_[1]:.1%} var)")
axes[1].legend(markerscale=4, fontsize=7)

# normalised cluster profile bar chart
profile  = seg["profile"]
plot_cols = ["Recency", "Frequency", "Monetary",
             "Loyalty_Score", "Customer_Satisfaction"]
prof_sub  = profile[plot_cols]
prof_norm = (prof_sub - prof_sub.min()) / (prof_sub.max() - prof_sub.min() + 1e-8)
prof_norm.T.plot(kind="bar", ax=axes[2],
                 color=seg_colors[:len(prof_norm)], edgecolor="white")
axes[2].set_title("Normalised Cluster Profiles", **FONT_TITLE)
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, fontsize=9)
axes[2].legend(fontsize=7)
axes[2].set_ylabel("Normalised value (0–1)")

plt.tight_layout()
save("chart09_segmentation")




# ================================================================
# chart 11 — clv regression evaluation
# note: High_Value_Flag and Risk_Flag removed as leakage
#       R² = 0.84 without leakage — still excellent
# ================================================================
print("chart 11: CLV regression evaluation")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f"CLV Regression — {clv['model_name']} "
             f"(leakage removed, R²={clv['best_metrics']['r2_log']:.2f})",
             **FONT_TITLE)

y_test_log = clv["y_test"]
best_pred  = clv["best_pred"]
residuals  = clv["residuals"]

# predicted vs actual
axes[0, 0].scatter(y_test_log, best_pred, s=4, alpha=0.25, color=BLUE)
mn, mx = float(y_test_log.min()), float(y_test_log.max())
axes[0, 0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="perfect fit")
axes[0, 0].set_title("Predicted vs Actual (log CLV)", **FONT_TITLE)
axes[0, 0].set_xlabel("Actual log(CLV)")
axes[0, 0].set_ylabel("Predicted log(CLV)")
axes[0, 0].legend()
r2 = clv["best_metrics"]["r2_log"]
axes[0, 0].text(0.05, 0.93, f"R² = {r2:.4f}",
                transform=axes[0, 0].transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

# residuals
axes[0, 1].scatter(best_pred, residuals, s=4, alpha=0.25, color=ORANGE)
axes[0, 1].axhline(0, color=RED, linestyle="--", linewidth=1.5)
axes[0, 1].set_title("Residual Plot", **FONT_TITLE)
axes[0, 1].set_xlabel("Predicted log(CLV)")
axes[0, 1].set_ylabel("Residual")

# feature importance — top 15 clean features
best_model_obj = clv["best_model"]
if hasattr(best_model_obj, "feature_importances_"):
    fi = pd.Series(best_model_obj.feature_importances_,
                   index=clv["features"]).nlargest(15)
    axes[1, 0].barh(fi.index[::-1], fi.values[::-1], color=GREEN)
    axes[1, 0].set_title("Top 15 Feature Importances\n(no leakage)", **FONT_TITLE)
    axes[1, 0].set_xlabel("Importance")

# model comparison
comp   = clv["comparison"]
names  = list(comp.keys())
scores = list(comp.values())
bars   = axes[1, 1].bar(names, scores,
                        color=PALETTE[:len(names)], edgecolor="white")
axes[1, 1].set_title("Model Comparison — R² (log scale)", **FONT_TITLE)
axes[1, 1].set_ylabel("R² Score")
axes[1, 1].set_ylim(0, 1)
for bar, val in zip(bars, scores):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=10)

plt.tight_layout()
save("chart11_clv_evaluation")





