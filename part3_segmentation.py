import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import KMeans
from sklearn.decomposition import PCA


# load data produced by part 1
df = pd.read_csv("customer_clean.csv")
print(f"rows loaded : {df.shape[0]:,}")


# features used for clustering
cluster_features = [
    "Recency", "Frequency", "Monetary",
    "Loyalty_Score", "Customer_Satisfaction",
    "Email_Open_Rate", "Social_Media_Engagement",
    "Annual_Income", "CLV_to_CAC_Ratio",
    "Engagement_Composite", "Support_Tickets",
    "Days_Since_Last_Purchase", "Purchase_Frequency",
    "Newsletter_Subscribed",   # binary 0/1
    "Is_Weekend",              # binary 0/1
    "Churn_Risk_Score", "Return_Rate"
]

X = df[cluster_features].fillna(0)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)


# evaluate k=2 to k=10 using two metrics:
#   inertia        — measures how tight the clusters are (lower = better)
#   davies-bouldin — measures how well separated clusters are (lower = better)
# we use both because silhouette alone gave k=2 which only splits
# customers into high vs low CLV — too coarse to be useful in practice

from sklearn.metrics import davies_bouldin_score

print("\nevaluating k=2 to k=10:")
print(f"  {'k':>2}  {'inertia':>12}  {'davies-bouldin':>15}  note")
print("  " + "-" * 50)

inertia   = []
db_scores = []

for k in range(2, 11):
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    db = davies_bouldin_score(X_scaled, labels)
    db_scores.append(db)
    note = "<-- silhouette pick (too coarse)" if k == 2 else ""
    print(f"  {k:>2}  {km.inertia_:>12,.0f}  {db:>15.4f}  {note}")

# pick k using davies-bouldin — the k with the lowest score
# has the best separated, most distinct clusters
best_k_db = list(range(2, 11))[db_scores.index(min(db_scores))]
K         = best_k_db

print(f"\nk selected : {K}  (davies-bouldin = {min(db_scores):.4f} — lowest across all k)")
print(f"k=2 rejected: davies-bouldin = {db_scores[0]:.4f} — clusters not well separated")
print(f"\nfitting k-means with k={K}")
km_final    = KMeans(n_clusters=K, random_state=42, n_init=15)
df["Cluster"] = km_final.fit_predict(X_scaled)

print("cluster sizes:")
print(df["Cluster"].value_counts().sort_index().to_string())


# name the clusters — highest avg CLV = Champions, lowest = Need Attention
clv_mean        = df.groupby("Cluster")["Customer_Lifetime_Value"].mean()
sorted_clusters = clv_mean.sort_values(ascending=False).index.tolist()
segment_names   = ["Champions", "Loyal Customers",
                   "Potential Loyalists", "At Risk", "Need Attention"]
label_map           = {c: segment_names[i] for i, c in enumerate(sorted_clusters)}
df["Segment_Label"] = df["Cluster"].map(label_map)

print("\nsegment mapping:")
for cluster, name in sorted(label_map.items()):
    print(f"  cluster {cluster} → {name}")


# rfm scoring
rfm = df[["CustomerID", "Recency", "Frequency", "Monetary"]].copy()
rfm["R_score"]   = pd.qcut(rfm["Recency"],
                            q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_score"]   = pd.qcut(rfm["Frequency"].rank(method="first"),
                            q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_score"]   = pd.qcut(rfm["Monetary"],
                            q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_Total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

df = df.merge(rfm[["CustomerID", "R_score", "F_score",
                    "M_score", "RFM_Total"]], on="CustomerID")

print("\nrfm score summary:")
print(rfm[["R_score", "F_score", "M_score", "RFM_Total"]].describe().round(2).to_string())


# cluster profiles
profile_cols = [
    "Recency", "Frequency", "Monetary",
    "Loyalty_Score", "Customer_Satisfaction",
    "Annual_Income", "Customer_Lifetime_Value",
    "Churn_Risk_Score", "Churned",
    "Email_Open_Rate", "Social_Media_Engagement",
    "Newsletter_Subscribed", "Is_Weekend",
    "Support_Tickets", "RFM_Total"
]
profile = df.groupby("Segment_Label")[profile_cols].mean().round(2)
print("\ncluster profiles:")
print(profile.to_string())


# business summary
summary = df.groupby("Segment_Label").agg(
    customers      = ("CustomerID",             "count"),
    avg_clv        = ("Customer_Lifetime_Value", "mean"),
    avg_revenue    = ("Revenue",                 "mean"),
    churn_rate     = ("Churned",                 "mean"),
    newsletter_pct = ("Newsletter_Subscribed",   "mean"),
    weekend_pct    = ("Is_Weekend",              "mean"),
    avg_rfm        = ("RFM_Total",               "mean")
).round(2).sort_values("avg_clv", ascending=False)

summary["churn_rate"]     = (summary["churn_rate"]     * 100).round(1).astype(str) + "%"
summary["newsletter_pct"] = (summary["newsletter_pct"] * 100).round(1).astype(str) + "%"
summary["weekend_pct"]    = (summary["weekend_pct"]    * 100).round(1).astype(str) + "%"
summary["avg_clv"]        = summary["avg_clv"].round(0).astype(int)
summary["avg_revenue"]    = summary["avg_revenue"].round(0).astype(int)
print("\nbusiness summary:")
print(summary.to_string())


# pca — reduce to 2 dimensions for scatter plot in visuals
pca        = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
df["PCA_1"] = pca_coords[:, 0]
df["PCA_2"] = pca_coords[:, 1]
print(f"\npca: PC1={pca.explained_variance_ratio_[0]:.2%}  "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}  "
      f"combined={sum(pca.explained_variance_ratio_[:2]):.2%}")


# save outputs
df.to_csv("customer_segments.csv", index=False)

with open("seg_results.pkl", "wb") as f:
    pickle.dump({
        "km_model":         km_final,
        "scaler":           scaler,
        "pca":              pca,
        "pca_coords":       pca_coords,
        "cluster_features": cluster_features,
        "label_map":        label_map,
        "inertia":          inertia,
        "db_scores":        db_scores,
        "k_range":          list(range(2, 11)),
        "best_k":           K,
        "rfm":              rfm,
        "profile":          profile,
        "profile_cols":     profile_cols
    }, f)

print("\nsaved: customer_segments.csv  and  seg_results.pkl")
