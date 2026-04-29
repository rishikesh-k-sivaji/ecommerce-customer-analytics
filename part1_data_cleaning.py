import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# load the dataset
df = pd.read_csv("customer.csv")
print(f"loaded {df.shape[0]:,} rows and {df.shape[1]} columns")

# basic checks
print(f"\nmissing values : {df.isnull().sum().sum()}")
print(f"duplicate rows : {df.duplicated(subset='CustomerID').sum()}")

# confirm binary columns only have 0 and 1
binary_cols = ["Newsletter_Subscribed", "Is_Weekend", "Churned"]
print("\nbinary column check:")
for col in binary_cols:
    vals   = sorted(df[col].unique().tolist())
    counts = df[col].value_counts().sort_index()
    print(f"  {col}: values={vals}  |  0→{counts.get(0,0):,}  1→{counts.get(1,0):,}")


# remove duplicates
df = df.drop_duplicates(subset="CustomerID", keep="first")

# parse dates
df["Acquisition_Date"] = pd.to_datetime(df["Acquisition_Date"], errors="coerce")
df["Purchase_Date"]    = pd.to_datetime(df["Purchase_Date"],    errors="coerce")

# clip outliers using 1st to 99th percentile
clip_cols = ["Revenue", "Customer_Lifetime_Value", "Annual_Income",
             "Net_Revenue", "Unit_Price"]
print("\nclipping outliers:")
for col in clip_cols:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)
    print(f"  {col} clipped to [{lo:,.0f} – {hi:,.0f}]")


# feature engineering
df["Recency"]              = df["Days_Since_Last_Purchase"]
df["Frequency"]            = df["Purchase_Frequency"]
df["Monetary"]             = df["Revenue"]
df["CLV_log"]              = np.log1p(df["Customer_Lifetime_Value"])
df["Revenue_log"]          = np.log1p(df["Revenue"])
df["Engagement_Composite"] = (df["Email_Open_Rate"]         * 0.40 +
                               df["Social_Media_Engagement"] * 0.35 +
                               df["Newsletter_Subscribed"]   * 0.25)
df["Discount_Impact"]      = df["Discount_Applied"] * df["Revenue"]
df["Revenue_per_Visit"]    = df["Revenue"] / (df["Purchase_Frequency"] + 1)
df["Weekend_Engagement"]   = df["Is_Weekend"] * df["Engagement_Composite"]

# High_Value_Flag and Risk_Flag were removed —
# High_Value_Flag was derived from CLV (the regression target) — direct leakage
# Risk_Flag was derived from Churn_Risk_Score — potential leakage for churn model

print("\n9 new features added")

# ordinal encoding
df["Education_Level_enc"] = df["Education_Level"].map(
    {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3})
df["Loyalty_Tier_enc"]    = df["Loyalty_Tier"].map(
    {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3})
df["Revenue_Tier_enc"]    = df["Revenue_Tier"].map(
    {"Low": 0, "Medium": 1, "High": 2, "Premium": 3})

# one-hot encoding for nominal columns
nominal_cols = [
    "Customer_Segment", "Gender", "Job_Category", "Marital_Status",
    "Product", "Payment_Method", "Season", "Marketing_Channel",
    "First_Purchase_Channel", "Age_Group", "Acquisition_Month"
]
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

print(f"\nfinal clean shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"final encoded shape : {df_encoded.shape[0]:,} rows × {df_encoded.shape[1]} columns")

print("\nkey stats:")
print(df[["Age", "Annual_Income", "Revenue", "Customer_Lifetime_Value",
          "Loyalty_Score", "Customer_Satisfaction"]].describe().round(1).to_string())

print(f"\nchurn          — 0: {(df['Churned']==0).sum():,}  |  1: {(df['Churned']==1).sum():,}")
print(f"newsletter     — 0: {(df['Newsletter_Subscribed']==0).sum():,}  |  1: {(df['Newsletter_Subscribed']==1).sum():,}")
print(f"weekend buyers — 0: {(df['Is_Weekend']==0).sum():,}  |  1: {(df['Is_Weekend']==1).sum():,}")

# save both files — other parts depend on these
df.to_csv("customer_clean.csv", index=False)
df_encoded.to_csv("customer_encoded.csv", index=False)
print("\nsaved: customer_clean.csv  and  customer_encoded.csv")
