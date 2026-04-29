import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing   import RobustScaler
from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGB_OK = False
    print("xgboost not found — using GradientBoostingRegressor instead")


# load data from part 1
df_clean   = pd.read_csv("customer_clean.csv")
df_encoded = pd.read_csv("customer_encoded.csv")

print(f"rows loaded : {df_clean.shape[0]:,}")
print(f"CLV range   : {df_clean['Customer_Lifetime_Value'].min():,.0f}"
      f" – {df_clean['Customer_Lifetime_Value'].max():,.0f}")
print(f"CLV mean    : {df_clean['Customer_Lifetime_Value'].mean():,.0f}")


# feature selection
# High_Value_Flag removed — created from CLV itself (direct leakage)
# Risk_Flag removed     — derived from Churn_Risk_Score (potential leakage)
# CLV_to_CAC_Ratio removed — directly calculated from CLV
drop_cols = {
    "CustomerID", "Acquisition_Date", "Purchase_Date",
    "Customer_Lifetime_Value",
    "CLV_log",
    "CLV_to_CAC_Ratio",     # leakage — calculated from CLV
    "High_Value_Flag",      # leakage — created as (CLV > CLV.median())
    "Risk_Flag",            # leakage — derived from Churn_Risk_Score
    "Loyalty_Tier", "Revenue_Tier", "Education_Level"
}
feature_cols = [c for c in df_encoded.select_dtypes(include=np.number).columns
                if c not in drop_cols]

# log-transform CLV — makes distribution normal, improves regression
y = np.log1p(df_encoded["Customer_Lifetime_Value"])
X = df_encoded[feature_cols].fillna(0)

print(f"\nfeatures used : {len(feature_cols)}")
print(f"target        : log(CLV)  mean={y.mean():.3f}  std={y.std():.3f}")


# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(f"\ntrain : {len(X_train):,}  |  test : {len(X_test):,}")

# scale for ridge
scaler     = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# cross validation — 5 folds
# same cv used for all models so the comparison is fair
cv = KFold(n_splits=5, shuffle=True, random_state=42)


# helper — print metrics on both log and original scale
def show_metrics(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2o  = r2_score(np.expm1(y_true), np.expm1(y_pred))
    maeo = mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))
    rmseo= float(np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred))))
    print(f"  log scale  → R²:{r2:.4f}  MAE:{mae:.4f}  RMSE:{rmse:.4f}")
    print(f"  orig scale → R²:{r2o:.4f}  MAE:{maeo:,.0f}  RMSE:{rmseo:,.0f}")
    return {"r2_log":r2,"mae_log":mae,"rmse_log":rmse,
            "r2_orig":r2o,"mae_orig":maeo,"rmse_orig":rmseo}


# model 1 — ridge regression with grid search
# searching over alpha (regularisation strength)
print("\nmodel 1: ridge regression — grid search...")
ridge_params = {"alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}
ridge_gs = GridSearchCV(
    Ridge(),
    ridge_params,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    verbose=0
)
ridge_gs.fit(X_train_sc, y_train)
print(f"  best params    : {ridge_gs.best_params_}")
print(f"  best CV R²     : {ridge_gs.best_score_:.4f}")

ridge_best    = ridge_gs.best_estimator_
ridge_pred    = ridge_best.predict(X_test_sc)
ridge_metrics = show_metrics("ridge", y_test, ridge_pred)


# model 2 — random forest with grid search
print("\nmodel 2: random forest regressor — grid search...")
rf_params = {
    "n_estimators":    [100, 200],
    "max_depth":       [10, 14, 18],
    "min_samples_leaf":[3, 5]
}
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_params,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    verbose=0
)
rf_gs.fit(X_train, y_train)
print(f"  best params    : {rf_gs.best_params_}")
print(f"  best CV R²     : {rf_gs.best_score_:.4f}")

rf_best    = rf_gs.best_estimator_
rf_pred    = rf_best.predict(X_test)
rfr_metrics = show_metrics("random forest", y_test, rf_pred)


# model 3 — xgboost with grid search
print(f"\nmodel 3: {'xgboost' if XGB_OK else 'gradient boosting'} — grid search...")
if XGB_OK:
    xgb_params = {
        "n_estimators":  [200, 300, 400],
        "max_depth":     [4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05]
    }
    xgb_gs = GridSearchCV(
        xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=0.1, reg_lambda=1.0,
                          random_state=42, n_jobs=-1, verbosity=0),
        xgb_params,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=0
    )
    m3_name = "xgboost"
else:
    from sklearn.ensemble import GradientBoostingRegressor
    xgb_params = {
        "n_estimators":  [200, 300],
        "max_depth":     [4, 6],
        "learning_rate": [0.03, 0.05]
    }
    xgb_gs = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        xgb_params,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=0
    )
    m3_name = "gradient boosting"

xgb_gs.fit(X_train, y_train)
print(f"  best params    : {xgb_gs.best_params_}")
print(f"  best CV R²     : {xgb_gs.best_score_:.4f}")

m3_best    = xgb_gs.best_estimator_
m3_pred    = m3_best.predict(X_test)
m3_metrics = show_metrics(m3_name, y_test, m3_pred)


# pick the best model by CV R² score
# using CV score for selection avoids choosing based on test set performance
scores = {
    "ridge":         ridge_gs.best_score_,
    "random forest": rf_gs.best_score_,
    m3_name:         xgb_gs.best_score_
}
best_name = max(scores, key=scores.get)
model_map = {
    "ridge":         (ridge_best, ridge_pred, ridge_metrics),
    "random forest": (rf_best,    rf_pred,    rfr_metrics),
    m3_name:         (m3_best,    m3_pred,    m3_metrics)
}
best_reg, best_pred, best_metrics = model_map[best_name]

print(f"\nbest model : {best_name}  (CV R² = {scores[best_name]:.4f})")
print(f"\nCV R² summary:")
for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} {score:.4f}")


# feature importance
print(f"\ntop 20 feature importances ({best_name}):")
if hasattr(best_reg, "feature_importances_"):
    fi = pd.Series(best_reg.feature_importances_, index=X.columns).nlargest(20)
    for feat, imp in fi.items():
        print(f"  {feat:40s} {imp:.4f}")


# residual check
residuals = y_test - best_pred
print(f"\nresiduals — mean: {residuals.mean():.4f}  std: {residuals.std():.4f}")
print(f"  mean close to 0 means the model is unbiased")


# newsletter and weekend impact on predicted CLV
# note: these show similar values (~20.7K) because CLV is driven
# by financial features, not these binary behavioural columns
df_clean["CLV_Predicted_log"] = best_reg.predict(
    scaler.transform(df_clean[feature_cols].fillna(0))
    if best_name == "ridge"
    else df_clean[feature_cols].fillna(0)
)
df_clean["CLV_Predicted"] = np.expm1(df_clean["CLV_Predicted_log"])

ns  = df_clean.groupby("Newsletter_Subscribed")["CLV_Predicted"].mean().round(0)
iw  = df_clean.groupby("Is_Weekend")["CLV_Predicted"].mean().round(0)
print(f"\npredicted CLV by newsletter subscribed:")
print(f"  not subscribed (0) : {ns.get(0,0):,.0f}")
print(f"  subscribed     (1) : {ns.get(1,0):,.0f}")
print(f"\npredicted CLV by is_weekend:")
print(f"  weekday (0) : {iw.get(0,0):,.0f}")
print(f"  weekend (1) : {iw.get(1,0):,.0f}")

tier_order = ["Bronze", "Silver", "Gold", "Platinum"]
tier_clv   = df_clean.groupby("Loyalty_Tier")["CLV_Predicted"].mean().reindex(tier_order).round(0)
print(f"\npredicted CLV by loyalty tier:")
print(tier_clv.to_string())


# save outputs
pd.DataFrame({
    "CustomerID":    df_clean.iloc[X_test.index]["CustomerID"].values,
    "actual_CLV":    np.expm1(y_test.values).round(0).astype(int),
    "predicted_CLV": np.expm1(best_pred).round(0).astype(int),
    "residual":      residuals.values.round(4)
}).to_csv("clv_predictions.csv", index=False)

df_clean.to_csv("customer_clean_with_predictions.csv", index=False)

comparison = {
    "ridge":         ridge_metrics["r2_log"],
    "random forest": rfr_metrics["r2_log"],
    m3_name:         m3_metrics["r2_log"]
}

with open("clv_results.pkl", "wb") as f:
    pickle.dump({
        "best_model":    best_reg,
        "scaler":        scaler,
        "features":      feature_cols,
        "model_name":    best_name,
        "y_test":        y_test,
        "best_pred":     best_pred,
        "ridge_pred":    ridge_pred,
        "rfr_pred":      rf_pred,
        "m3_pred":       m3_pred,
        "m3_name":       m3_name,
        "ridge_metrics": ridge_metrics,
        "rfr_metrics":   rfr_metrics,
        "m3_metrics":    m3_metrics,
        "best_metrics":  best_metrics,
        "residuals":     residuals,
        "cv_scores":     np.array([ridge_gs.best_score_,
                                    rf_gs.best_score_,
                                    xgb_gs.best_score_]),
        "comparison":    comparison,
        "ridge_gs":      ridge_gs,
        "rf_gs":         rf_gs,
        "xgb_gs":        xgb_gs
    }, f)

print("\nsaved: clv_predictions.csv  |  customer_clean_with_predictions.csv  |  clv_results.pkl")
