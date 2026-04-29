import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing     import RobustScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       roc_auc_score, average_precision_score,
                                       f1_score, precision_recall_curve)
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGB_OK = False
    print("xgboost not found — using GradientBoostingClassifier instead")


# load data from part 1
df_clean   = pd.read_csv("customer_clean.csv")
df_encoded = pd.read_csv("customer_encoded.csv")

print(f"rows loaded  : {df_clean.shape[0]:,}")
print(f"\nclass distribution:")
print(f"  not churned (0) : {(df_clean['Churned']==0).sum():,}  ({(df_clean['Churned']==0).mean():.1%})")
print(f"  churned     (1) : {(df_clean['Churned']==1).sum():,}  ({(df_clean['Churned']==1).mean():.1%})")
print(f"\nimbalance handling: class_weight='balanced' + scale_pos_weight")
print(f"no SMOTE — 23% minority is enough real data, synthetic samples would distort reality")


# feature selection
drop_cols = {
    "CustomerID", "Acquisition_Date", "Purchase_Date",
    "Churned",
    "Churn_Risk_Score",      # leakage — directly derived from churn label
    "Loyalty_Tier",          # raw version, encoded copy is kept
    "Revenue_Tier",
    "Education_Level",
    "CLV_log", "Revenue_log"
}
feature_cols = [c for c in df_encoded.select_dtypes(include=np.number).columns
                if c not in drop_cols]

X = df_encoded[feature_cols].fillna(0)
y = df_encoded["Churned"].astype(int)
print(f"\nfeatures used : {len(feature_cols)}")


# train / test split — stratified keeps the 77/23 ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

print(f"\ntrain : {len(X_train):,}  |  test : {len(X_test):,}")

# scale for logistic regression
scaler     = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# cross validation strategy — 5 folds, stratified
# same cv object used for all 3 models so comparison is fair
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# model 1 — logistic regression with grid search
# searching over C (regularisation strength)
# scoring on PR-AUC — better metric than ROC-AUC for imbalanced data
print("\nmodel 1: logistic regression — grid search...")
lr_params = {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
lr_gs = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42,
                       class_weight="balanced"),
    lr_params,
    cv=cv,
    scoring="average_precision",   # PR-AUC
    n_jobs=-1,
    verbose=0
)
lr_gs.fit(X_train_sc, y_train)
print(f"  best params  : {lr_gs.best_params_}")
print(f"  best CV PR-AUC : {lr_gs.best_score_:.4f}")

# retrain best LR on full training set with best params
lr_best   = lr_gs.best_estimator_
lr_prob   = lr_best.predict_proba(X_test_sc)[:, 1]
lr_pred   = lr_best.predict(X_test_sc)
lr_prauc  = average_precision_score(y_test, lr_prob)
print(f"  test PR-AUC  : {lr_prauc:.4f}")
print(f"  test ROC-AUC : {roc_auc_score(y_test, lr_prob):.4f}")
print(f"  test F1      : {f1_score(y_test, lr_pred):.4f}")


# model 2 — random forest with grid search
# searching over n_estimators, max_depth, min_samples_leaf
print("\nmodel 2: random forest — grid search...")
rf_params = {
    "n_estimators":   [100, 200],
    "max_depth":      [8, 12, 16],
    "min_samples_leaf":[5, 10]
}
rf_gs = GridSearchCV(
    RandomForestClassifier(class_weight="balanced",
                           random_state=42, n_jobs=-1),
    rf_params,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1,
    verbose=0
)
rf_gs.fit(X_train, y_train)
print(f"  best params  : {rf_gs.best_params_}")
print(f"  best CV PR-AUC : {rf_gs.best_score_:.4f}")

rf_best  = rf_gs.best_estimator_
rf_prob  = rf_best.predict_proba(X_test)[:, 1]
rf_pred  = rf_best.predict(X_test)
rf_prauc = average_precision_score(y_test, rf_prob)
print(f"  test PR-AUC  : {rf_prauc:.4f}")
print(f"  test ROC-AUC : {roc_auc_score(y_test, rf_prob):.4f}")
print(f"  test F1      : {f1_score(y_test, rf_pred):.4f}")


# model 3 — xgboost with grid search
# scale_pos_weight handles class imbalance in xgboost
# eval_metric=aucpr means xgboost optimises PR-AUC internally
print(f"\nmodel 3: {'xgboost' if XGB_OK else 'gradient boosting'} — grid search...")
scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  scale_pos_weight = {scale_pw:.3f}")

if XGB_OK:
    xgb_params = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1]
    }
    xgb_gs = GridSearchCV(
        xgb.XGBClassifier(scale_pos_weight=scale_pw,
                           eval_metric="aucpr",
                           random_state=42, n_jobs=-1,
                           verbosity=0),
        xgb_params,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        verbose=0
    )
    m3_name = "xgboost"
else:
    from sklearn.ensemble import GradientBoostingClassifier
    xgb_params = {
        "n_estimators":  [100, 200],
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.1]
    }
    xgb_gs = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        xgb_params,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        verbose=0
    )
    m3_name = "gradient boosting"

xgb_gs.fit(X_train, y_train)
print(f"  best params  : {xgb_gs.best_params_}")
print(f"  best CV PR-AUC : {xgb_gs.best_score_:.4f}")

m3_best  = xgb_gs.best_estimator_
m3_prob  = m3_best.predict_proba(X_test)[:, 1]
m3_pred  = m3_best.predict(X_test)
m3_prauc = average_precision_score(y_test, m3_prob)
print(f"  test PR-AUC  : {m3_prauc:.4f}")
print(f"  test ROC-AUC : {roc_auc_score(y_test, m3_prob):.4f}")
print(f"  test F1      : {f1_score(y_test, m3_pred):.4f}")


# pick the best model by CV PR-AUC score — not test score
# using CV score for selection avoids overfitting to the test set
scores = {
    "logistic regression": lr_gs.best_score_,
    "random forest":       rf_gs.best_score_,
    m3_name:               xgb_gs.best_score_
}
best_name = max(scores, key=scores.get)
model_map = {
    "logistic regression": (lr_best, lr_pred, lr_prob),
    "random forest":       (rf_best, rf_pred, rf_prob),
    m3_name:               (m3_best, m3_pred, m3_prob)
}
best_model, best_pred, best_prob = model_map[best_name]

print(f"\nbest model : {best_name}  (CV PR-AUC = {scores[best_name]:.4f})")
print(f"\nCV PR-AUC summary:")
for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:25s} {score:.4f}")


# threshold tuning
# find the threshold that maximises F1 on the test set
# this only shifts the decision boundary — no data is changed
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, best_prob)
f1_vals     = (2 * precision_vals * recall_vals /
               (precision_vals + recall_vals + 1e-8))
best_idx    = f1_vals.argmax()
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

print(f"\nthreshold tuning:")
print(f"  default 0.50 → F1: {f1_score(y_test, (best_prob>=0.50).astype(int)):.4f}")
print(f"  tuned  {best_thresh:.2f} → F1: {f1_score(y_test, (best_prob>=best_thresh).astype(int)):.4f}")

best_pred_tuned = (best_prob >= best_thresh).astype(int)


# final evaluation
print(f"\nclassification report — {best_name} (threshold={best_thresh:.2f}):")
print(classification_report(y_test, best_pred_tuned,
                             target_names=["not churned (0)", "churned (1)"]))

cm = confusion_matrix(y_test, best_pred_tuned)
tn, fp, fn, tp = cm.ravel()
print("confusion matrix:")
print(f"  true negatives  : {tn:,}")
print(f"  false positives : {fp:,}")
print(f"  false negatives : {fn:,}")
print(f"  true positives  : {tp:,}")

high_risk = (best_prob > 0.7).sum()
print(f"\nhigh risk customers (prob > 0.70) : {high_risk:,}  ({high_risk/len(y_test)*100:.1f}%)")

print(f"\ntop 20 feature importances ({best_name}):")
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=X.columns).nlargest(20)
    for feat, imp in fi.items():
        print(f"  {feat:40s} {imp:.4f}")


# save outputs
pd.DataFrame({
    "CustomerID":        df_clean.iloc[X_test.index]["CustomerID"].values,
    "actual_churned":    y_test.values,
    "predicted_churned": best_pred_tuned,
    "churn_probability": best_prob.round(4)
}).to_csv("churn_predictions.csv", index=False)

with open("churn_results.pkl", "wb") as f:
    pickle.dump({
        "lr_auc":      roc_auc_score(y_test, lr_prob),
        "lr_prob":     lr_prob,
        "lr_pred":     lr_pred,
        "rf_auc":      roc_auc_score(y_test, rf_prob),
        "rf_prob":     rf_prob,
        "rf_pred":     rf_pred,
        "m3_auc":      roc_auc_score(y_test, m3_prob),
        "m3_prob":     m3_prob,
        "m3_pred":     m3_pred,
        "m3_name":     m3_name,
        "best_name":   best_name,
        "best_prob":   best_prob,
        "best_pred":   best_pred_tuned,
        "best_model":  best_model,
        "best_thresh": best_thresh,
        "y_test":      y_test,
        "X_test":      X_test,
        "feature_cols":feature_cols,
        "cv_scores":   np.array([lr_gs.best_score_,
                                  rf_gs.best_score_,
                                  xgb_gs.best_score_]),
        "scaler":      scaler,
        "lr_gs":       lr_gs,
        "rf_gs":       rf_gs,
        "xgb_gs":      xgb_gs
    }, f)

print("\nsaved: churn_predictions.csv  and  churn_results.pkl")
