import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
# pyrefly: ignore [missing-import]
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("customer_data.csv")

print(df.head())

median = df["LoyaltyPoints"].median()
df["Target"] = (df["LoyaltyPoints"] >= median).astype(int)

df["CustomerSince"] = pd.to_datetime(df["CustomerSince"])
df["CustomerSince"] = (pd.Timestamp.today() - df["CustomerSince"]).dt.days

X = df.drop(["CustomerID", "LoyaltyPoints", "Target"], axis=1)
y = df["Target"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pre = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy Score:", accuracy_score(y_test, rf_pre))
print(classification_report(y_test, rf_pre))

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
xgb_pre = xgb.predict(X_test)

print("\nXGBoost")
print("Accuracy Score:", accuracy_score(y_test, xgb_pre))
print(classification_report(y_test, xgb_pre))

lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
lgbm_pre = lgbm.predict(X_test)

print("\nLightGBM")
print("Accuracy Score:", accuracy_score(y_test, lgbm_pre))
print(classification_report(y_test, lgbm_pre))

Comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost",
        "LightGBM"
    ],
    "Accuracy": [
        accuracy_score(y_test, rf_pre),
        accuracy_score(y_test, xgb_pre),
        accuracy_score(y_test, lgbm_pre)
    ]
})

print("\nModel Comparison")
print(Comparison)

important=pd.Series(
    rf.feature_importances_,
    index=X.columns
)

important.sort_values(
    ascending=False
).head(10).plot.barh()