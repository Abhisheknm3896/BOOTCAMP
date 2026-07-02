from sklearn.metrics import roc_curve
from sklearn import linear_model
from sklearn.model_selection import _split
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from pandas import plotting
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from ast import Compare
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
# pyrefly: ignore [missing-import]
from lightgbm import LGBMClassifier
# pyrefly: ignore [missing-import]
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
# pyrefly: ignore [missing-import]
import optuna
data = load_breast_cancer(as_frame=True)
df = data.frame
print(df.head())

X=df.drop("target",axis=1)
y=df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#build randomforest
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)

print("accuracy score:",accuracy_score(y_test,rf_pre))
print("\nclassfication_report\n",classification_report(y_test,rf_pre))

#build Xgboost
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pre=xgb.predict(X_test)

print("accuracy score:",accuracy_score(y_test,xgb_pre))
print("\nclassfication_report\n",classification_report(y_test,xgb_pre))

#build lightGBM
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
lgbm_pre = lgbm.predict(X_test)

print("\nLightGBM")
print("Accuracy Score:", accuracy_score(y_test, lgbm_pre))
print("\nClassification Report\n")
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
print(Comparison)

precision= precision_score(y_test,rf_pre)
recall = recall_score(y_test,rf_pre)
f1 = f1_score(y_test,rf_pre)

print(f"Precision {precision}")
print(f"Recall {recall}")
print(f"f1 {f1}")

cm = confusion_matrix(y_test,rf_pre)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)
disp.plot(cmap="Purples")
plt.title("Confusion matrix")
plt.show()

y_pre = rf.predict(X_test)

y_prob=rf.predict_proba(X_test)[:,1]
roc_score = roc_auc_score(y_test,y_prob)

print("roc_auc_score",roc_score)


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_score:.3f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 1. baseline
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
print("Baseline Accuracy:",accuracy_score(y_test,y_preds))


#2.GridSearchCV

param_grid={
    "n_estimators":[100,200,300],
    "max_depth":[3,4,5],
    "learning_rate":[0.1,0.01,0.001],
}
grid= GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
)
grid.fit(X_train,y_train)
print("\nBest Parameters")
print(grid.best_params_)
best_model=grid.best_estimator_
best_model_pred=best_model.predict(X_test)

print("\nBest Model Accuracy:",accuracy_score(y_test,best_model_pred))

# 3. Random Search(simple)

random_param={
    "n_estimators":[100,200,300],
    "max_depth":[3,4,5],
    "learning_rate":[0.1,0.01,0.001],
}
random_search=RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_distributions=random_param,
    n_iter=10,
    cv=3,
    random_state=42,
)
random_search.fit(X_train,y_train)
print("\nRandomized Search Best Parameters")
print(random_search.best_params_)
best_model_rand=random_search.best_estimator_
prediction =best_model_rand.predict(X_test)

print("\nRandomized Search Best Model Accuracy:",accuracy_score(y_test,prediction))

#optuna

def objective(trail):
    model = XGBClassifier(
        n_estimators=trail.suggest_int("n_estimators",50,200),
        max_depth=trail.suggest_int("max_depth",3,7),
        learning_rate=trail.suggest_float("learning_rate",0.01,0.2),
        random_state=42
    )
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    return accuracy_score(y_test,prediction)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(study.best_params)
best_model = XGBClassifier(
    **study.best_params,
    random_state=42
)
best_model.fit(X_train,y_train)
prediction = best_model.predict(X_test)
print("Optuna Best Model Accuracy",accuracy_score(y_test,prediction))

from sklearn.tree import DecisionTreeClassifier
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
dt_model = DecisionTreeClassifier(class_weight="balanced")
dt_model.fit(X_train, y_train)

# XGBoost 
ratio = len(y_train[y_train==0])/len(y_train[y_train==1])
xgb_model = XGBClassifier(scale_pos_weight=ratio, random_state=42)
xgb_model.fit(X_train, y_train)