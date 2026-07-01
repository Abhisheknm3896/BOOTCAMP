from ast import Compare
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

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
    classification_report
)

data = load_breast_cancer(as_frame=True)
df = data.frame
print(df.head())

X=df.drop("target",axis=1)
y=df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)

print("accuracy score:",accuracy_score(y_test,rf_pre))
print("\nclassfication_report\n",classification_report(y_test,rf_pre))


xgb=XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pre=xgb.predict(X_test)

print("accuracy score:",accuracy_score(y_test,xgb_pre))
print("\nclassfication_report\n",classification_report(y_test,xgb_pre))

#build lightGBM
import pandas as pd
from sklearn.metrics import accuracy_score

from ast import Compare
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

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
    classification_report
)

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