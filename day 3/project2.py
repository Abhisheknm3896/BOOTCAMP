# pyrefly: ignore [missing-import] 
from openpyxl.workbook import defined_name
import pandas as pd  
# pyrefly: ignore [missing-import]
import numpy as np  

from sklearn.impute import SimpleImputer  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.linear_model import LinearRegression  
from sklearn.feature_selection import SelectKBest, mutual_info_regression  
import os 
try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder=None
    print("warning: category_encoders not installed")

def main():  
    print("Loading IPL Batting Dataset")  
    file_path = "batting_stats.csv"  
    if not os.path.exists(file_path):  
        print(f"Error: batting_stats.csv not found: '{file_path}'")  
        return  
    df = pd.read_csv(file_path)  
    print(f"Dataset Loaded Successfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n") 
    print(df.head(5))
    print(df.tail(5))
    print(df.describe())

    if TargetEncoder is not None:
        print("Applying Target Encoder")
        encoder = TargetEncoder()
        df["Team_Encoded"] = encoder.fit_transform(df["team"], df["runs"])
        print(df[["team", "Team_Encoded"]].head())
    else:
        print("Category Encoder not installed")
    print("\n")
    #---------feature selection---------
    feature_to_test = ["Team_Encoded","average", "strike_rate", "fours", "sixes"]
    x_features = df[feature_to_test].fillna(0)
    y_target = df["runs"]
    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(x_features, y_target)
    winning_features = selector.get_support()
    best_features = x_features.columns[winning_features].tolist()
    print(best_features)
    print("\n")

    # ---------- Splitting the Dataset ----------
    X = df[best_features]  
    Y = df["runs"]  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=17)
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}\n")

    model=LinearRegression()

    model.fit(X_train,Y_train)
    
    prediction = model.predict(X_test)
    print("Prediction")
    print(prediction)
    print("\n")
    
    
    actual_wins = Y_test.head(3).values
    predicted_wins = prediction[:3]
    
    for i in range(3):
        predicted = round(predicted_wins[i])
        actual = actual_wins[i]
        difference=abs(actual-predicted)
        print("\n")
        print(f"Model Gussed: {predicted}")
        print(f"Real Answer: {actual}")
        print(f"Difference: {difference}")

        

if __name__ == "__main__":
    main()