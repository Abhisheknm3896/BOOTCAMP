import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import os

# Category_encoder is needed for Target Encoding test

try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:   
    TargetEncoder = None
    print("warning : category_encoders not installed.Target Encoding will not work")

def main():

    print("Loading datasets")
    file_path="train.csv"
    if not os.path.exists(file_path):
        print(f"Error: train.csv not found: '{file_path}'")
        return

    
    df=pd.read_csv(file_path)
    print(f"Dataset Loaded Sucessfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n")


if __name__ == "__main__":
    main()
    

# Handling Missing Values
print("Artifically creating missing data for the demonstration")    
