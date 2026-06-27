import pandas as pd  # Import pandas library to handle tables of data
# pyrefly: ignore [missing-import]
import numpy as np  # Import numpy library for fast math operations

from sklearn.impute import SimpleImputer  # Import tool to fill in missing values
from sklearn.model_selection import train_test_split  # Import tool to split data for training and testing
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import tools to encode categories and scale numbers
from sklearn.linear_model import LinearRegression  # Import the linear regression prediction model
import os  # Import os module to check files on your computer

# Category_encoder is needed for Target Encoding test

try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder  # Try to import advanced target encoder
except ImportError:   
    TargetEncoder = None  # If not installed, set encoder to None
    print("warning : category_encoders not installed.Target Encoding will not work")  # Print a warning

def main():  # Define the main function to run the process

    print("Loading datasets")  # Print that dataset loading has started
    file_path="train.csv"  # Set the filename of the data to "train.csv"
    if not os.path.exists(file_path):  # Check if train.csv does not exist
        print(f"Error: train.csv not found: '{file_path}'")  # Print error if file is missing
        return  # Stop the function if file is missing

    
    df=pd.read_csv(file_path)  # Read the CSV file into a table (DataFrame)
    print(f"Dataset Loaded Sucessfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n")  # Print row and column count

    print("HANDLING MISSING DATA")
    print("Artificially deleting some 'Hits' (H) data to demonstrate")

    df.loc[0.25, 'H'] = np.nan
    imputer = SimpleImputer(strategy='median')
    df['H'] = imputer.fit_transform(df[['H']])
    print("Imputation completed 'Hits' (H)")

    print(df.head())
    print(df.tail())

    print("\nBasic Statistics:")
    print(df.describe())

if __name__ == "__main__":  # Check if this script is being run directly
    main()  # Run the main function
    

# Handling Missing Values
print("Artifically creating missing data for the demonstration")    
