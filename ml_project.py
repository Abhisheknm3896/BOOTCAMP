# pyrefly: ignore [missing-import]
import pandas as pd  # Import pandas library to handle tables of data
# pyrefly: ignore [missing-import]
import numpy as np  # Import numpy library for fast math operations

from sklearn.impute import SimpleImputer  # Import tool to fill in missing values
from sklearn.model_selection import train_test_split  # Import tool to split data for training and testing
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import tools to encode categories and scale numbers
from sklearn.linear_model import LinearRegression  # Import the linear regression prediction model
from sklearn.feature_selection import SelectKBest, mutual_info_regression  # Import tools for feature selection
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

    print("HANDLING MISSING DATA")  # Print header for the data cleaning section
    print("Artificially deleting some 'Hits' (H) data to demonstrate")  # Print description of the demonstration

    df.loc[25, 'H'] = np.nan  # Artifically set the value in column 'H' for row index 25 to missing (NaN)
    imputer = SimpleImputer(strategy='median')  # Create an imputer tool to fill missing values using the median
    df['H'] = imputer.fit_transform(df[['H']])  # Replace any missing values in 'H' with the calculated median
    print("Imputation completed 'Hits' (H)")  # Print confirmation that the missing data is filled

    print(df.head())  # Print the first 5 rows of the data to verify results
    print(df.tail())  # Print the last 5 rows of the data to verify results

    print("\nBasic Statistics:")  # Print header for the statistical summary
    print(df.describe())  # Print summary statistics (like mean, median, min, max) for numeric columns
    df["Team_ID"] = ["Team_" + str(np.random.randint(1,150)) for _ in range(len(df))]

    if TargetEncoder is not None:
        print("Applying Target Encoder")

        encoder = TargetEncoder()

        df["Team_ID_Encoder"] = encoder.fit_transform(df["Team_ID"], df["H"])

    else:
        print("Category Encoder not installed")   

    # feature Selection
    
    feature_to_test =["R","HR","SO","SB"]

    x_features = df[feature_to_test].fillna(0)
    y_target = df['W']
    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(x_features, y_target)
    winning_features = selector.get_support()
    best_features = x_features.columns[winning_features].tolist()

    print(best_features)
    
    
if __name__ == "__main__":  # Check if this script is being run directly
    main()  # Run the main function
    

# Handling Missing Values
print("Artifically creating missing data for the demonstration")    
