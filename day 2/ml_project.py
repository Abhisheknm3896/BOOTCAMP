# pyrefly: ignore [missing-import]  # Ignore missing import warning from Pyrefly
import pandas as pd  # Import the pandas library to work with tabular data (DataFrames)

# pyrefly: ignore [missing-import]
import numpy as np  # Import the NumPy library for numerical operations and arrays

from sklearn.impute import SimpleImputer  # Import SimpleImputer to replace missing values
from sklearn.model_selection import train_test_split  # Import train_test_split to divide the dataset into training and testing sets
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import LabelEncoder for categorical encoding and StandardScaler for feature scaling
from sklearn.linear_model import LinearRegression  # Import Linear Regression model for prediction
from sklearn.feature_selection import SelectKBest, mutual_info_regression  # Import feature selection tools to choose the best features
import os  # Import the os module to work with file paths

# Try to import TargetEncoder for target encoding
try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder  # Import TargetEncoder if the library is installed
except ImportError:
    TargetEncoder = None  # Set TargetEncoder to None if the library is not installed
    print("warning : category_encoders not installed.Target Encoding will not work")  # Display a warning message

# Define the main function
def main():

    print("Loading datasets")  # Display a message that dataset loading has started

    file_path = "train.csv"  # Store the dataset file name

    if not os.path.exists(file_path):  # Check whether the dataset file exists
        print(f"Error: train.csv not found: '{file_path}'")  # Display an error if the file is missing
        return  # Stop program execution

    df = pd.read_csv(file_path)  # Read the CSV file into a Pandas DataFrame

    print(f"Dataset Loaded Successfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n")  # Display the number of rows and columns

    print("HANDLING MISSING DATA")  # Display the heading for missing value handling

    print("Artificially deleting some 'Hits' (H) data to demonstrate")  # Explain that a missing value is created for demonstration

    df.loc[25, 'H'] = np.nan  # Replace the value in row 25, column H with NaN (missing value)

    imputer = SimpleImputer(strategy='median')  # Create an imputer object that fills missing values using the median

    df['H'] = imputer.fit_transform(df[['H']])  # Calculate the median and replace all missing values in column H

    print("Imputation completed 'Hits' (H)")  # Display confirmation after filling missing values

    print(df.head())  # Display the first five rows of the dataset

    print(df.tail())  # Display the last five rows of the dataset

    print("\nBasic Statistics:")  # Display the heading for descriptive statistics

    print(df.describe())  # Display summary statistics such as mean, median, minimum, and maximum

    df["Team_ID"] = ["Team_" + str(np.random.randint(1,150)) for _ in range(len(df))]  
    # Create a new categorical column called Team_ID with random team names like Team_10, Team_75, etc.

    if TargetEncoder is not None:  # Check whether TargetEncoder is available

        print("Applying Target Encoder")  # Display a message before applying target encoding

        encoder = TargetEncoder()  # Create a TargetEncoder object

        df["Team_ID_Encoder"] = encoder.fit_transform(df["Team_ID"], df["H"])
        # Convert Team_ID into numerical values using the target column H

    else:
        print("Category Encoder not installed")  # Display a message if TargetEncoder is unavailable

    # -------------------- Feature Selection --------------------

    feature_to_test = ["R", "HR", "SO", "SB"]  
    # Select four features that will be tested for importance

    x_features = df[feature_to_test].fillna(0)  
    # Store the selected features and replace any missing values with 0

    y_target = df["W"]  
    # Select column W as the target variable

    selector = SelectKBest(score_func=mutual_info_regression, k=2)  
    # Create a feature selector that chooses the best 2 features using Mutual Information Regression

    selector.fit(x_features, y_target)  
    # Calculate the importance score for every selected feature

    winning_features = selector.get_support()  
    # Return a Boolean array indicating which features are selected

    best_features = x_features.columns[winning_features].tolist()  
    # Convert the selected Boolean values into feature names

    print(best_features)  
    # Display the names of the selected best features

    # -------------------- Splitting the Dataset --------------------

    X = df[best_features]  
    # Store the selected best features as input variables

    Y = df["W"]  
    # Store column W as the output (target) variable

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=40
    )
    # Split the dataset into:
    # 80% Training Data
    # 20% Testing Data
    # random_state=40 ensures the split is the same every time

    print(f"Training Data Shape: {X_train.shape}")  
    # Display the size of the training dataset

    print(f"Testing Data size: {X_test.shape}\n")  
    # Display the size of the testing dataset

    #training model

    model=LinearRegression()

    model.fit(X_train,Y_train)
    
    prediction = model.predict(X_test)
    print("Prediction")
    print(prediction)

    # computing model predection on the train data

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
        


# Check whether the file is executed directly
if __name__ == "__main__":
    main()  # Execute the main function
print("\n")
print("Artificially creating missing data for the demonstration")
# Display a final message after program execution