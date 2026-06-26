from openpyxl.styles import named_styles  # Import Excel styling tool (not used in this script)
import pandas as pd  # Import Pandas library for data manipulation and analysis
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt  # Import Matplotlib for creating visualizations/plots
import seaborn as sns  # Import Seaborn for making statistical graphics/heatmaps
import os  # Import os module to work with files and directories

print("Understanding dataset")  # Print section title to indicate dataset exploration starts

file_name = 'sales_data_1.csv'  # Set the name of the CSV file to be loaded
if not os.path.exists(file_name):  # Check if the CSV file exists in the current folder
    print(f"error: {file_name} is not found")  # Print an error message if the file doesn't exist
    exit()  # Stop the program execution if the file is missing

df = pd.read_csv(file_name)  # Load the CSV file data into a Pandas DataFrame
print("Successfully loaded ")  # Print confirmation that the file was loaded successfully
print(f"Shape of the dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")  # Print total rows and columns in the dataset

print("\n--- Head ---")  # Print label for the first few rows
print(df.head())  # Show the first 5 rows of the data to preview it
print("\n--- Tail ---")  # Print label for the last few rows
print(df.tail())  # Show the last 5 rows of the data to preview it
print("\n--- Info ---")  # Print label for metadata/data types
print(df.info())  # Show structural details, column types, and non-null counts
print("\n--- Describe ---")  # Print label for statistics
print(df.describe())  # Show basic statistics (mean, min, max, etc.) for numeric columns

print("\nHandling Missing Values")  # Print section title for handling missing data
print("Missing values per column:")  # Print header for missing value counts
print(df.isnull().sum())  # Count and display how many empty/null cells are in each column

# Handle missing Age (with median Age)
median_age = df['Age'].median()  # Calculate the middle value (median) of the Age column
df['Age'] = df['Age'].fillna(median_age)  # Fill any empty cells in the Age column with the median value
print(f"Imputed missing Age with median: {median_age}")  # Print the age value used for imputation

# Handle missing Spending (with mean Spending)
mean_spending = df['Spending'].mean()  # Calculate the average (mean) value of the Spending column
df['Spending'] = df['Spending'].fillna(mean_spending)  # Fill empty cells in the Spending column with the average value
print(f"Imputed missing Spending with mean: {mean_spending:.2f}")  # Print the spending value used for imputation

# Handle missing Visits_Per_Month (with median Visits_Per_Month)
if df['Visits_Per_Month'].isnull().any():  # Check if there are any missing values in Visits_Per_Month
    median_visits = df['Visits_Per_Month'].median()  # Calculate the middle value (median) of Visits_Per_Month
    df['Visits_Per_Month'] = df['Visits_Per_Month'].fillna(median_visits)  # Fill empty visits cells with the median value
    print(f"Imputed missing Visits_Per_Month with median: {median_visits}")  # Print the visits value used for imputation

# Plotting Distribution of Spending
plt.figure(figsize=(10, 6))  # Set up a new plotting window/figure with size 10x6 inches
df['Spending'].hist(bins=20, edgecolor='black')  # Draw a histogram of the Spending column with 20 bars and black borders
plt.title('Distribution of Spending (sales_data_1.csv)')  # Add a title to the histogram plot
plt.xlabel('Spending')  # Label the horizontal axis as 'Spending'
plt.ylabel('Frequency')  # Label the vertical axis as 'Frequency'
plt.show()  # Display the histogram chart on the screen

# Compute Correlation
correlation = df.corr(numeric_only=True)  # Compute how numeric columns relate to each other (-1 to 1)
print("\nCorrelation Matrix:")  # Print label for correlation values
print(correlation)  # Display the calculated correlation matrix in text format

# Plotting Correlation Heatmap
print("Plotting correlation Heatmap")  # Print status message indicating map is plotting
plt.figure(figsize=(10, 8))  # Set up a new plotting window/figure with size 10x8 inches
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')  # Draw a colored heatmap showing correlation values
plt.title('Correlation Heatmap (sales_data_1.csv)')  # Add a title to the heatmap plot
plt.show()  # Display the correlation heatmap chart on the screen

# Find Outliers in Age (where Age > 100)
print("\nFinding the Outliers in Age (Age > 100)")  # Print section title for finding age outliers
outliers = df[df['Age'] > 100]  # Filter the data to find any rows where Age is greater than 100
print("Found Outlier(s):")  # Print label for displaying the results
print(outliers)  # Display the rows containing age outliers
