
import pandas as pd

# Function to load data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


# Function to explore data
def explore_data(df: pd.DataFrame):
    """
    Perform basic data exploration, printing key details about the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to explore.
    """
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Corrected column names based on your dataset
    print("\nGender Distribution:")
    print(df['Gender'].value_counts())

    print("\nCourse Distribution:")
    print(df['Course'].value_counts())

    print("\nTuition Fees Up-to-Date Distribution:")
    print(df['Tuition fees up to date'].value_counts())

    print("\nDebtor Distribution:")
    print(df['Debtor'].value_counts())


# Function to create approval rate feature
def create_approval_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the approval rate feature based on 'Curricular units 1st sem (approved)'
    and 'Curricular units 1st sem (enrolled)' columns.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with the 'approval_rate' column.
    """
    df['approval_rate'] = (
        df['Curricular units 1st sem (approved)'] /
        df['Curricular units 1st sem (enrolled)']
    )
    return df


# Function to create performance score feature
def create_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the performance score feature based on 'Curricular units 1st sem (approved)'
    and 'Curricular units 1st sem (evaluations)' columns.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with the 'performance_score' column.
    """
    df['performance_score'] = (
        df['Curricular units 1st sem (approved)'] /
        df['Curricular units 1st sem (evaluations)']
    )
    return df


# Function to create all engineered features
def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features needed for the analysis.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Add engineered features
    df = create_approval_rate(df)
    df = create_performance_score(df)

    # Print to verify new columns
    print("\nData with new columns:")
    print(df[['Course', 'Gender', 'approval_rate', 'performance_score']].head())

    return df


# ------------------- Main Execution Block -------------------
if __name__ == "__main__":
    # Path to the dataset (update this path as needed)
    file_path = 'dataset.csv'

    # Step 1: Load the dataset
    df = load_data(file_path)

    # Step 2: Explore the dataset
    explore_data(df)

    # Step 3: Create engineered features
    df = create_engineered_features(df)
