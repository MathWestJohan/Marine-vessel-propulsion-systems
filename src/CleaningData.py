import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path='Data/data.csv'):
    """
    Load and clean the marine vessel propulsion dataset from a CSV file.

    This function performs the following operations:
    1. Locates and loads the CSV file from the specified path
    2. Sanitizes column names by:
        - Stripping whitespace
        - Replacing '[' with '('
        - Replacing ']' with ')'
        - Replacing '<' with 'less_than'
        (These changes ensure XGBoost compatibility)
    3. Identifies and exports duplicate rows to 'dropped_duplicates.csv'
    4. Removes duplicate rows (keeping first occurrence)
    5. Removes rows with missing values

    Parameters
    ----------
    file_path : str, optional
         Path to the CSV data file, by default 'Data/data.csv'
         The function will also check one directory up if the file is not found
         at the specified path.

    Returns
    -------
    pandas.DataFrame
         A cleaned DataFrame with sanitized column names, no duplicates,
         and no missing values.

    Raises
    ------
    FileNotFoundError
         If the data file cannot be found at the specified path or one
         directory up.

    Notes
    -----
    - Duplicate rows are saved to 'dropped_duplicates.csv' in the current
      working directory before being removed.
    - Column name sanitization is critical for XGBoost model compatibility.

    """
    if not os.path.exists(file_path) and os.path.exists(os.path.join('..', file_path)):
        file_path = os.path.join('..', file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the data file at {file_path}")

        # 1. Load data
    df = pd.read_csv(file_path)

    # 2. Sanitize headers for XGBoost compatibility
    df.columns = df.columns.str.strip().str.replace('[', '(', regex=False).str.replace(']', ')',
                                                                                       regex=False).str.replace('<',
                                                                                                                'less_than',
                                                                                                                regex=False)

    # 3. Identify and remove constant columns (Temperature/Pressure inputs that don't change)
    # nunique() <= 1 finds columns where every row has the same value
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # 4. Remove duplicate rows and missing values
    df = df.drop_duplicates().dropna()

    print("Data cleaning complete.")
    return df


def split_and_save_data(df, data_folder='Data'):
    """
    Splits the data into 30/70 sets, removes specific coefficients,
    and saves without indexes.
    """
    if not os.path.exists(data_folder) and os.path.exists(os.path.join('..', data_folder)):
        data_folder = os.path.join('..', data_folder)

    # 5. Drop the 'GT Turbine' coefficient as requested, but KEEP the 'GT Compressor' target
    # so the model scripts can still evaluate performance.
    cols_to_remove = ['GT Turbine decay state coefficient']
    existing_removals = [c for c in cols_to_remove if c in df.columns]
    df = df.drop(columns=existing_removals)

    # 6. Perform the split
    train_df, test_df = train_test_split(df, test_size=0.70, random_state=42)

    # 7. Define file paths
    train_path = os.path.join(data_folder, 'train.csv')
    test_path = os.path.join(data_folder, 'test.csv')

    # 8. Save to CSV - index=False ensures no index columns are created
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved training set to: {train_path}")
    print(f"Saved testing set to: {test_path}")

    return train_df, test_df