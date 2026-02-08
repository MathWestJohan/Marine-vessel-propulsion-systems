import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path='Data/data.csv'):
    """
    Imports the data, strips column names, and removes duplicates/missing values.
    Also sanitizes column names for compatibility with XGBoost.
    """
    if not os.path.exists(file_path) and os.path.exists(os.path.join('..', file_path)):
        file_path = os.path.join('..', file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the data file at {file_path}")

    # 1. Load data
    df = pd.read_csv(file_path)

    # 2. Sanitize headers (Crucial for XGBoost)
    # Replaces [, ], and < with characters allowed by XGBoost
    df.columns = df.columns.str.strip().str.replace('[', '(', regex=False).str.replace(']', ')', regex=False).str.replace('<', 'less_than', regex=False)

    # 3. Check for duplicates
    duplicate_rows = df[df.duplicated(keep='first')]
    if not duplicate_rows.empty:
        duplicate_rows.to_csv('dropped_duplicates.csv', index=False)
        print(f"Captured {len(duplicate_rows)} duplicate rows to 'dropped_duplicates.csv'.")

    # 4. Clean the data
    df = df.drop_duplicates()
    df = df.dropna()

    print("Data cleaning complete.")
    return df

def split_and_save_data(df, data_folder='Data'):
    """
    Splits the dataframe into 30% training and 70% testing sets and saves them as CSVs.
    """
    if not os.path.exists(data_folder) and os.path.exists(os.path.join('..', data_folder)):
        data_folder = os.path.join('..', data_folder)

    # Perform the split
    train_df, test_df = train_test_split(df, test_size=0.70, random_state=42)

    # Define file paths
    train_path = os.path.join(data_folder, 'train.csv')
    test_path = os.path.join(data_folder, 'test.csv')

    # Save to CSV
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved training set (30%) to: {train_path}")
    print(f"Saved testing set (70%) to: {test_path}")

    return train_df, test_df