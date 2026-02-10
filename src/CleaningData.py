import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path='Data/data.csv'):
    """
    Loads and cleans the dataset, keeping both decay coefficients.
    """
    # Adjust path if running from 'src' folder
    if not os.path.exists(file_path) and os.path.exists(os.path.join('..', file_path)):
        file_path = os.path.join('..', file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the data file at {file_path}")

    df = pd.read_csv(file_path)

    # Sanitize headers for XGBoost compatibility
    df.columns = df.columns.str.strip().str.replace('[', '(', regex=False).str.replace(']', ')',
                                                                                       regex=False).str.replace('<',
                                                                                                                'less_than',
                                                                                                                regex=False)

    # Remove constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)

    df = df.drop_duplicates().dropna()
    return df

def split_and_save_data(df, data_folder='Data'):
    """
    Splits the data into sets while keeping both decay coefficients for evaluation.
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

    print(f"Saved training set to: {train_path}")
    print(f"Saved testing set to: {test_path}")

    return train_df, test_df