import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_random_forest(train_path, test_path, image_dir):
    """
    Train a Random Forest Regressor model to predict GT Compressor decay state coefficient.

    This function loads training and testing data from CSV files, trains a Random Forest
    model on the training data, and evaluates its performance on both training and test sets.

    Args:
        train_path (str): File path to the training data CSV file.
        test_path (str): File path to the testing data CSV file.
        image_dir (str): Directory path for saving images (currently not used).

    Returns:
        dict: A dictionary containing model evaluation metrics with the following keys:
            - "Model" (str): Name of the model ("Random Forest")
            - "Train R2" (float): R-squared score on training data
            - "Test R2" (float): R-squared score on test data
            - "Train MAE" (float): Mean Absolute Error on training data
            - "Test MAE" (float): Mean Absolute Error on test data

    Note:
        - The function drops 'GT Compressor decay state coefficient' and 
          'GT Turbine decay state coefficient' from features.
        - The target variable is 'GT Compressor decay state coefficient'.
        - Uses 100 estimators with random_state=42 for reproducibility.
    """
    # Load Data
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    target = 'GT Compressor decay state coefficient'
    drop_cols = [target, 'GT Turbine decay state coefficient']

    X_train, y_train = train_df.drop(columns=drop_cols), train_df[target]
    X_test, y_test = test_df.drop(columns=drop_cols), test_df[target]

    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate both
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return {
        "Model": "Random Forest",
        "Train R2": r2_score(y_train, train_preds),
        "Test R2": r2_score(y_test, test_preds),
        "Train MAE": mean_absolute_error(y_train, train_preds),
        "Test MAE": mean_absolute_error(y_test, test_preds)
    }