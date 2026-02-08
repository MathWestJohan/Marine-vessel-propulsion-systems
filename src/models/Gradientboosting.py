import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_gradient_boosting(train_path, test_path, image_dir):
    """
    Train an XGBoost gradient boosting model for predicting GT Compressor decay state coefficient.

    This function loads training and testing data from CSV files, trains an XGBoost regression model,
    and evaluates its performance using R² score and Mean Absolute Error (MAE) metrics.

    Args:
        train_path (str): File path to the training dataset CSV file.
        test_path (str): File path to the testing dataset CSV file.
        image_dir (str): Directory path for saving images (currently unused).

    Returns:
        dict: A dictionary containing model performance metrics with the following keys:
            - "Model" (str): Name of the model ("XGBoost")
            - "Train R2" (float): R² score on training data
            - "Test R2" (float): R² score on testing data
            - "Train MAE" (float): Mean Absolute Error on training data
            - "Test MAE" (float): Mean Absolute Error on testing data

    Notes:
        - The function predicts 'GT Compressor decay state coefficient' as the target variable.
        - 'GT Turbine decay state coefficient' is dropped from features along with the target.
        - Model uses 100 estimators with a learning rate of 0.1 and random_state=42 for reproducibility.
    """
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    target = 'GT Compressor decay state coefficient'
    drop_cols = [target, 'GT Turbine decay state coefficient']

    X_train, y_train = train_df.drop(columns=drop_cols), train_df[target]
    X_test, y_test = test_df.drop(columns=drop_cols), test_df[target]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    return {
        "Model": "XGBoost",
        "Train R2": r2_score(y_train, model.predict(X_train)),
        "Test R2": r2_score(y_test, model.predict(X_test)),
        "Train MAE": mean_absolute_error(y_train, model.predict(X_train)),
        "Test MAE": mean_absolute_error(y_test, model.predict(X_test))
    }