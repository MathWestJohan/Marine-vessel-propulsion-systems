import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def train_svm(train_path, test_path, target_col, image_dir=None):
    """
    Train a Support Vector Machine (SVM) regression model for predicting GT Compressor decay state coefficient.

    This function loads training and testing data from CSV files, preprocesses the features using
    StandardScaler, trains an SVR model with RBF kernel, and evaluates its performance.

    Args:
        train_path (str): Path to the training data CSV file.
        test_path (str): Path to the testing data CSV file.
        image_dir (str): Directory path for saving images (currently unused).

    Returns:
        dict: A dictionary containing model evaluation metrics with the following keys:
            - "Model" (str): Name of the model ("SVM")
            - "Train R2" (float): R² score on training data
            - "Test R2" (float): R² score on testing data
            - "Train MAE" (float): Mean Absolute Error on training data
            - "Test MAE" (float): Mean Absolute Error on testing data

    Note:
        - The function drops 'GT Compressor decay state coefficient' and 
          'GT Turbine decay state coefficient' columns before training.
        - Uses RBF kernel with C=1.0 and epsilon=0.01 hyperparameters.
        - Features are standardized using StandardScaler.
    """
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    drop_cols = ['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']

    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='rbf', C=10.0, epsilon=0.001)
    model.fit(X_train_scaled, y_train)

    target_name = "Compressor" if "Compressor" in target_col else "Turbine"

    return {
        "Model": f"SVM ({target_name})",
        "Train R2": r2_score(y_train, model.predict(X_train_scaled)),
        "Test R2": r2_score(y_test, model.predict(X_test_scaled)),
        "Train MAE": mean_absolute_error(y_train, model.predict(X_train_scaled)),
        "Test MAE": mean_absolute_error(y_test, model.predict(X_test_scaled))
    }