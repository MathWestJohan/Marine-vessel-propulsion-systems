import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_gradient_boosting(train_path, test_path, image_dir):
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