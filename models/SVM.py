import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


def train_svm(train_path, test_path, image_dir):
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    target = 'GT Compressor decay state coefficient'
    drop_cols = [target, 'GT Turbine decay state coefficient']

    X_train, y_train = train_df.drop(columns=drop_cols), train_df[target]
    X_test, y_test = test_df.drop(columns=drop_cols), test_df[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    model.fit(X_train_scaled, y_train)

    return {
        "Model": "SVM",
        "Train R2": r2_score(y_train, model.predict(X_train_scaled)),
        "Test R2": r2_score(y_test, model.predict(X_test_scaled)),
        "Train MAE": mean_absolute_error(y_train, model.predict(X_train_scaled)),
        "Test MAE": mean_absolute_error(y_test, model.predict(X_test_scaled))
    }