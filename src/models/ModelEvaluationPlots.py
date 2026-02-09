import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def run_model_comparison_plots(train_path, test_path, image_dir):
    """
    Genererer avanserte sammenligningsplot for modellene som kreves i Presentation 2.
    """
    # Last inn data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target = 'GT Compressor decay state coefficient'
    drop_cols = [target, 'GT Turbine decay state coefficient']

    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df[target]
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df[target]

    # Skalering for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modeller som skal plottes
    models = {
        "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test),
        "XGBoost": (XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42), X_train, X_test),
        "SVM": (SVR(kernel='rbf', C=1.0, epsilon=0.01), X_train_scaled, X_test_scaled)
    }

    predictions = {}
    maes = {}
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for name, (model, xtrain, xtest) in models.items():
        model.fit(xtrain, y_train)
        preds = model.predict(xtest)
        predictions[name] = preds
        maes[name] = mean_absolute_error(y_test, preds)

    # Plot 1: Actual vs Predicted
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    for i, (name, preds) in enumerate(predictions.items()):
        axes1[i].scatter(y_test, preds, alpha=0.5, color=colors[i])
        axes1[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes1[i].set_title(f'{name}: Faktisk vs Predikert')
        axes1[i].set_xlabel('Faktisk verdi')
        axes1[i].set_ylabel('Predikert verdi')
        axes1[i].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'actual_vs_predicted_comparison.png'))

    # Plot 2: Residual (Feil) Fordeling
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    for i, (name, preds) in enumerate(predictions.items()):
        residuals = y_test - preds
        axes2[i].hist(residuals, bins=30, color=colors[i], alpha=0.7, edgecolor='black')
        axes2[i].set_title(f'{name}: Feilfordeling (Residuals)')
        axes2[i].set_xlabel('Feil (Faktisk - Predikert)')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'residual_distribution_comparison.png'))

    # Plot 3: MAE Sammenligning
    plt.figure(figsize=(10, 6))
    plt.bar(maes.keys(), maes.values(), color=colors)
    plt.title('Gjennomsnittlig absolutt feil (MAE) - Lavere er bedre')
    plt.ylabel('MAE Verdi')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(image_dir, 'mae_comparison_plot.png'))

    print(f"\nNye evalueringsplotter er lagret i {image_dir}")