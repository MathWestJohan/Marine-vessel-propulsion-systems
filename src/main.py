import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from CleaningData import load_and_clean_data, split_and_save_data
from Plots import run_all_plots

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src', 'models'))

from Random_forest import train_random_forest
from Gradientboosting import train_gradient_boosting
from SVM import train_svm
from ModelEvaluationPlots import run_model_comparison_plots
from DigitalTwin import PropulsionDigitalTwin
from dashboard import launch_dashboard

def select_best_model(results, metric="Test R2"):
    """
    Selects the best model based on the specified metric.

    Args:
        results (list of dict): List of model performance dictionaries.
        metric (str): The metric to use for selection (default is "Test R2").

    Returns:
        dict: The dictionary containing the best model's performance and object.
    """
    best = max(results, key=lambda r: r[metric])
    print(f"  Best model: {best['Model']} (Test R²={best[metric]:.6f})")
    return best


def main():
    # 1. Data setup
    df = load_and_clean_data()
    split_and_save_data(df)

    if os.path.basename(os.getcwd()) == 'src':
        train_path, test_path = '../Data/train.csv', '../Data/test.csv'
        image_dir = '../images'
    else:
        train_path, test_path = 'Data/train.csv', 'Data/test.csv'
        image_dir = 'images'

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 2. EDA plots
    run_all_plots(df, image_dir)

    # 3. Model training and comparison
    targets = ['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']
    all_results = []
    best_models = {}
    best_scalers = {}

    model_dir = 'saved_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for target in targets:
        target_key = "Compressor" if "Compressor" in target else "Turbine"
        model_path = os.path.join(model_dir, f"{target_key.lower()}_model.joblib")
        scaler_path = os.path.join(model_dir, f"{target_key.lower()}_scaler.joblib")

        if os.path.exists(model_path):
            print(f"\n--- Loading existing model for {target} ---")
            best_models[target_key] = joblib.load(model_path)
            best_scalers[target_key] = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        else:
            print(f"\n--- Training for {target} ---")
            results = [
                train_random_forest(train_path, test_path, target, image_dir),
                train_gradient_boosting(train_path, test_path, target, image_dir),
                train_svm(train_path, test_path, target, image_dir),
            ]
            all_results.extend(results)

            best = select_best_model(results, metric="Test R2")
            best_models[target_key] = best["model_object"]
            best_scalers[target_key] = best.get("scaler", None)

            # Save the best model and scaler
            joblib.dump(best_models[target_key], model_path)
            if best_scalers[target_key] is not None:
                joblib.dump(best_scalers[target_key], scaler_path)
            print(f"  Saved best {target_key} model to {model_path}")

            # Pass the already-trained models to the comparison plots
            # so they don't retrain from scratch
            trained_models_for_plots = {}
            for r in results:
                trained_models_for_plots[r["Model"].split(" (")[0]] = (
                    r["model_object"],
                    r.get("scaler", None),
                )
            run_model_comparison_plots(
                train_path, test_path, target, image_dir,
                trained_models=trained_models_for_plots,
            )

    # 4. Summary table, CSV export, and comparison plot
    if all_results:
        comparison_df = pd.DataFrame(all_results)

        # Drop non-metric columns for display
        display_cols = [c for c in comparison_df.columns if c not in ['model_object', 'scaler']]
        display_df = comparison_df[display_cols]

        print("\nFinal Model Comparison Table:")
        print(display_df.to_string(index=False))

        # Save to CSV for the report
        csv_path = os.path.join(image_dir, 'model_comparison.csv')
        display_df.to_csv(csv_path, index=False)
        print(f"\nSaved comparison table to: {csv_path}")

        # Save as a matplotlib table figure
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('off')
        table_data = display_df.round(6)
        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Model Performance Comparison', fontsize=14, pad=20)
        plt.tight_layout()
        table_path = os.path.join(image_dir, 'model_comparison_table.png')
        plt.savefig(table_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison table figure to: {table_path}")

        # R² bar chart
        comparison_df.set_index('Model')[['Train R2', 'Test R2']].plot(kind='bar', figsize=(12, 6))
        plt.title('Overall R² Score Comparison')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'overall_r2_comparison.png'), dpi=150)
        plt.close()

        # Cross-validation bar chart (if CV data available)
        if 'CV R2 Mean' in comparison_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = range(len(comparison_df))
            bars = ax.bar(x, comparison_df['CV R2 Mean'],
                          yerr=comparison_df['CV R2 Std'], capsize=5,
                          color=['tab:blue', 'tab:orange', 'tab:green'] * 2)
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.set_ylabel('R² Score')
            ax.set_title('5-Fold Cross-Validation R² (mean ± std)')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'cross_validation_r2.png'), dpi=150)
            plt.close()
            print(f"Saved cross-validation plot to: {os.path.join(image_dir, 'cross_validation_r2.png')}")
    else:
        print("\nAll models loaded from disk — skipping R² comparison chart.")

    # 5. Digital Twin & Dashboard
    print("\n--- Launching Digital Twin Dashboard ---")
    print(" Note: Dashboard analyses historical CSV data (not live telemetry)")
    dt_twin = PropulsionDigitalTwin(
        compressor_model=best_models["Compressor"],
        turbine_model=best_models["Turbine"],
        comp_scaler=best_scalers["Compressor"],
        turb_scaler=best_scalers["Turbine"],
    )
    launch_dashboard(dt_twin)


if __name__ == "__main__":
    main()