import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
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
    # 1. Existing Data Setup
    df = load_and_clean_data()
    split_and_save_data(df)

    if os.path.basename(os.getcwd()) == 'src':
        train_path, test_path = '../Data/train.csv', '../Data/test.csv'
        image_dir = '../images'
    else:
        train_path, test_path = 'Data/train.csv', 'Data/test.csv'
        image_dir = 'images'

    # 2. Existing EDA Plots
    run_all_plots(df, image_dir)

    # 3. Model Training and Comparison
    targets = ['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']
    all_results = []
    best_models = {}

    for target in targets:
        target_key = "Compressor" if "Compressor" in target else "Turbine"
        print(f"\n--- Training for {target} ---")
        results = [
            train_random_forest(train_path, test_path, target, image_dir),
            train_gradient_boosting(train_path, test_path, target, image_dir),
            train_svm(train_path, test_path, target, image_dir)
        ]
        all_results.extend(results)

        best = select_best_model(results, metric="Test R2")
        best_models[target_key] = best["model_object"]

        # Detailed comparison plots for this specific target
        run_model_comparison_plots(train_path, test_path, target, image_dir)

    # 4. Existing Final Summary Table & Plot
    comparison_df = pd.DataFrame(all_results)
    print("\nFinal Model Comparison Table:\n", comparison_df.drop(columns=['model_object'], errors='ignore'))

    comparison_df.set_index('Model')[['Train R2', 'Test R2']].plot(kind='bar', figsize=(12, 6))
    plt.title('Overall R2 Score Comparison')
    plt.ylabel('R2 Score')
    plt.savefig(os.path.join(image_dir, 'overall_r2_comparison.png'))
    plt.show()

    # 5. Digital Twin & Predictive Maintenance Dashboard
    print("\n--- Launching Digital Twin Dashboard ---")
    print(" Note: Dashboard analyses historical CSV data (not live telemetry)")
    dt_twin = PropulsionDigitalTwin(
        compressor_model=best_models["Compressor"],
        turbine_model=best_models["Turbine"]
    )
    launch_dashboard(dt_twin)


if __name__ == "__main__":
    main()