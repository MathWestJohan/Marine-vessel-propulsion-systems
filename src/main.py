import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from CleaningData import load_and_clean_data, split_and_save_data
from Plots import run_all_plots


# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src', 'models'))

# NOW import the models and evaluation plots
from Random_forest import train_random_forest
from Gradientboosting import train_gradient_boosting
from SVM import train_svm
from ModelEvaluationPlots import run_model_comparison_plots # Move import here

def main():
    """
    Main entry point for the marine vessel propulsion systems analysis and modeling pipeline.

    This function orchestrates the complete workflow including:
    - Loading and cleaning the dataset
    - Splitting data into training and test sets
    - Generating exploratory data analysis plots
    - Training multiple machine learning models (Random Forest, Gradient Boosting, SVM)
    - Comparing model performance metrics
    - Visualizing and saving performance comparison graphs

    The function automatically detects the working directory to set appropriate file paths
    for data and output images. It trains three regression models, collects their performance
    metrics (R2 scores on both train and test sets), and generates a comparative visualization.

    Outputs:
        - Train/test CSV files saved to Data/ directory
        - Analysis plots saved to images/ directory
        - Model performance comparison table printed to console
        - Model performance comparison graph saved as 'model_performance_comparison.png'

    Raises:
        FileNotFoundError: If required data files or directories cannot be found
        ImportError: If required model modules (Random_forest, Gradientboosting, SVM) are not available

    Note:
        This function assumes the presence of helper functions: load_and_clean_data(),
        split_and_save_data(), run_all_plots(), train_random_forest(),
        train_gradient_boosting(), and train_svm().
    """
    df = load_and_clean_data()
    split_and_save_data(df)

    if os.path.basename(os.getcwd()) == 'src':
        train_path, test_path = '../Data/train.csv', '../Data/test.csv'
        image_dir = '../images'
    else:
        train_path, test_path = 'Data/train.csv', 'Data/test.csv'
        image_dir = 'images'

    # EDA Plots
    run_all_plots(df, image_dir)

    targets = ['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']
    all_results = []

    for target in targets:
        print(f"\n--- Training for {target} ---")
        results = [
            train_random_forest(train_path, test_path, target, image_dir),
            train_gradient_boosting(train_path, test_path, target, image_dir),
            train_svm(train_path, test_path, target, image_dir)
        ]
        all_results.extend(results)

        # Detailed comparison plots for this specific target
        run_model_comparison_plots(train_path, test_path, target, image_dir)

    # Final summary table
    comparison_df = pd.DataFrame(all_results)
    print("\nFinal Model Comparison Table:\n", comparison_df)

    # Summary Plot for R2 Scores
    comparison_df.set_index('Model')[['Train R2', 'Test R2']].plot(kind='bar', figsize=(12, 6))
    plt.title('Overall R2 Score Comparison')
    plt.ylabel('R2 Score')
    plt.savefig(os.path.join(image_dir, 'overall_r2_comparison.png'))
    plt.show()


if __name__ == "__main__":
    main()