import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from CleaningData import load_and_clean_data, split_and_save_data
from Plots import run_all_plots
from ModelEvaluationPlots import run_model_comparison_plots

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src', 'models'))

from Random_forest import train_random_forest
from Gradientboosting import train_gradient_boosting
from SVM import train_svm


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

    # Run Analysis Plots
    run_all_plots(df, image_dir)

    # Run ML Models and collect metrics
    print("\n--- Comparing Train vs Test Performance ---")
    results = [
        train_random_forest(train_path, test_path, image_dir),
        train_gradient_boosting(train_path, test_path, image_dir),
        train_svm(train_path, test_path, image_dir)
    ]

    # Create Comparison Table
    comparison_df = pd.DataFrame(results)
    print("\n", comparison_df)

    # Generate Comparison Graph
    comparison_df.set_index('Model')[['Train R2', 'Test R2']].plot(kind='bar', figsize=(10, 6))
    plt.title('R2 Score Comparison: Training vs. Testing')
    plt.ylabel('R2 Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path = os.path.join(image_dir, 'model_performance_comparison.png')
    plt.savefig(save_path)
    print(f"\nComparison graph saved to {save_path}")
    plt.show()

    save_path = os.path.join(image_dir, 'model_performance_comparison.png')
    plt.savefig(save_path)
    print(f"\nComparison graph saved to {save_path}")
    plt.show()  # Dette viser R2-grafen først

    # Generer de dypere analysene for Presentation 2
    run_model_comparison_plots(train_path, test_path, image_dir)

    # Vis de nye plottene (Actual vs Predicted og Residuals)
    plt.show()


if __name__ == "__main__":
    main()