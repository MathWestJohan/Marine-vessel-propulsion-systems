# Marine Vessel Propulsion Systems

A machine learning project that predicts Gas Turbine shaft torque in marine vessels based on operational parameters.

## Overview

This project analyzes data from marine vessel propulsion systems to predict performance metrics. It uses machine learning algorithms to understand the relationship between various ship operational parameters and the torque produced by gas turbines.

## What It Does

- Cleans and processes marine vessel operational data
- Visualizes relationships between different propulsion system parameters
- Trains three different machine learning models to predict gas turbine torque
- Compares model performance to find the best predictor

## Models Used

- **Random Forest**: Ensemble learning approach
- **Gradient Boosting**: Sequential learning for improved accuracy
- **Support Vector Machine (SVM)**: Non-linear regression modeling

## How to Run

```bash
cd src
python main.py
```

This will process the data, train all models, and generate performance comparisons.

## Project Structure

```
├── src/
│   ├── main.py                 # Main pipeline
│   ├── CleaningData.py         # Data preprocessing
│   ├── Plots.py                # Visualizations
│   └── models/                 # ML models
├── data/                       # Datasets
└── images/                     # Generated plots
```

## Requirements

- Python 3.7+
- pandas, numpy, matplotlib, seaborn, scikit-learn

## Results

The project outputs performance metrics comparing how well each model predicts gas turbine torque, helping identify the most accurate prediction method.