# Marine Vessel Propulsion Systems

A machine learning project that predicts Gas Turbine shaft torque in marine vessels based on operational parameters.

## Overview

This project analyzes data from marine vessel propulsion systems to predict performance metrics. It uses machine learning algorithms to understand the relationship between various ship operational parameters and the torque produced by gas turbines.

## What It Does

- Cleans and processes marine vessel operational data
- Visualizes relationships between different propulsion system parameters
- Trains various machine learning models (Random Forest, SVM, Gradient Boosting) to predict compressor and turbine degradation.
- **Interactive Digital Twin Dashboard**: A Gradio-based interface to monitor vessel health in real-time.
- **AI Assistant**: An Ollama-powered chatbot that provides technical insights and maintenance recommendations based on live sensor data.

## Models Used

- **Random Forest**: Ensemble learning approach
- **Gradient Boosting**: Sequential learning for improved accuracy
- **Support Vector Machine (SVM)**: Non-linear regression modeling

## How to Run

1. **Setup Environment**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib gradio ollama python-dotenv
   ```
2. **Setup Ollama**:
   Ensure [Ollama](https://ollama.com/) is installed and running, then pull a model:
   ```bash
   ollama pull llama3
   ```
3. **Launch Dashboard**:
   ```bash
   python src/main.py
   ```

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

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- gradio, ollama, python-dotenv

## Results

The project outputs performance metrics comparing how well each model predicts gas turbine torque, helping identify the most accurate prediction method.