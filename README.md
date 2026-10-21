# Spaceship Titanic — Machine Learning Project

## Overview
This project predicts whether a passenger aboard the Spaceship Titanic was transported to another dimension after a mysterious incident. It uses machine learning techniques to identify patterns in passenger information, cabin details, and onboard spending.

## Problem Statement
The objective is to build and evaluate classification models that predict the `Transported` status of passengers using the available passenger dataset.

## Objectives
- Perform Exploratory Data Analysis (EDA) to understand the dataset.
- Clean and preprocess data, including handling missing values.
- Engineer useful features from passenger IDs, cabin details, and spending information.
- Train and compare multiple classification models.
- Develop an interactive Streamlit application for predictions.

## Dataset
The project uses the Spaceship Titanic dataset, which contains passenger information such as:
- Home planet and destination
- Age, VIP status, and CryoSleep
- Cabin details
- Spending across onboard services
- Target variable: `Transported`

The dataset is available in the `data/` directory.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Project Workflow
1. Data loading and inspection
2. Exploratory Data Analysis
3. Data cleaning and missing-value handling
4. Feature engineering
5. Categorical encoding and feature scaling
6. Training and evaluation of classification models
7. Model comparison and selection
8. Streamlit application development

## Machine Learning Models
Four classification algorithms were evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree Classifier

## Results
The models were evaluated using accuracy, precision, recall, and F1-score.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 78.38% |
| K-Nearest Neighbors (KNN, K=9) | 78.95% |
| Gaussian Naive Bayes | 72.40% |
| Decision Tree | 74.87% |

K-Nearest Neighbors achieved the highest accuracy among these baseline models, at approximately 78.95%. Model selection should also consider precision, recall, F1-score, and the project's prediction requirements.

## Streamlit Application
The project includes a Streamlit application that provides an interactive interface for making predictions.

To run the application, install the required libraries and execute:

```bash
streamlit run app.py
```

Ensure the required model files, preprocessing functions, and dataset paths are correctly configured.

## Repository Structure
```text
Spaceship-Titanic-ML/
├── data/
├── notebooks/
│   └── spaceship_titanic_analysis.ipynb
├── app.py
├── preprocessing.py
└── README.md
```

Additional files, such as saved models, `requirements.txt`, and `.gitignore`, may be included as needed.

## Future Scope
- Experiment with additional feature engineering techniques.
- Improve model performance through systematic experimentation.
- Compare alternative encoding and missing-value handling strategies.
- Deploy the Streamlit application online.

## Conclusion
This project demonstrates the end-to-end machine learning workflow, from exploratory data analysis and preprocessing to model evaluation and an interactive prediction application.