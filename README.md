# Spaceship Titanic — Machine Learning Project

## Overview
This project predicts whether a passenger aboard the Spaceship Titanic was transported to another dimension following a mysterious incident. It uses machine learning to identify patterns in passenger details, cabin information, and onboard spending.

## Problem Statement
The objective is to develop and evaluate machine learning classification models that predict the `Transported` status of passengers based on the available passenger dataset.

## Objectives
- Perform Exploratory Data Analysis (EDA) to understand the dataset and identify patterns.
- Clean and preprocess data, including handling missing values.
- Engineer meaningful features from passenger IDs, cabin details, and spending information.
- Train and compare multiple classification models.
- Evaluate models using appropriate performance metrics.
- Develop an interactive Streamlit application for making predictions.

## Dataset
The project uses the Spaceship Titanic dataset, containing passenger information such as:
- Home planet and destination
- Age, VIP status, and CryoSleep
- Cabin details, including deck and side
- Spending across onboard services
- Target variable: `Transported`

The dataset is stored in the `data/` directory.

## Technologies Used
- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook, VS Code

## Project Workflow
1. Data loading and initial inspection
2. Exploratory Data Analysis (EDA)
3. Data cleaning and missing-value handling
4. Feature engineering using passenger IDs, cabin details, group size, and spending
5. Categorical encoding and feature scaling
6. Training and evaluating classification models
7. Comparing model performance
8. Developing an interactive Streamlit prediction application

## Machine Learning Models
The following classification algorithms were trained and evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree Classifier

## Model Evaluation
The models were evaluated using accuracy, precision, recall, and F1-score.

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 78.38% | 78.92% |
| K-Nearest Neighbors (K=9) | 78.95% | 78.32% |
| Gaussian Naive Bayes | 72.40% | 76.95% |
| Decision Tree | 74.87% | 74.99% |

K-Nearest Neighbors achieved the highest accuracy among the evaluated baseline models, at approximately **78.95%**. Logistic Regression achieved a slightly higher F1-score than KNN in this experiment. The results illustrate why multiple evaluation metrics are useful when comparing classification models.

## Streamlit Application
The project includes an interactive Streamlit application that allows users to enter passenger details and obtain a prediction of whether the passenger was transported.

When the application starts, it loads and preprocesses the required data, trains the selected machine learning model, and uses the trained model to generate predictions based on user inputs.

### Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/PragatiTripathi000/Spaceship-Titanic-ML.git
```

**2. Navigate to the project directory**
```bash
cd Spaceship-Titanic-ML
```

**3. Install the required dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the Streamlit application**
```bash
streamlit run app.py
```

Make sure the dataset and all required project files are present in their expected locations.

## Repository Structure
```text
Spaceship-Titanic-ML/
├── data/
│   └── train.csv
├── notebooks/
│   └── spaceship_titanic_analysis.ipynb
├── app.py
├── preprocessing.py
├── requirements.txt
└── README.md
```

*Note: The structure above shows the main project files. Additional files, such as `.gitignore`, may also be present.*

## Future Scope
- Hyperparameter tuning with cross-validation (GridSearchCV).
- Ensemble methods (Random Forest, XGBoost).

## Conclusion
This project demonstrates an end-to-end machine learning workflow, including exploratory data analysis, data preprocessing, feature engineering, classification model comparison, and development of an interactive prediction application. It provides practical experience in applying machine learning techniques to a real-world-style classification problem.
