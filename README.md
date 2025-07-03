# Parkinson's Disease Classification

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%2C%20Pandas%2C%20Numpy-orange?style=for-the-badge)

This project focuses on classifying patients with Parkinson's disease based on a range of biomedical voice measurements. The primary goal is to apply and evaluate various machine learning algorithms and cross-validation techniques to build a robust predictive model.

This project was developed as a practical exercise for the Master's in Artificial Intelligence program.

---

## 📋 Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
4.  [Results](#results)
5.  [How to Run](#how-to-run)
6.  [File Structure](#file-structure)

---

## 📌 Project Overview

Parkinson's disease is a neurodegenerative disorder that affects motor functions. Early diagnosis is crucial for treatment and management. This project leverages machine learning to distinguish between healthy individuals and those with Parkinson's by analyzing features extracted from voice recordings. The exercise specifically involves rigorous model validation and performance analysis.

## 📊 Dataset

The dataset used is the popular "Parkinson's Disease" dataset from the UCI Machine Learning Repository.

- **Initial Analysis:** A preliminary analysis was conducted to check for missing values and class imbalance. The project proceeds even with class imbalance to observe the impact on model performance, as per the exercise instructions.

---

## ⚙️ Methodology

The core of this project is a structured comparison of different modeling approaches and validation strategies.

### 1. Data Preprocessing
- **Data Cleaning:** Assessed and handled any missing values.
- **Feature Scaling:** Applied standard scaling to normalize the feature set, ensuring that all features contribute equally to model training.

### 2. Model Training & Cross-Validation Techniques
To ensure the robustness and generalizability of the models, three distinct cross-validation methods were implemented in separate notebooks:

- **Stratified Hold-Out:** The data was split into 80% for training and 20% for testing, preserving the class distribution in both sets.
- **K-Fold Cross-Validation:** Performed with **k=10**, providing a robust estimate of model performance by training and testing on 10 different subsets of the data.
- **Leave-One-Out (LOO):** An exhaustive cross-validation where each data point is used as a test set once, ideal for smaller datasets.

### 3. Machine Learning Algorithms Implemented
The following algorithms were trained and evaluated:

- **K-Nearest Neighbors (KNN):** With `k` values of 3, 5, and 27.
- **Multilayer Perceptron (MLP):**
    - A simple MLP with one hidden layer (testing between 10-100 neurons).
    - A deep MLP (`MLPDeepLearning`) with 3 hidden layers (testing between 10-100 neurons per layer).
- **Other Explored Algorithms:** My broader skill set also includes algorithms like **Support Vector Machines (SVM)** and **XGBoost**, which are excellent candidates for this type of classification task.

---

## 📈 Results

Model performance was evaluated using a comprehensive set of metrics. For each algorithm and validation technique, the following were calculated over 30 independent runs to ensure statistical significance:

- **Metrics:** Accuracy, Precision, Recall (Sensitivity), and Specificity.
- **Statistical Analysis:** The final results are presented in tables showing the **average, maximum, and minimum** values for each metric across the 30 runs.

*(Note: The detailed statistical tables can be found in the final section of each Jupyter Notebook.)*

---

## 🚀 How to Run

To replicate the analysis, please follow these steps:

1.  **Prerequisites:**
    - Python 3.8+
    - Jupyter Notebook or JupyterLab
    - Libraries: `scikit-learn`, `pandas`, `numpy`

2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hectorcaraucan/parkinsons-disease-classification.git](https://github.com/hectorcaraucan/parkinsons-disease-classification.git)
    cd parkinsons-disease-classification
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file listing the libraries).*

4.  **Run the notebooks:**
    Launch Jupyter and open one of the `.ipynb` files to see the implementation for each validation method.
    ```bash
    jupyter notebook
    ```

---

## 🗂️ File Structure

The repository is organized as follows:

├── parkinsons_hold_out.ipynb      # Notebook for the Stratified Hold-Out (80/20) method. 

├── parkinsons_k_fold_cv.ipynb     # Notebook for K-Fold Cross-Validation (k=10) method. 

├── parkinsons_leave_one_out.ipynb # Notebook for the Leave-One-Out method.
