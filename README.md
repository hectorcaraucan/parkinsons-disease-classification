# Parkinson's Disease Classification

This project implements and evaluates various machine learning models to classify Parkinson's disease based on a public dataset. The primary challenge addressed is the significant class imbalance within the data. A comprehensive machine learning pipeline was developed, including data preprocessing, model training, and robust evaluation using multiple validation techniques.

## Project Overview

The main objective was to develop an accurate and reliable predictive model for diagnosing Parkinson's disease using a dataset with a notable class imbalance. The project involved a complete machine learning pipeline, from data analysis to model deployment.

### Dataset

The project utilizes a public dataset containing various biomedical voice measurements. The dataset consists of 195 entries with 24 features. A key characteristic of this dataset is the class imbalance, with **75.38%** of the records corresponding to individuals with Parkinson's and **24.62%** to healthy individuals.

### Features

The features in the dataset include a range of voice metrics such as:
* **MDVP:Fo(Hz)**: Average vocal fundamental frequency.
* **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency.
* **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency.
* **Jitter** and **Shimmer**: Measures of frequency and amplitude variation.
* **HNR**: Harmonics-to-noise ratio.
* **RPDE**: Recurrence period density entropy.
* **DFA**: Detrended fluctuation analysis.
* **PPE**: Pitch period entropy.

## Methods

A thorough data analysis was conducted to identify and quantify the class imbalance and check for missing values. The dataset was segmented using three distinct validation techniques to ensure robust evaluation:

1.  **Stratified Hold-Out**: 80% for training and 20% for testing.
2.  **10-Fold Cross-Validation** 
3.  **Leave-One-Out Cross-Validation** 

A suite of algorithms was trained and evaluated, including:
* Multilayer Perceptron (MLP) with 10 and 100 neurons.
* A deep MLP with three hidden layers.
* K-Nearest Neighbors (KNN) with K values of 3, 5, and 27.

For statistical significance, 30 independent runs were performed for each experiment, systematically calculating the mean, minimum, and maximum for key performance metrics.

## Results

The **KNN model (k=3) using stratified hold-out validation** emerged as the top performer. It achieved the following average results:

* **Accuracy**: 85.1% 
* **Precision**: 89.0% 
* **Recall**: 92.0% 

These results demonstrated a strong ability to correctly identify patients with the disease while maintaining a low rate of false positives, proving the model's effectiveness despite the imbalanced dataset.

| Validation Method | Algorithm | Accuracy | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Stratified Hold-Out** | **KNN (k=3)** | **85.1%**  | **89.0%**  | **92.0%**  |
| Stratified Hold-Out | KNN (k=5) | 85.3%  | 87.7%  | 94.1%  |
| K-Fold Cross-Validation | KNN (k=3) | 85.0%  | 88.4%  | 92.4%  |
| Leave-One-Out | KNN (k=3) | 85.1%  | 88.8%  | 91.8%  |

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hectorcaraucan/parkinsons-disease-classification.git](https://github.com/hectorcaraucan/parkinsons-disease-classification.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd parkinsons-disease-classification
    ```
3.  **The project contains the following key files:**
    * `parkinsons_data.csv`: The dataset used for training and evaluation.
    * `hold_out_estratificado.ipynb`: Jupyter Notebook for the stratified hold-out validation.
    * `k_fold_cross_validation.ipynb`: Jupyter Notebook for the 10-fold cross-validation.
    * `leave_one_out.ipynb`: Jupyter Notebook for the leave-one-out cross-validation.
    * `requirements.txt`: A file listing the Python libraries needed to run the notebooks.

4.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Jupyter Notebooks:**
    Launch Jupyter Notebook and open the `.ipynb` files to see the implementation and results.
