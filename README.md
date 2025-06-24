# Breast Cancer Diagnosis Classification

This project focuses on classifying breast cancer as either benign or malignant based on various cellular features. It involves a complete machine learning pipeline, from data collection and exploration to model building, training, evaluation, and interpretation. The goal is to develop an accurate predictive model that can assist in early diagnosis.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Exploration and Visualization](#data-exploration-and-visualization)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building and Training](#model-building-and-training)
6. [Model Evaluation](#model-evaluation)
7. [Interpretation of Results](#interpretation-of-results)
8. [Conclusion](#conclusion)

## Project Overview
The core objective of this project is to build and evaluate classification models to predict breast cancer diagnosis. The project covers:

- Initial data inspection and cleaning.
- Feature engineering and selection based on statistical analysis and importance.
- Training various classification algorithms (Logistic Regression, K-Nearest Neighbors, Decision Tree).
- Optimizing model hyperparameters using GridSearchCV.
- Comprehensive evaluation of models using metrics like accuracy, precision, recall, and F1-score.
- Visualizing model performance and interpreting feature importance for each model.

## Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset. It contains 569 instances with 32 features, including an ID, diagnosis (M = malignant, B = benign), and 30 real-valued features describing characteristics of the cell.

### Dimensions: 
- (569,33) before cleaning.
### Target Variable Distribution:
- Benign (B): 357 instances
- Malignant (M): 212 instances

## Data Exploration and Visualization
__Initial exploration involved:__
- Checking the dataset shape and displaying the first few rows.
- Reviewing data types and non-null counts using my_ds.info().
- Analyzing descriptive statistics using my_ds.describe().
- Visualizing the relationship between features and the diagnosis class using Box Plots and Violin Plots. This helped in identifying features that show clear separation between benign and malignant cases.

__Key Findings from Exploration:__
- Features ending with _se (standard error) often showed very close or identical medians between the two diagnostic classes (Benign and Malignant). These features were identified as less discriminative and were subsequently removed from the dataset to simplify the model and potentially improve performance.
- Feature importance was initially assessed using RandomForestClassifier on the filtered dataset. The most impactful features identified were: concavity_mean, concave_points_mean, radius_worst, area_worst, perimeter_worst, and concave_points_worst.

## Data Preprocessing
This phase prepared the data for model training:
- **Missing Values:** Checked for and confirmed no significant missing values in the relevant columns.
- **Irrelevant Columns:** The id and Unnamed: 32 columns were dropped as they do not contribute to the prediction.
- **Outlier Detection:** Outliers were identified using the Interquartile Range (IQR) method, providing insights into data distribution but not explicitly removed at this stage.
- **Normalization:** Data features were standardized using StandardScaler. This is crucial for models sensitive to feature scales, such as Logistic Regression and K-Nearest Neighbors. The fitted scaler was saved as fitted_scaler.pkl.
- **Data Splitting:** The dataset was split into training, validation, and test sets to ensure robust model evaluation and prevent overfitting:
*Original Dataset → 80% Training+Validation, 20% Test*
*Training+Validation → 75% Training, 25% Validation* (effectively 60% Training, 20% Validation, 20% Test of the original data)
Stratified splitting was used to maintain the class distribution (benign/malignant) across all sets.

## Model Building and Training
Three classification models were selected and optimized:
- Logistic Regression (SGDClassifier)
- K-Nearest Neighbors (KNN)
- Decision Tree

### Hyperparameter Optimization with GridSearchCV:
For each model, GridSearchCV was employed to systematically search for the best combination of hyperparameters that maximize accuracy on the training set using 5-fold cross-validation. This approach ensures that the models are fine-tuned for optimal performance.

### Optimized Models:
The best performing models and their optimal hyperparameters were identified and saved:
- best_logistic_regression_model.pkl
- best_knn_model.pkl
- best_decision_tree_model.pkl

## Model Evaluation
Models were evaluated on the unseen test set using a comprehensive set of metrics:
- **Accuracy:** Overall correctness of predictions.
- **Precision:** Proportion of true positives among all positive predictions.
- **Recall:** Proportion of true positives among all actual positives (sensitivity).
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Visual representation of correct and incorrect classifications.
- **Classification Report:** Detailed breakdown of precision, recall, f1-score, and support for each class.


### Conclusion from Evaluation:
Logistic Regression demonstrated the best performance in this study, achieving a remarkable 98.25% accuracy and perfect precision (1.00) on the test set. This indicates its strong ability to correctly identify malignant cases without generating false positives, which is critical in a medical diagnosis context. KNN and Decision Tree also performed well, but Logistic Regression showed a slight edge, particularly in precision.

## Interpretation of Results
Beyond overall performance, the project also explored which features were most influential for each model:
- Logistic Regression: Feature importance is derived from the absolute values of its coefficients.
- K-Nearest Neighbors: Permutation importance was calculated to understand feature influence.
- Decision Tree: Feature importance is inherently provided by the Gini impurity or entropy reduction.

While RandomForestClassifier initially identified a set of important features (concavity_mean, concave_points_mean, radius_worst, area_worst, perimeter_worst, concave_points_worst), the specific top features varied slightly across individual models due to their different learning mechanisms. However, features related to concave points, radius, area, and perimeter (especially _worst and _mean values) consistently ranked high across multiple models, confirming their strong discriminative power for breast cancer diagnosis.

## Conclusion
This project successfully developed and evaluated machine learning models for breast cancer diagnosis. Logistic Regression emerged as the most effective model, demonstrating high accuracy and precision, making it a promising candidate for assisting in medical diagnostic processes. The analysis also highlighted critical features that significantly contribute to the classification, offering valuable insights into the characteristics differentiating benign and malignant tumors. Further work could include exploring more advanced ensemble methods or deep learning techniques for potentially even higher predictive power.