# Comparative-analysis-of-ML-algorithms

# Student Admission Prediction using Machine Learning Models

This project demonstrates a machine learning pipeline for predicting student admission based on various machine learning algorithms. The pipeline involves data preprocessing, model training, and evaluation using cross-validation to assess model performance.

## Project Overview

The aim of this project is to identify efficient Algorithm (which can perform best ) in this classification. It is to classify the student as Admit or Reject based on their Language Test Score, University Rating, Strength of their motivation letter (rated from 1-5), strength of their recommendation letter (rated from 1-5), CGPA and Research Experience. 
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

The models are evaluated based on their accuracy using k-fold cross-validation.

## Installation and Requirements

The following libraries are required to run the code:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

To install the libraries, you can use:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Code Overview

1. **Data Loading**: The dataset is loaded from a CSV file named `StudentData1.csv`.
2. **Data Preprocessing**:
   - The target variable (`Admission`) is separated from the feature variables.
   - Missing values in the dataset are identified (if any).
3. **Model Selection**:
   - The code uses four machine learning algorithms (Logistic Regression, KNN, Decision Tree, SVM) for prediction.
4. **Model Evaluation**:
   - Each model is evaluated using 10-fold cross-validation, measuring accuracy.
   - Results are printed for each model, displaying mean accuracy and standard deviation.
5. **Model Comparison**:
   - A boxplot is generated to compare the performance of different models.

## Usage

1. **Run the Code**:
   - Place `StudentData1.csv` in the same directory as the code.
   - Execute the code in a Python environment.

2. **Interpret Results**:
   - After running, the code will print the mean accuracy and standard deviation of each model.
   - The boxplot provides a visual comparison between the models.

## Example Output

```
LR: 0.85 (0.02)
KNN: 0.82 (0.03)
DTree: 0.79 (0.05)
SVM: 0.83 (0.04)
```

A boxplot will also display the comparative performance of each model.

## Future Improvements

- **Hyperparameter Tuning**: Use GridSearchCV to tune model hyperparameters.
- **Feature Engineering**: Create additional features or apply dimensionality reduction techniques.
- **Model Ensemble**: Combine models to improve predictive accuracy.

## License

This project is licensed under the MIT License.