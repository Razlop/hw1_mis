# hw1

# House Price Classification Project

This project is a binary classification task where we try to predict whether a house's price is greater than $1 million (denoted by 1) or not (denoted by 0), based on various features of the house.

## Getting Started

The dataset used in this project is `kc_house_data_classification.csv` and `kc_house_data_regression.csv`.

## Tasks and Approach

### Task 1 - Load Data
We loaded our dataset using pandas' `read_csv` function.

```python
housing_df = pd.read_csv('kc_house_data_classification.csv')
```

### Task 2 - EDA

We performed Exploratory Data Analysis (EDA) on our dataset, primarily using the `describe` and `info` methods from pandas DataFrames to get a summary of the data.

```python
housing_df.describe()
housing_df.info()
```

### Task 3 - Preprocessing

In this step, we preprocessed the data, which involved splitting it into feature and target variables (X and y, respectively). We then further divided these into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X = housing_df.iloc[:, 0:18]
y = housing_df.iloc[:, 18]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)
```

### Task 4 - Logistic Regression Models

In this task, we built several logistic regression models using different regularization techniques and hyperparameters, such as Ridge and Lasso regression with varying C values. The models were built using sklearn's `LogisticRegression` and `LogisticRegressionCV` classes, within a pipeline that included preprocessing steps.

For each model, we computed accuracy scores for both the training and test data, and created confusion matrices.

### Task 5 - Simple Decision Tree

In this task, we fit a decision tree model to predict `price_gt_1M` using sklearn's `DecisionTreeClassifier`. We computed the accuracy score, created a confusion matrix for both train and test datasets, and discussed the performance relative to our logistic regression models.


### Task 6 - Error Exploration

In this task, we examined the errors made by Model 2 (the Lasso model with C=1.0) to gain more insights into its performance. We generated predictions from Model 2, identified where they differed from the actual target values, and created a histogram of the actual house prices for the test samples that were misclassified.
