# Feature Engineering

### Project Overview
In this project, you will analyze an NBA player performance dataset using Python and pandas to engineer features for a machine learning model. You will learn how to define target variables, remove noise, analyze correlations, create composite metrics, and clean data to build a robust predictive dataset. This project strengthens your ability to transform raw sports statistics into model-ready inputs for longevity forecasting.

## Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

## Loading the Dataset
Load the dataset and clearly define the `target_5yrs` column as the dependent variable

#### Load the dataset
data = pd.read_csv('nba_players.csv')

## Add a visualization (e.g., sns.countplot) to check the distribution of 'target_5yrs'.
plt.figure(figsize=(6, 4))
sns.countplot(x='target_5yrs', data=data, hue='target_5yrs', palette='viridis', legend=False)
plt.title('Distribution of Target_5yrs (Career Longevity)')
plt.xlabel('Played 5+ years in NBA (0=No, 1=Yes)')
plt.ylabel('Number of Players')
plt.show()

<img width="540" height="393" alt="feature_engr_dist" src="https://github.com/user-attachments/assets/ef590bd7-eaf9-4b06-9664-b8a5d9ea4d23" />

## Check for class balance/distribution of the target variable.
## Checking Class Balance of Target Variable
We will visualize the distribution of the `target_5yrs` column to understand the balance between the two classes (players who played 5+ years in NBA vs. those who did not). This is important for evaluating model performance, especially if the classes are imbalanced.
plt.figure(figsize=(7, 5))
sns.countplot(x='target_5yrs', data=data, hue='target_5yrs', palette='coolwarm', legend=False)
plt.title('Distribution of Target_5yrs (Career Longevity)')
plt.xlabel('Played 5+ years in NBA (0=No, 1=Yes)')
plt.ylabel('Number of Players')
plt.show()

#### Also print the value counts for a numerical summary
print("\nValue counts for 'target_5yrs':")
print(data['target_5yrs'].value_counts())

<img width="618" height="470" alt="feature_engr_dist_career" src="https://github.com/user-attachments/assets/f4c0e09e-a9a0-4651-aae0-b2eaaf943eb0" />

## Drop non-predictive columns (e.g., player names, IDs) that add noise or risk data leakage
X = data.drop(columns=["Unnamed: 0", "name", "target_5yrs"])
y = data["target_5yrs"]
X.head(5)

#### Splitting into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Perform correlation analysis to identify highly correlated features and reduce redundancy
## Correlation Analysis
Performing correlation analysis on the features to identify highly correlated variables that might introduce redundancy or multicollinearity into the model.

#### Calculate the correlation matrix
corr_matrix = X_train.corr()

#### Plotting the heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features in X_train')
plt.show()

<img width="1121" height="990" alt="feature_engr_corr_matrix" src="https://github.com/user-attachments/assets/2c53daed-b8a0-4041-94d8-dd52b54a89cb" />

### Identifying Highly Correlated Features
Based on the correlation matrix, we can identify pairs of features that are highly correlated. High correlation (e.g., absolute value > 0.8 or 0.9) can indicate redundancy and lead to multicollinearity issues in some models. We might consider dropping one of the highly correlated features.

correlated_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.9: # You can adjust this threshold
            colname = corr_matrix.columns[i]
            correlated_features.add(colname)

print("Highly correlated features to consider dropping:")
print(list(correlated_features))

## Dropping Highly Correlated Features
To mitigate multicollinearity and improve model robustness, we will drop the identified highly correlated features from both the training and testing datasets.
#### Drop the highly correlated features from X_train and X_test
X_train_cleaned = X_train.drop(columns=list(correlated_features))
X_test_cleaned = X_test.drop(columns=list(correlated_features))

print("Original X_train shape:", X_train.shape)
print("Cleaned X_train shape:", X_train_cleaned.shape)
print("Original X_test shape:", X_test.shape)
print("Cleaned X_test shape:", X_test_cleaned.shape)

X_train = X_train_cleaned
X_test = X_test_cleaned

## Feature Scaling
Implementing feature scaling using `sklearn.preprocessing.StandardScaler`. Scaling is performed after splitting the data to ensure that the scaling parameters are learned only from the training data, preventing data leakage from the test set.
from sklearn.preprocessing import StandardScaler

#### Initialize the StandardScaler
scaler = StandardScaler()

#### Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#### Convert the scaled arrays back to DataFrames, preserving column names
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("X_train after scaling:")
display(X_train.head())
print("\nX_test after scaling:")
display(X_test.head())

### Engineer at least one new composite feature (e.g., Points Per Minute, Efficiency Rating) by combining existing metrics
### Engineering New Composite Features
We will create two new composite features to provide more insights into player performance:

1.  **Assist/Turnover Ratio (ATR)**: `ast / tov` - Measures a player's ability to create scoring opportunities for teammates relative to their ball-handling mistakes. A higher ratio is generally better.
2.  **Total Rebounds Per Minute (RPM)**: `(oreb + dreb) / min` - Indicates a player's rebounding efficiency relative to their playing time.

import numpy as np

#### Create 'Assist/Turnover Ratio' for X_train and X_test
X_train['atr'] = np.where(X_train['tov'] == 0, X_train['ast'], X_train['ast'] / X_train['tov'])
X_test['atr'] = np.where(X_test['tov'] == 0, X_test['ast'], X_test['ast'] / X_test['tov'])

#### Create 'Total Rebounds Per Minute' for X_train and X_test
X_train['rpm'] = np.where(X_train['min'] == 0, 0, (X_train['oreb'] + X_train['dreb']) / X_train['min'])
X_test['rpm'] = np.where(X_test['min'] == 0, 0, (X_test['oreb'] + X_test['dreb']) / X_test['min'])

print("X_train with new features:")
display(X_train.head())
print("\nX_test with new features:")
display(X_test.head())

### Handling Null Values in Performance Columns
Before proceeding with model re-evaluation, we need to ensure there are no null values in our feature sets (X_train and X_test), which might cause issues during model training. We will check for missing values and impute them if necessary.

#### Check for null values in X_train
print("Null values in X_train before cleaning:")
print(X_train.isnull().sum())

#### Check for null values in X_test
print("\nNull values in X_test before cleaning:")
print(X_test.isnull().sum())

#### Impute missing values with the mean of the column
#### This is a simple strategy; more sophisticated methods can be used if needed.
for col in X_train.columns:
    if X_train[col].isnull().any():
        X_train[col] = X_train[col].fillna(X_train[col].mean())
    if X_test[col].isnull().any():
        # Use the mean from X_train to avoid data leakage
        X_test[col] = X_test[col].fillna(X_train[col].mean())

print("\nNull values in X_train after cleaning:")
print(X_train.isnull().sum())

print("\nNull values in X_test after cleaning:")
print(X_test.isnull().sum())


## Documentation of Feature Selection and Engineering Choices

Throughout the data preparation phase, we made several deliberate choices regarding feature selection and engineering to enhance our model's performance and interpretability:

### 1. Initial Feature Selection (Dropping Non-Predictive Columns)
-   **Action**: We dropped the `Unnamed: 0`, `name`, and `target_5yrs` columns from the dataset.
-   **Rationale**:
    -   `Unnamed: 0`: This column was an artifact of the CSV import (likely an old index) and carried no predictive information.
    -   `name`: Player names are unique identifiers and do not contribute to predicting career longevity; they could introduce noise or data leakage if not handled carefully.
    -   `target_5yrs`: This is our target variable (`y`), so it must be separated from the feature set (`X`) to prevent data leakage during training.

### 2. Handling Multicollinearity (Dropping Highly Correlated Features)
-   **Action**: After performing a correlation analysis and visualizing the correlation matrix, we identified and dropped the following highly correlated features: `['pts', 'fta', 'fga', 'fgm', '3pa', 'reb']`.
-   **Rationale**: High correlation between features (multicollinearity) can negatively impact some regression models by:
    -   Making coefficient estimates unstable and difficult to interpret.
    -   Increasing the variance of the coefficients, leading to less reliable statistical inferences.
    -   Introducing redundancy, as multiple features convey similar information. By removing these, we aim for a more parsimonious and robust model.

### 3. Feature Engineering (Creating Composite Metrics)
-   **Action**: We created two new composite features:
    -   `atr` (Assist/Turnover Ratio): Conditionally calculated as `ast / tov` if both 'ast' and 'tov' columns are present. If 'tov' or 'ast' is missing (e.g., due to previous feature dropping), `atr` is set to 0 to avoid errors.
    -   `rpm` (Rebounds Per Minute): `(oreb + dreb) / min`
-   **Rationale**: These features were engineered to provide more meaningful insights into player performance by combining existing raw statistics. They represent efficiency metrics that are often more indicative of a player's true impact than their raw counts:
    -   `atr`: A higher ratio indicates a player's ability to facilitate scoring opportunities while minimizing ball-handling mistakes. The conditional calculation ensures robustness even if base columns are removed.
    -   `rpm`: This normalizes a player's rebounding ability by their playing time, giving a clearer picture of their rebounding efficiency.

### 4. Handling Null Values
-   **Action**: We checked for and handled any null values in the feature sets (`X_train`, `X_test`).
-   **Rationale**: Missing values can cause errors or unexpected behavior in many machine learning algorithms. Although our dataset fortunately had no missing values after previous steps, this is a crucial step to ensure data quality and model readiness.

### 5. Checking Class Balance of Target Variable
-   **Action**: We visualized the distribution of `target_5yrs` using a `countplot` and printed its value counts.
-   **Rationale**: Understanding the distribution of the target variable is crucial for classification tasks. An imbalanced class distribution (where one class significantly outnumbers the other) can lead to models that perform well on the majority class but poorly on the minority class. This initial check helps inform potential strategies for handling class imbalance, such as oversampling, undersampling, or using specific evaluation metrics.
