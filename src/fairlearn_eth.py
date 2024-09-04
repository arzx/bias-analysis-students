import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.reductions import GridSearch, DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data/exams.csv')

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
X = data.drop(columns=['math score', 'reading score', 'writing score'])
y = data[['math score', 'reading score', 'writing score']].sum(axis=1)

# Normalize the features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Discretize the target variable into bins (e.g., low, medium, high)
y_binned = pd.qcut(y, q=3, labels=[0, 1, 2])  # Binning into 3 categories

# Train-test split
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y_binned, data[['gender', 'race/ethnicity']], test_size=0.3, random_state=42
)

# Apply Fairlearn GridSearch with Demographic Parity
logreg = LogisticRegression(max_iter=1000)
mitigator = GridSearch(logreg, constraints=DemographicParity(), grid_size=10)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)

# Choose the best estimator
best_estimator = mitigator.best_estimator_

# Evaluate the model on the test set
y_pred_binned = best_estimator.predict(X_test)

# Calculate accuracy as an example metric
accuracy = accuracy_score(y_test, y_pred_binned)
print(f"Accuracy after Fairlearn mitigation: {accuracy:.2f}")

# Reverse the binning for prediction (optional if you want to map it back to continuous scale)
y_pred_continuous = best_estimator.predict_proba(X_test) @ y.unique()

# Analyze residuals (difference between actual and predicted scores in original scale)
y_test_continuous = pd.qcut(y_test, q=3, labels=y.unique())  # Reverse discretization for comparison
residuals = y_test_continuous - y_pred_continuous

# Add predictions and residuals to a DataFrame
results_df = pd.DataFrame({
    'gender': sensitive_test['gender'].map(label_encoders['gender'].inverse_transform),
    'race/ethnicity': sensitive_test['race/ethnicity'].map(label_encoders['race/ethnicity'].inverse_transform),
    'predicted_score': y_pred_continuous,
    'residual': residuals
})

# Plotting the results
sns.boxplot(x='gender', y='predicted_score', data=results_df)
plt.title('Predicted Scores by Gender after Bias Mitigation')
plt.show()

sns.boxplot(x='race/ethnicity', y='predicted_score', data=results_df)
plt.title('Predicted Scores by Ethnicity after Bias Mitigation')
plt.show()

sns.boxplot(x='gender', y='residual', data=results_df)
plt.title('Residuals by Gender after Bias Mitigation')
plt.show()

sns.boxplot(x='race/ethnicity', y='residual', data=results_df)
plt.title('Residuals by Ethnicity after Bias Mitigation')
plt.show()