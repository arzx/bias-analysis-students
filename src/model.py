import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('data/exams.csv')

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define the features (X) and targets (y)
X = df.drop(columns=['math score', 'reading score', 'writing score'])
y = df[['math score', 'reading score', 'writing score']]

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred, multioutput='raw_values')

print("Mean Squared Error for each target variable on validation set:")
print(f"Math score MSE: {mse[0]:.2f}")
print(f"Reading score MSE: {mse[1]:.2f}")
print(f"Writing score MSE: {mse[2]:.2f}")

# Plotting predicted vs actual values for validation set
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.show()

# Math score plot
plot_predictions(y_val['math score'], y_val_pred[:, 0], 'Math Score: Actual vs Predicted')

# Reading score plot
plot_predictions(y_val['reading score'], y_val_pred[:, 1], 'Reading Score: Actual vs Predicted')

# Writing score plot
plot_predictions(y_val['writing score'], y_val_pred[:, 2], 'Writing Score: Actual vs Predicted')

# Test the model (optional step)
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred, multioutput='raw_values')

print("\nMean Squared Error for each target variable on test set:")
print(f"Math score MSE: {test_mse[0]:.2f}")
print(f"Reading score MSE: {test_mse[1]:.2f}")
print(f"Writing score MSE: {test_mse[2]:.2f}")